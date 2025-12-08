#ifndef N_D_TENSOR_WITH_GHOST_CELL_MPI_H
#define N_D_TENSOR_WITH_GHOST_CELL_MPI_H

#include <iostream>
#include <vector>
#include <array>
#include <tuple>
#include <type_traits>
#include <mpi.h>

using Index = int;

// --- Slice Helper ---
struct FullSlice {};

template<Index START, Index END>
struct Slice {
    static constexpr Index START_val = START;
    static constexpr Index END_val = END;
};

// --- Tensor Class ---
template<typename T, typename... Axes>
class NdTensorWithGhostCell {
public:
    static constexpr int N_dim = sizeof...(Axes);
    static constexpr std::array<Index, N_dim> local_shape = {Axes::num_grid...};
    static constexpr std::array<Index, N_dim> L_ghost_lengths = {Axes::L_ghost_length...};
    static constexpr std::array<Index, N_dim> R_ghost_lengths = {Axes::R_ghost_length...};

    static constexpr std::array<Index, N_dim> data_shape = [](){
        std::array<Index, N_dim> s = {};
        for(int i=0; i<N_dim; ++i){
            s[i] = local_shape[i] + L_ghost_lengths[i] + R_ghost_lengths[i];
        }
        return s;
    }();

    static_assert(
        []()constexpr{
            for(int i=0;i<N_dim;++i){
                if(L_ghost_lengths[i]!=R_ghost_lengths[i])return false;
            }
            return true;
        }(),
        "ghost cell size in L and R mismatch."
    );

private:
    static constexpr Index total_size = []() constexpr {
        Index prod = 1;
        for (auto s : data_shape) prod *= s;
        return prod;
    }();

    std::vector<T> data;
    mutable std::vector<T> send_buf, recv_buf;

    static constexpr std::array<Index, N_dim> strides = []() constexpr {
        std::array<Index, N_dim> s = {};
        Index current_stride = 1;
        for (int i = N_dim - 1; i >= 0; --i) {
            s[i] = current_stride;
            current_stride *= data_shape[i];
        }
        return s;
    }();

    static constexpr Index offset = []() constexpr {
        Index ret_val = 0;
        for(int i=0; i<N_dim; ++i){
            ret_val += strides[i] * L_ghost_lengths[i];
        }
        return ret_val;
    }();

    template<size_t Depth>
    static constexpr void calc_buf_size(Index& buf_size,Index d_start, Index d_end, Index current_dim_idx) {
        if constexpr (Depth == N_dim) {
        } else {
            Index start = 0; 
            Index end = local_shape[Depth]; 

            if (Depth == current_dim_idx) {
                start = d_start;
                end = d_end;
            }
            buf_size*=(end-start);
            calc_buf_size<Depth + 1>(buf_size, d_start, d_end, current_dim_idx);
        }
    }

    static constexpr Index buf_size = []()constexpr{
        Index ans=0;
        for(Index d = 0;d<N_dim;++d){
            Index buf_size_in_one_dim = 1;
            calc_buf_size<0>(
                buf_size_in_one_dim,
                local_shape[d] - R_ghost_lengths[d], 
                local_shape[d], 
                d
            );
            ans+=buf_size_in_one_dim;
        }
        return ans;
    }();

    // MPI Members
    MPI_Comm comm_cart;
    int my_rank;
    std::array<int, N_dim> proc_dims;
    std::array<std::array<int, 2>, N_dim> neighbors; 

public:
    NdTensorWithGhostCell(MPI_Comm comm, const std::array<int, N_dim>& procs_per_dim) {
        data.resize(total_size, T{0});
        recv_buf.resize(buf_size,T{0});
        send_buf.resize(buf_size,T{0});
        setup_mpi(comm, procs_per_dim);
    }
    
    ~NdTensorWithGhostCell() {
        if(comm_cart != MPI_COMM_NULL) {
            MPI_Comm_free(&comm_cart);
        }
    }

private:
    void setup_mpi(MPI_Comm comm, const std::array<int, N_dim>& dims_array) {
        proc_dims = dims_array;
        int dims[N_dim];
        int periods[N_dim];
        for(int i=0; i<N_dim; ++i) {
            dims[i] = proc_dims[i];
            periods[i] = 0; 
        }

        int ierr = MPI_Cart_create(comm, N_dim, dims, periods, 1, &comm_cart);
        if (ierr != MPI_SUCCESS) {
            MPI_Abort(comm, 1);
        }
        MPI_Comm_rank(comm_cart, &my_rank);

        for(int i=0; i<N_dim; ++i) {
            MPI_Cart_shift(comm_cart, i, 1, &neighbors[i][0], &neighbors[i][1]);
        }
    }

    // --- Index Helper ---
    template<size_t I = 0, typename... IdT>
    constexpr int flatten_index_helper(Index i, IdT... rest) const noexcept {
        if constexpr (sizeof...(rest) == 0) {
            return i * strides[I] + offset; 
        } else {
            return i * strides[I] + flatten_index_helper<I + 1>(rest...);
        }
    }

    template<typename... Idx>
    constexpr int flatten_index(Idx... indices) const noexcept {
        return flatten_index_helper(indices...);
    }

    // --- Pack/Unpack Helper ---
    template<typename BufT, bool IS_PACKING, size_t Depth, typename... Idx>
    void pack_unpack_helper(BufT& buf, size_t& buf_idx, Index d_start, Index d_end, Index current_dim_idx, Idx... indices) {
        if constexpr (Depth == N_dim) {
            if constexpr (IS_PACKING) {
                buf[buf_idx++] = this->at(indices...);
            } else {
                this->at(indices...) = buf[buf_idx++];
            }
        } else {
            Index start = 0; 
            Index end = local_shape[Depth]; 

            if (Depth == current_dim_idx) {
                start = d_start;
                end = d_end;
            }
            for (Index i = start; i < end; ++i) {
                pack_unpack_helper<BufT, IS_PACKING, Depth + 1>(buf, buf_idx, d_start, d_end, current_dim_idx, indices..., i);
            }
        }
    }

    // --- Set Value Helper ---
    template<size_t Depth, typename Func, typename... Slices, typename... Idx>
    void set_value_sliced_helper(Func func, Idx... indices) {
        if constexpr (Depth == N_dim) {
            this->at(indices...) = func(indices...);
        } else {
            using CurrentSlice = std::tuple_element_t<Depth, std::tuple<Slices...>>;
            
            // Default: Physical region [0, local_shape)
            Index start_idx = 0;
            Index end_idx   = local_shape[Depth];

            if constexpr (!std::is_same_v<CurrentSlice, FullSlice>) {
                constexpr Index req_start = CurrentSlice::START_val;
                constexpr Index req_end   = CurrentSlice::END_val;
                constexpr Index min_bound = -L_ghost_lengths[Depth];
                constexpr Index max_bound = local_shape[Depth] + R_ghost_lengths[Depth];
                start_idx = (req_start > min_bound) ? req_start : min_bound;
                end_idx   = (req_end < max_bound) ? req_end : max_bound;
            }

            for (Index i = start_idx; i < end_idx; ++i) {
                set_value_sliced_helper<Depth+1, Func, Slices...>(func, indices..., i);
            }
        }
    }

public:
    // Accessors
    template<typename... Idx>
    inline T& at(Idx... indices) noexcept {
        return data[flatten_index(indices...)];
    }
    template<typename... Idx>
    inline const T& at(Idx... indices) const noexcept {
        return data[flatten_index(indices...)];
    }

    template<typename... Slices, typename Func>
    void set_value_sliced(Func func) {
        set_value_sliced_helper<0, Func, Slices...>(func);
    }

    // --- Sync Ghosts ---
    void sync_ghosts() {
        //if(send_buf.size() < total_size) send_buf.resize(total_size);
        //if(recv_buf.size() < total_size) recv_buf.resize(total_size);

        for (int d = 0; d < N_dim; ++d) {
            int left_rank = neighbors[d][0];
            int right_rank = neighbors[d][1];

            // 1. Shift Right
            {
                size_t idx = 0;
                pack_unpack_helper<std::vector<T>, true, 0>(
                    send_buf, idx, 
                    local_shape[d] - R_ghost_lengths[d], local_shape[d], 
                    d
                );
                MPI_Sendrecv(
                    send_buf.data(), idx * sizeof(T), MPI_BYTE, right_rank, 0,
                    recv_buf.data(), idx * sizeof(T), MPI_BYTE, left_rank, 0,
                    comm_cart, MPI_STATUS_IGNORE
                );
                if (left_rank != MPI_PROC_NULL) {
                    size_t u_idx = 0;
                    pack_unpack_helper<std::vector<T>, false, 0>(
                        recv_buf, u_idx, -L_ghost_lengths[d], 0, d
                    );
                }
            }
            // 2. Shift Left
            {
                size_t idx = 0;
                pack_unpack_helper<std::vector<T>, true, 0>(
                    send_buf, idx, 
                    0, L_ghost_lengths[d], 
                    d 
                );
                MPI_Sendrecv(
                    send_buf.data(), idx * sizeof(T), MPI_BYTE, left_rank, 1,
                    recv_buf.data(), idx * sizeof(T), MPI_BYTE, right_rank, 1,
                    comm_cart, MPI_STATUS_IGNORE
                );
                if (right_rank != MPI_PROC_NULL) {
                    size_t u_idx = 0;
                    pack_unpack_helper<std::vector<T>, false, 0>(
                        recv_buf, u_idx, local_shape[d], local_shape[d] + R_ghost_lengths[d], d
                    );
                }
            }
        }
        MPI_Barrier(comm_cart);
    }
    
    int get_my_rank() const { return my_rank; }
};

// Factory
template<typename T, typename... Axes>
auto make_mpi_tensor(MPI_Comm comm, const std::array<int, sizeof...(Axes)>& dims, const Axes&... axes) 
 -> NdTensorWithGhostCell<T, Axes...> {
    return NdTensorWithGhostCell<T, Axes...>(comm, dims);
}

#endif