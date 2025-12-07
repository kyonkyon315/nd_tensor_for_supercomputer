#include <iostream>
#include <array>
#include <mpi.h>
#include "axis.h"
#include "n_d_tensor_with_ghost_cell_mpi.h"

int main(int argc, char** argv) {
    std::cerr << "Debug: Before MPI_Init" << std::endl;

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // ランク0だけが喋る
    if(rank == 0) {
        std::cerr << "Debug: MPI_Init success. Rank 0 started." << std::endl;
    }

    // X軸: ローカル100グリッド, 袖3
    using X_Axis = Axis<0, 1000, 3, 3>;
    // Y軸: ローカル100グリッド, 袖3
    using Y_Axis = Axis<1, 1000, 3, 3>;
    
    // プロセス分割 (2x2=4プロセス必須)
    std::array<int, 2> dims = {100, 10};

    try {
        if(rank == 0) std::cout << "[Info] Creating Tensor..." << std::endl;
        
        auto tensor = make_mpi_tensor<double>(
            MPI_COMM_WORLD, dims, X_Axis{}, Y_Axis{}
        );

        // 時間計測用の変数を準備
        double start_time, end_time;

        // 時間発展ループ
        for(int step=0; step<5; ++step) {

            // 1. 計測開始
            // 全プロセスの足並みを揃えてからタイマーをスタートさせます
            MPI_Barrier(MPI_COMM_WORLD); 
            start_time = MPI_Wtime();
            if(rank == 0) std::cout << "Starting Step " << step << "..." << std::endl;

            // 1. 通信 (袖領域の同期)
            tensor.sync_ghosts();
            
            // 2. 計算 (物理領域のみ更新)
            // FullSlice は「物理領域 0 ~ 99」を指します
            tensor.set_value_sliced<FullSlice, FullSlice>(
                [&](int x, int y) {
                    // ゴースト領域(マイナス座標や100以上)も参照可能
                    // 例: 拡散などの計算
                    return tensor.at(x, y) + 1.0; 
                }
            );
            
            // 2. 計測終了
            // 全プロセスが計算を終えるのを待ってからタイマーを止めます
            // (これをしないと、Rank0だけ早く終わった場合に間違った時間が表示されます)
            MPI_Barrier(MPI_COMM_WORLD);
            end_time = MPI_Wtime();

            // 3. 結果表示 (Rank 0のみ)
            if(rank == 0) {
                double elapsed = end_time - start_time;
                std::cout << "Step " << step << " done. "
                          << "Time: " << elapsed << " sec" << std::endl;
            }
        }

        if(rank == 0) std::cout << "All Steps finished successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " Exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}