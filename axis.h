#ifndef AXIS_H
#define AXIS_H

template<int LABEL,int LENGTH,int L_GHOST=3,int R_GHOST=3>
class Axis {
    private:
    int val;
    public:
    static constexpr int label = LABEL;
    static constexpr int num_grid = LENGTH;
    static constexpr int L_ghost_length = L_GHOST;
    static constexpr int R_ghost_length = R_GHOST;
    int operator=(int r){
        return val=r;
    }
    int operator()(){
        return val;
    }
};

#endif //AXIS_H