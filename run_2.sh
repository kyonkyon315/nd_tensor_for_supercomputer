#!/bin/bash
#============ Slurm Options ===========
#SBATCH -p gr20001a
#SBATCH -t 00:10:00
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH --rsc p=1000:t=1:c=1:m=2G

#============ Shell Script ============
# 1. 環境初期化
. /usr/share/Modules/init/bash

# モジュールパスの追加とロード
module use /opt/system/app/env/intel/Compiler
module use /opt/system/app/env/intel/MPI
module purge
module load intel
module load intelmpi

mpiicpc -O3 -std=c++17 -Wl,-rpath=${I_MPI_ROOT}/libfabric/lib test_mpi_omp.cpp -o test_mpi_omp


# 2. 通信設定 【ここを修正】
# OFI (Omni-Path/InfiniBand) を諦めて、汎用TCPを使います。
# これでドライバが見つからないエラーは消えます。
# export I_MPI_FABRICS=shm:tcp

# 3. 実行
# srun に任せて128プロセス起動
srun --mpi=pmi2 -n 1000 ./test_mpi_omp