#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

rm -rf log ncu && mkdir -p log ncu

handle_error() {
    echo "An error occurred, but the script will continue."
}
trap 'handle_error' ERR
# $1: M. $2: N, $3: K, $4=N_mult (N_mult * MMA_N) $5: cop20k_A.mtx
evaluate_hgemm() {
    echo "Evaluating $1 * $2 * $3 n_mult $4 file $5"
    $WORK_PATH/output/bin/hgemm -M=$1 -N=$2 -K=$3 -enable_wmma=true -enable_mma=true -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false -n_mult=$4 -filename=$5  > log/hgemm_${1}_${2}_${3}.log 2>&1
    sleep 3
}

# $1: M. $2: N, $3: K
ncu_hgemm() {
    echo "NCU $1 * $2 * $3"
    sudo ncu --set full --target-processes all --force-overwrite -o ncu/hgemm_${1}_${2}_${3} $WORK_PATH/output/bin/hgemm -M=$1 -N=$2 -K=$3 -enable_wmma=true -enable_mma=true -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_hgemm_${1}_${2}_${3}.log 2>&1
    sleep 3
}

benchmark_hgemm() {
    #dims=(256 512 768 1024 1536 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384)
    dims=(512)
    # M == N == K
    M=512
    i=1
    path="<path>/matrices/band_matrices_4_times"
    for file in  "$path"/*; do
       evaluate_hgemm $M $M $M $i $file
    done


    i=1
    path="<path>/matrices/suitesparse"
    for file in  "$path"/*; do
       evaluate_hgemm $M $M $M $i $file
    done
}

benchmark_hgemm