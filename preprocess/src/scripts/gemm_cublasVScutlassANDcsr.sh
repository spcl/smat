if [[ $# < 2 ]]
then
    echo "usage: gemm_cublasVScutlassANDcsr.sh MATRICES_PATH RESULTS_PATH"
else

MATRICES_PATH=$1
RESULTS_PATH=$2

B_COLs=(1024 8192)
EXPERIMENTs=("CSR" "CUBLAS_GEMM" "CUTLAS_GEMM")

for fullpath in ${MATRICES_PATH}/*.*
do

    MATRIX_FILE=$(basename -- "${fullpath}")
    MATRIX_NAME="${MATRIX_FILE%.*}"

    MATRIX_FOLDER=${RESULTS_PATH}/${MATRIX_NAME}
    mkdir -p ${MATRIX_FOLDER} 2>/dev/null
    for b_cols in ${B_COLs[@]}
    do
        for exp in ${EXPERIMENTs[@]}
        do

            EXP_NAME="blocking_G_${MATRIX_NAME}_bcols_${b_cols}_e_${exp}"
            OUTFILE=${RESULTS_PATH}/${EXP_NAME}.out

            if [[ -f "${OUTFILE}" ]]
            then
                echo "FILE ${OUTFILE} ALREADY EXISTS. SKIPPING"
            else
#                 echo "sbatch batch/${exp} ${fullpath} 0 ${b_cols}"
                sbatch batch/${exp} ${fullpath} 0 ${b_cols}
            fi

        done
    done
done

fi
