export MATRICES_PATH=$1
export RESULTS_PATH=$2
export PROGRAM=$3

function create_launch {

script_body="#!/bin/bash -l
#SBATCH --job-name="${EXP_NAME}"
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --output="${RESULTS_PATH}/scripts/outputs/${EXP_NAME}".%j.o
#SBATCH --error="${RESULTS_PATH}/scripts/errors/${EXP_NAME}".%j.e
#SBATCH --account="g34"
#SBATCH --partition=normal
#SBATCH --constraint=ssd

./${PROGRAM} ${ARGS}

sleep 1s
"
script_name=___tmp_script_${EXP_NAME}
script_folder=${RESULTS_PATH}/scripts

if [[ -f "${script_folder}/${script_name}" ]]; then    
	echo "experiment exists already"
else
	echo "${script_body}" > ${script_folder}/${script_name}.sbatch
	sbatch ${script_folder}/${script_name}.sbatch
fi
}


TAUs=(0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5)
USE_PATTERN=(0 1)
USE_GROUP=(0)
BLOCK_SIZEs=(16 64 256)
ROW_BLOCK_SIZEs=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
ALGOs=(0 1 2)
REORDERINGs=(1) #pre-reordering

mkdir ${RESULTS_PATH}/scripts
mkdir ${RESULTS_PATH}/scripts/outputs
mkdir ${RESULTS_PATH}/scripts/errors


for fullpath in ${MATRICES_PATH}/*.el; do
	MATRIX_FILE=$(basename -- "${fullpath}")
	MATRIX_NAME="${MATRIX_FILE%.*}"
	echo "============= processing matrix ${MATRIX_NAME}"
	MATRIX_FOLDER=${RESULTS_PATH}/${MATRIX_NAME}
	mkdir ${MATRIX_FOLDER}
	for b in ${BLOCK_SIZEs[@]}; do
		for r in ${REORDERINGs[@]}; do
			for a in ${ALGOs[@]}; do
				if [ ${a} -eq 2 ]; then
					for B in ${ROW_BLOCK_SIZEs[@]};do
						export EXP_NAME="blocking_G_${MATRIX_NAME}_b_${b}_a_${a}_B_${B}_r_${r}"
						OUTFILE=${MATRIX_FOLDER}/${EXP_NAME}.txt
						if [[ -f "${OUTFILE}" ]]; 
						then
							echo "FILE ${OUTFILE} ALREADY EXISTS. SKIPPING"
						else
						export ARGS="-f ${fullpath} -b ${b} -a ${a} -B ${B} -r 0 -v 1 -o ${OUTFILE} -P 1 -n ${EXP_NAME}"
						create_launch
						fi
					done
				elif [ ${a} -eq 0 ]; then
					for t in ${TAUs[@]}; do
						for p in ${USE_PATTERN[@]}; do
							for g in ${USE_GROUP[@]}; do
								export EXP_NAME="blocking_G_${MATRIX_NAME}_b_${b}_a_${a}_t_${t}_p_${p}_g_${g}_r_${r}"
								OUTFILE=${MATRIX_FOLDER}/${EXP_NAME}.txt
								if [[ -f "${OUTFILE}" ]]; 
								then
									echo "FILE ${OUTFILE} ALREADY EXISTS. SKIPPING"
								else
									export ARGS="-f ${fullpath} -b ${b} -t ${t} -a ${a} -r ${r} -p ${p} -g ${g} -v 1 -o ${OUTFILE} -P 1 -n ${EXP_NAME}"
									create_launch
								fi
							done
						done
					done
				else
					for t in ${TAUs[@]}; do
						for g in ${USE_GROUP[@]}; do
							export EXP_NAME="blocking_G_${MATRIX_NAME}_b_${b}_a_${a}_t_${t}_p_${p}_g_${g}_r_${r}"
							OUTFILE=${MATRIX_FOLDER}/${EXP_NAME}.txt
							if [[ -f "${OUTFILE}" ]]; 
							then
								echo "FILE ${OUTFILE} ALREADY EXISTS. SKIPPING"
							else
								export ARGS="-f ${fullpath} -b ${b} -t ${t} -a ${a} -r ${r} -p ${p} -g ${g} -v 1 -o ${OUTFILE} -P 1 -n ${EXP_NAME}"
								create_launch
							fi
						done
					done
				fi
			done
		done
	done
done
