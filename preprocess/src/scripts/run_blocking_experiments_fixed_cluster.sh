export MATRICES_PATH=$1
export RESULTS_PATH=$2
export PROGRAM=$3


TAUs=(0.001 0.01 0.1 0.2 0.4 0.6 0.8)
BLOCK_SIZEs=(64 128 256 512 1024)
ALGOs=(5)


USE_PATTERN=1
USE_GROUP=0
REORDERING=0
SIM=1 #0: hamming 1:jaccard; +2 for OPENMP versions

BASIC_ARGS="-P 1 -v 1 -r ${REORDERING} -m ${SIM} -p ${USE_PATTERN} -g ${USE_GROUP} -R 1 -e 1"

function create_launch {

script_body="#!/bin/bash -l
#SBATCH --job-name="${EXP_NAME}"
#SBATCH --time=00:10:00
#SBATCH --output="${RESULTS_PATH}_scripts/${EXP_NAME}".%j.o
#SBATCH --error="${RESULTS_PATH}_scripts/${EXP_NAME}".%j.e
#SBATCH --account=flavio.vella
#SBATCH --ntasks=1
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

./${PROGRAM} ${BASIC_ARGS} ${ARGS}

sleep 1s
"
script_name=___tmp_script_${EXP_NAME}
script_folder=${RESULTS_PATH}_scripts

if [[ -f "${script_folder}/${script_name}" ]]; then    
	echo "experiment exists already"
else
	echo "${script_body}" > ${script_folder}/${script_name}.sbatch
	sbatch ${script_folder}/${script_name}.sbatch
fi
}

mkdir ${RESULTS_PATH}
mkdir ${RESULTS_PATH}_scripts

for fullpath in ${MATRICES_PATH}/*.*; do
	MATRIX_FILE=$(basename -- "${fullpath}")
	MATRIX_NAME="${MATRIX_FILE%.*}"
	echo "============= processing matrix ${MATRIX_NAME}"
	MATRIX_FOLDER=${RESULTS_PATH}/${MATRIX_NAME}
	mkdir ${MATRIX_FOLDER}
	for block in ${BLOCK_SIZEs[@]}; do
		B=${block}
		b=${block}
		for a in ${ALGOs[@]}; do
			for t in ${TAUs[@]}; do
				export EXP_NAME="blocking_G_${MATRIX_NAME}_b_${b}_B_${B}_a_${a}_m_${SIM}_t_${t}_p_${USE_PATTERN}_g_${USE_GROUP}_r_${REORDERING}_F_1"
				OUTFILE=${MATRIX_FOLDER}/${EXP_NAME}.txt
				if [[ -f "${OUTFILE}" ]]; 
				then
					echo "FILE ${OUTFILE} ALREADY EXISTS. SKIPPING"
				else
					export ARGS="-f ${fullpath} -b ${b} -B ${B} -t ${t} -F 1 -a ${a} -o ${OUTFILE} -n ${EXP_NAME}"
					echo "running ./${PROGRAM} ${ARGS} ${BASIC_ARGS}"
					create_launch
				fi
			done
		done

		export a=2
		export t=0
		export EXP_NAME="blocking_G_${MATRIX_NAME}_b_${b}_B_${B}_a_${a}_t_${t}_p_${p}_g_${g}_r_${r}_F_1"
		OUTFILE=${MATRIX_FOLDER}/${EXP_NAME}.txt
		if [[ -f "${OUTFILE}" ]]; 
		then
			echo "FILE ${OUTFILE} ALREADY EXISTS. SKIPPING"
		else
			export ARGS="-f ${fullpath} -b ${b} -B ${B} -t ${t} -F 1 -a ${a} -o ${OUTFILE} -n ${EXP_NAME}"
			echo "running ./${PROGRAM} ${ARGS} ${BASIC_ARGS}"
			create_launch
		fi
	done
done
