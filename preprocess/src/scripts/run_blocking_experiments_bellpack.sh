export MATRICES_PATH=$1
export RESULTS_PATH=$2
export PROGRAM=$3


TAUs=(1 2 3 4 5 6 7 8 9 10 16 32 64 128 256)
BLOCK_SIZEs=(32 64 128 256 512)
ALGOs=(5)


USE_PATTERN=1
USE_GROUP=0
REORDERING=0
SIM=0 #0: hamming 1:jaccard; +2 for OPENMP versions

BASIC_ARGS="-P 1 -v 1 -r ${REORDERING} -m ${SIM} -p ${USE_PATTERN} -g ${USE_GROUP}"


mkdir ${RESULTS_PATH}

for fullpath in ${MATRICES_PATH}/*.el; do
	MATRIX_FILE=$(basename -- "${fullpath}")
	MATRIX_NAME="${MATRIX_FILE%.*}"
	echo "============= processing matrix ${MATRIX_NAME}"
	MATRIX_FOLDER=${RESULTS_PATH}/${MATRIX_NAME}
	mkdir ${MATRIX_FOLDER}
	for b in ${BLOCK_SIZEs[@]}; do
		for a in ${ALGOs[@]}; do
			for t in ${TAUs[@]}; do
				export B=${b}
				export EXP_NAME="blocking_G_${MATRIX_NAME}_b_${b}_B_${B}_a_${a}_m_${SIM}_t_${t}_p_${USE_PATTERN}_g_${USE_GROUP}_r_${REORDERING}_F_1"
				OUTFILE=${MATRIX_FOLDER}/${EXP_NAME}.txt
				if [[ -f "${OUTFILE}" ]]; 
				then
					echo "FILE ${OUTFILE} ALREADY EXISTS. SKIPPING"
				else
					export ARGS="-f ${fullpath} -b ${b} -B ${B} -t ${t} -F 1 -a ${a} -o ${OUTFILE} -n ${EXP_NAME}"
					echo "running ./${PROGRAM} ${ARGS} ${BASIC_ARGS}"
					./${PROGRAM} ${ARGS} ${BASIC_ARGS}
				fi
			done
		done
		export a=2
		export t=0
		export B=${b}
		export EXP_NAME="blocking_G_${MATRIX_NAME}_b_${b}_B_${B}_a_${a}_t_${t}_p_${p}_g_${g}_r_${r}_F_1"
		OUTFILE=${MATRIX_FOLDER}/${EXP_NAME}.txt
		if [[ -f "${OUTFILE}" ]]; 
		then
			echo "FILE ${OUTFILE} ALREADY EXISTS. SKIPPING"
		else
			export ARGS="-f ${fullpath} -b ${b} -B ${B} -t ${t} -F 1 -a ${a} -o ${OUTFILE} -n ${EXP_NAME}"
			echo "running ./${PROGRAM} ${ARGS} ${BASIC_ARGS}"
			./${PROGRAM} ${ARGS} ${BASIC_ARGS}
		fi
	done
done
