export MATRICES_PATH=$1
export RESULTS_PATH=$2
export PROGRAM=$3


TAUs=(0.9 0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001)
USE_PATTERN=(1)
USE_GROUP=(0)
BLOCK_SIZEs=(32)
ROW_BLOCK_SIZEs=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
ALGOs=(2 3)
REORDERINGs=(0 1) #pre-reordering

SIM=1 #0: hamming 1:jaccard; +2 for OPENMP versions

BASIC_ARGS="-P 1 -v 1"


mkdir ${RESULTS_PATH}

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
						export ARGS="-f ${fullpath} -b ${b} -a ${a} -B ${B} -r 0 -o ${OUTFILE} -n ${EXP_NAME}"
						echo "running ./${PROGRAM} ${ARGS} ${BASIC_ARGS}"
						./${PROGRAM} ${ARGS} ${BASIC_ARGS}
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
									export ARGS="-f ${fullpath} -b ${b} -t ${t} -a ${a} -r ${r} -p ${p} -g ${g} -o ${OUTFILE} -n ${EXP_NAME} -m ${SIM}"
									echo "running ./${PROGRAM} ${ARGS} ${BASIC_ARGS}"
									./${PROGRAM} ${ARGS} ${BASIC_ARGS}
								fi
							done
						done
					done
				else
					for t in ${TAUs[@]}; do
						for g in ${USE_GROUP[@]}; do
							p=1
							g=1
							export EXP_NAME="blocking_G_${MATRIX_NAME}_b_${b}_a_${a}_t_${t}_p_${p}_g_${g}_r_${r}"
							OUTFILE=${MATRIX_FOLDER}/${EXP_NAME}.txt
							if [[ -f "${OUTFILE}" ]]; 
							then
								echo "FILE ${OUTFILE} ALREADY EXISTS. SKIPPING"
							else
								export ARGS="-f ${fullpath} -b ${b} -t ${t} -a ${a} -r ${r} -p ${p} -g ${g} -o ${OUTFILE} -n ${EXP_NAME} -m ${SIM}"
								echo "running ./${PROGRAM} ${ARGS} ${BASIC_ARGS}"
								./${PROGRAM} ${ARGS} ${BASIC_ARGS}
							fi
						done
					done
				fi
			done
		done
	done
done
