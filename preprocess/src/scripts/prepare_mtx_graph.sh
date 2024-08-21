MATRIX=${1}
OUTPUT=${2}

$ tail -n +4
sort -k1n -k2n ${OUTPUT} > ${OUTPUT}
