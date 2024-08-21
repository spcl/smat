

MATRIX_PATH=$1
mkdir $MATRIX_PATH


nmin=10000
nmax=200000
increment=10000

dmin=10000
dmax=100

array=()

for ((i=nmin; i<=nmax; i+=increment)); do
	nmin_tmp=$i
	nmax_tmp=$((i+increment))
	nzmin=$((nmin_tmp*nmin_tmp/dmin))
	nzmax=$((nmax_tmp*nmax_tmp/dmax))
	ssget_line="[ @cols -ge ${nmin_tmp} ] && [ @cols -le ${nmax_tmp} ] && [ @rows -ge ${nmin_tmp} ] && [ @rows -le ${nmax_tmp} ] && [ @nonzeros -ge ${nzmin} ] && [ @nonzeros -le ${nzmax} ]"
	tmp_array=($(ssget -s "${ssget_line}"))
	array+=("${tmp_array[@]}")
done

for id in "${array[@]}";do
	filename="$(ssget -f -i $id -e -t MM)"
	basefile="$(basename -- $filename)" || echo "invalid name?"
	if [[ ! -f "${MATRIX_PATH}/${basefile}" ]]
	then
		cp "$filename" "${MATRIX_PATH}/${basefile}"
		#ssget -i $id -r
		echo "DOWNLOADED $basefile $filename"
	else
		echo "${basefile} EXISTS ALREADY"
	fi
done
