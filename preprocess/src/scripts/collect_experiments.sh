SOURCES=("$@")
outputfile=${SOURCES[0]}
unset SOURCES[0]


first=0


first_folder=${SOURCES[1]}
for csv_file in ${first_folder}/*/*.*; do
	line=$(sed -n '1 p' "${csv_file}" | cut -d',' -f1)
	if [ "$line" == "matrix" ]
	then
		sed -n '1 p' "${csv_file}" > ${outputfile}
		break
	fi
done

for sou_folder in ${SOURCES[@]}; do
	for csv_file in ${sou_folder}/*/*.*; do
		line=$(sed -n '1 p' "${csv_file}" | cut -d',' -f1)
		if [ "$line" == "matrix" ]		
		then
			sed -n '2 p' "${csv_file}" >> "${outputfile}"
		fi
	done
done




