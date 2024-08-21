import os
import sys
import csv


download_path = "./matrices/suitesparse"
os.system(f"mkdir ${download_path}")

filename = "./matrix_list.csv"

total = sum(1 for line in open(filename))
print(total)

with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for i in range(1, total):
        cur_row = next(csv_reader)
        matrix_group = f"${download_path}/" + cur_row[1]
        matrix_name = cur_row[2]
        if os.path.exists(matrix_group + "/" + matrix_name + "/" + matrix_name + ".mtx") == False:
            if os.path.exists(matrix_group) == False:
                os.system("mkdir " + matrix_group)
            matrix_url = "https://suitesparse-collection-website.herokuapp.com/MM/" + cur_row[1] + "/" + cur_row[2] + ".tar.gz"
            # matrix_url = "http://sparse-files.engr.tamu.edu/MM/" + cur_row[1] + "/" + cur_row[2] + ".tar.gz"
            # os.system("axel -n 4 " + matrix_url)
            os.system("wget " + matrix_url)
            os.system(f"mv ${matrix_name} ${download_path}")
            os.system("tar -zxvf " + f"${download_path}/${matrix_name}" + ".tar.gz " + "-C " + matrix_group + "/")
            os.system("rm -rf " + f"${download_path}/${matrix_name}" + ".tar.gz")
