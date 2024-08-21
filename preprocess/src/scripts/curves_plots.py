# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:35:55 2022

@author: Paolo
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import operator

def evaluate_blocking(grouping,nz_block_count,block_width):
    #get list of row_block_sizes from grouping;
    row_block_heights = []
    last_visited = -1
    for group in np.sort(np.array(grouping)):
        if group != last_visited:
            row_block_heights.append(0)
        row_block_heights[-1] += 1
        last_visited = group

    nztot = 0
    total_height = 0
    total_blocks = 0
    for count,block_height in zip(nz_block_count, row_block_heights):
        nztot += count*block_height*block_width
        total_height = count*block_height
        total_blocks += 1

    avg_height = total_height/total_blocks

    return nztot, avg_height


def isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def extract_data(filename):
    #example:
    #matrix,rows,cols,nonzeros,blocking_algo,tau,row_block_size,col_block_size,use_pattern,sim_use_groups,sim_measure,reorder,exp_name,time_to_block,VBR_nzcount,VBR_nzblocks_count,VBR_average_height,merge_counter,comparison_counter,

    with open(filename) as infile:
        complete = False
        for idx, line in enumerate(infile):
            line = line.split(",")[:-1]
            if idx == 0: 
                vars = line
            elif idx == 1:
                if len(line) != len(vars):
                    print(f"ERROR: CSV ENTRIES MISMATCH IN {filename}")
                    break
                data = {var:(float(value) if isfloat(value) else value) for var, value in zip(vars, line)}
        return data



def check_constraints(data_dict, constraints_dict):
    for constraint in constraints_dict:
        if not constraint in data_dict:
            print(f"WARNING: checking non existent property: {constraint} ")
            return False
        else:
            if constraints_dict[constraint] != data_dict[constraint]:
                return False 
    return True


def get_data_line(folder, constraints):
    datapoints = []
    for experiment in glob.glob(f"{folder}*/*.txt"):
        data = extract_data(experiment)
        if check_constraints(data,constraints):
            data["padding"] = data["VBR_nzcount"] - data["nonzeros"]
            data["density"] = data["nonzeros"]/data["VBR_nzcount"]
            #if data["blocking_algo"] == 1: #structured
            #    data["padding"] = data["VBR_nzcount"]/2 - data["nonzeros"]
            #    if data["padding"] <= 0:
            #        print("padding!", data["padding"])
            datapoints.append(data)
    return datapoints


def check_unique(l):
    return len(l) == len(set(l))

def add_curve(folder, x_name = "tau", y_name = "padding", constraints = {}, label = ""):
    datapoints = get_data_line(folder, constraints)
    x = [data[x_name] for data in datapoints]
    y = [data[y_name] for data in datapoints]
    L = sorted(zip(x,y), key=operator.itemgetter(0))
    x_s, y_s = zip(*L)
    #if (not check_unique(x_s)):
        #print("WARNING: plotting elements are not unique. Define better constraints")
    plt.plot(x_s, y_s, label = label, marker = "o")



reorder = 1;
algos = {}
#algos["basic"] = {"blocking_algo": 0, "reorder": 0, "use_pattern": 0, "sim_use_groups": 0}
#algos["pattern"] = {"blocking_algo": 0, "reorder": 0, "use_pattern": 1, "sim_use_groups":0}
#algos["pattern_group"] = {"blocking_algo": 0, "reorder": 0, "use_pattern": 1, "sim_use_groups":1}
#algos["pattern_reorder"] = {"blocking_algo": 0, "reorder": 1, "use_pattern": 1, "sim_use_groups":0}
algos["IC"] = {"blocking_algo": 0, "reorder": 1, "use_pattern": 1, "sim_use_groups":1}

algos["structured"] = {"blocking_algo": 1, "reorder": 1, "use_pattern": 1, "sim_use_groups": 1}
#algos["structured_nore"] = {"blocking_algo": 1, "reorder": reorder, "use_pattern": 0}
algos["fixed"] = {"blocking_algo": 2, "reorder": 0}
#algos["fixed_reordered"] = {"blocking_algo": 2, "reorder": 1}


#result_folder = "results/"
result_folder = "results/tinydense"
saving_folder = "images"

#for matrix_name in ["0",]:
for matrix_name in ["0","1","2","3"]:
        for block_size in [16,64,256]:
            x_name = "density"
            y_name = "VBR_average_height"

            print(f"Making {x_name} vs {y_name} image for graph {matrix_name}; block size = {block_size}")

            plt.figure()
            for algo, constraints in algos.items():
                try:
                    constraints["col_block_size"] = block_size
                    add_curve(f"{result_folder}/{matrix_name}", x_name = x_name, y_name = y_name, constraints = constraints, label = algo)
                except:
                    print(f"******missing curve for {algo}")
            savename = f"{saving_folder}/reordering_curves_mat_{matrix_name}_X_{x_name}_Y_{y_name}_b_{block_size}_.jpeg"
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.yscale("log")
            #plt.xscale("log")
            #plt.xlim(1,256)
            plt.legend()
            plt.tight_layout()
            plt.savefig(savename,  bbox_inches='tight', dpi = 300)

