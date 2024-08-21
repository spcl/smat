# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:35:55 2022

@author: Paolo
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import operator
import pandas as pd
import os as os
import seaborn as sns
import argparse



global_label_dict = {
    "block_density" : "Blocked density after DenseAMP",
    "density" : "Original density",
    "dense-amp" : "Density amplification",
    "relative_dense_amp" : "Density amplification (against natural blocking)",
    "block ratio": "Shape (height / width)",
    "speed-vs-CSR": "Speed-up against cuSparse-CSR",
    "speed-vs-VBR-no-reord": "Speed-up against BCSR (natural blocking)",
    "speed-vs-GEMM": "Speed-up against GEMM",
    "speed-vs-BELLPACK-no-reord": "Speed-up against BELLPACK (natural blocking)",
    "speed-vs-CUTLASS_GEMM": "Speed-up against CUTLASS GEMM",
    "speed-vs-CUTLASS_BELLPACK": "Speed-up against CUTLASS BELLPACK",
    "GFLOPs": "GFLOPs",
    "VBR_nzblocks_count" : "Number of nonzero blocks"
}

global_exp_dict = {}

global_exp_dict["VBR-reord"] = {
    "hatch" : "//",
    "edgecolor" : "black",
    "color" : "orange",
    "label" : "cuBLAS-BCSR (IC blocking)"

}

global_exp_dict["VBR-no-reord"] = {
    "hatch" : "//",
    "edgecolor" : "black",
    "color" : "blue",
    "label" : "cuBLAS-BCSR (natural blocking)"
}

global_exp_dict["BELLPACK-no-reord"] = {
    "hatch" : "/",
    "edgecolor" : "black",
    "color" : "green",
    "label" : "cuSparse-BELLPACK (natural blocking)"

}

global_exp_dict["BELLPACK-reord"] = {
    "hatch" : "//",
    "edgecolor" : "black",
    "color" : "green",
    "label" : "cuSparse-BELLPACK (IC blocking)"

}


global_exp_dict["GEMM"] = {
    "hatch" : "--",
    "edgecolor" : "black",
    "color" : "red",
    "label" : "cuBLAS-gemm"
}

global_exp_dict["CUTLASS_GEMM"] = {
    "hatch" : "-",
    "edgecolor" : "black",
    "color" : "red",
    "label" : "CUTLASS-gemm"
}

global_exp_dict["CSR"] = {
    "hatch" : "..",
    "edgecolor" : "black",
    "color" : "white",
    "label" : "cuSparse-CSR"

}

global_exp_dict["CUTLASS_BELLPACK"] = {
    "hatch" : "//",
    "edgecolor" : "black",
    "color" : "gray",
    "label" : "CUTLASS-BELLPACK"

}


def get_dataframe_folder(folder):
    df = pd.DataFrame()
    for matrix_folder in glob.glob(f"{folder}/*"):
        for experiment in glob.glob(f"{matrix_folder}/*"):
            try:
                df1 = pd.read_csv(experiment, skiprows = 0)
                df = pd.concat([df, df1], ignore_index=True, sort=False)
            except:
                print(f"CANNOT LOAD {experiment}")
    if df.empty:
        print("ALTERNATIVE LOAD", folder)
        for experiment in glob.glob(f"{folder}/*"):
            print(experiment)
            try:
                df1 = pd.read_csv(experiment, skiprows = 0)
                df = pd.concat([df, df1], ignore_index=True, sort=False)
            except:
                print(f"CANNOT LOAD {experiment}")
    return df        

def get_line(df, constraints):
    query = ""
    for key in constraints:
        query += f"{key} == {constraints[key]} &"
    query = query[:-2]
    return df.query(query)


def barplot(x_labels, x_ax_label, ys, y_styles = {} , y_ax_label = "", title = ""):
    plt.xlabel(x_ax_label)
    plt.ylabel(y_ax_label)
    
    bars = len(ys)
    tot_width = 0.9
    increment = tot_width/bars
    barpos = -tot_width/2
    width = increment*0.95

    for y, style in zip(ys, y_styles):
        x_pos = np.arange(barpos,len(x_labels) + barpos)
        plt.bar(x_pos,y, width = width, **style)
        barpos += increment

    plt.grid("both")
    plt.title(title)
    plt.xticks(range(len(x_labels)), x_labels, rotation=90)
 


experiments = {}
experiments["VBR-no-reord"] = {"multiplication_algo": 6, "blocking_algo": 2}
experiments["VBR-reord"] = {"multiplication_algo": 6, "blocking_algo": 5}
experiments["BELLPACK-no-reord"] = {"multiplication_algo": 3, "blocking_algo": 2}
experiments["CSR"] = {"multiplication_algo": 2, "blocking_algo": 3}
experiments["GEMM"] = {"multiplication_algo": 1, "blocking_algo": 3}
experiments["CUTLASS_GEMM"] = {"multiplication_algo":9, "blocking_algo": 3}
experiments["CUTLASS_BELLPACK"] = {"multiplication_algo":8, "blocking_algo": 2}



def make_barplot(df, image_folder,B_cols, row_block_size, col_block_size, variable = "speed-vs-cusparse"):
    tmp_df = df.loc[(df["col_block_size"]==col_block_size) & (df["row_block_size"] == row_block_size) & (df["b_cols"] == B_cols)]
    matrices_names = [val.split("/")[-1].split(".")[0] for val in tmp_df["matrix"].unique()]

    data_lines = {exp_name : [] for exp_name in experiments}
    data_lines.pop("CSR")
    if col_block_size != row_block_size: data_lines.pop("BELLPACK-no-reord")

    for matrix in tmp_df["matrix"].unique():
        for exp_name in data_lines.keys():
            M_algo, B_algo = experiments[exp_name]
            values = tmp_df.loc[(tmp_df["matrix"] == matrix) & (tmp_df["multiplication_algo"] == M_algo) & (tmp_df["blocking_algo"] == B_algo)][variable].values
            if len(values) == 0:
                data_lines[exp_name].append( 0 )
            else:
                data_lines[exp_name].append( values[0] )

    savename = f"{image_folder}/SpMM_barplot_{variable}_{row_block_size}_{col_block_size}_{B_cols}"

    styles = [global_exp_dict[exp] for exp in data_lines.keys()]


    plt.figure()
    barplot(
                x_labels = matrices_names,
                x_ax_label="graphs",
                ys=data_lines.values(),
                y_styles = styles,
                y_ax_label = global_label_dict[variable]
                )
    plt.legend()
    plt.savefig(savename + ".png",  bbox_inches='tight', dpi = 300)
    plt.close()   
    

def make_barplot_best(df, image_folder,B_cols, variable = "speed-vs-cusparse"):
    tmp_df = df.loc[df["b_cols"] == B_cols]
    tmp_df = tmp_df.loc[df["row_block_size"] == df["col_block_size"]]
    tmp_df = tmp_df.loc[tmp_df.groupby(["matrix","blocking_algo","multiplication_algo"])["avg_time_multiply"].idxmin()]
    tmp_df.sort_values(by=['density','matrix'], inplace=True)
    matrices_names = [val.split("/")[-1].split(".")[0] for val in tmp_df["matrix"].unique()]
    print(tmp_df)
    data_lines = {exp_name : [] for exp_name in experiments}
    data_lines.pop("CSR")

    for matrix in tmp_df["matrix"].unique():
        for exp_name in data_lines.keys():
            M_algo, B_algo = experiments[exp_name].values()
            values = tmp_df.loc[(tmp_df["matrix"] == matrix) & (tmp_df["multiplication_algo"] == M_algo) & (tmp_df["blocking_algo"] == B_algo)][variable].values
            if len(values) == 0:
                data_lines[exp_name].append( 0 )
            else:
                data_lines[exp_name].append( values[0] )

    savename = f"{image_folder}/SpMM_time_barplot_BEST_{B_cols}"

    styles = [global_exp_dict[exp] for exp in data_lines.keys()]


    plt.figure()
    barplot(
                x_labels = matrices_names,
                x_ax_label="Sparse matrices",
                ys=data_lines.values(),
                y_styles = styles,
                y_ax_label = global_label_dict[variable])
    plt.axhline(linewidth=2, y = 1, color = "red", label = global_exp_dict["CSR"]["label"])
    plt.legend()
    plt.savefig(savename + ".png",  bbox_inches='tight', dpi = 300)
    plt.close()   


def make_boxplot_best(df, image_folder,B_cols, exps = ("VBR-no-reord","GEMM")):
    plt.figure()
    tmp_df = df.loc[df["b_cols"] == B_cols]
    tmp_df = tmp_df.loc[df["row_block_size"] == df["col_block_size"]]
    tmp_df = tmp_df.loc[tmp_df.groupby(["matrix","blocking_algo","multiplication_algo"])["avg_time_multiply"].idxmin()]
    tmp_df.sort_values(by=['density','matrix'], inplace=True)

    variables = ["speed-vs-" + exp for exp in exps] 
    tmp_df = tmp_df[variables]
    tmp_df = tmp_df[tmp_df[variables] < 10]
    plt.axhline(y = 1, color = "red")

    ax = sns.violinplot(data = tmp_df, trim=(1,5), cut = 0)
    savename = f"{image_folder}/SpMM_time_violin_BEST_{B_cols}"
    plt.legend()
    plt.savefig(savename + ".png",  bbox_inches='tight', dpi = 300)
    plt.close()   

def make_histo_best(df, image_folder,B_cols, x_var = 'density'):
    plt.figure()

    tmp_df = df.loc[df["b_cols"] == B_cols]
    tmp_df = tmp_df.loc[df["row_block_size"] == df["col_block_size"]]
    tmp_df = tmp_df.loc[tmp_df.groupby(["matrix","blocking_algo","multiplication_algo"])["avg_time_multiply"].idxmin()]

    exps = ("VBR-reord", "CSR","BELLPACK-no-reord")
    vars = ["avg_time_multiply_" + exp for exp in exps]
    tmp_df['fastest_time'] = df[vars].min(axis=1)
    tmp_df['fastest_routine'] = df[vars].idxmin(axis=1)

    bin_num = 10  # Adjust bin size as needed
    bin_edges = np.logspace(np.log10(tmp_df[x_var].min()), np.log10(tmp_df[x_var].max()), bin_num)
    tmp_df["density_bin"] = pd.cut(tmp_df[x_var], bins=bin_edges)
    best = tmp_df.groupby("density_bin")
    best = best["fastest_routine"].value_counts()

    best = best.unstack().fillna(0)

    best["total"] = best[vars].sum(axis=1)
    print(best)

    for exp in exps: 
        var = "percent_" + exp
        best[var] = best["avg_time_multiply_" + exp]/best["total"] *100

    ax = sns.barplot(data=best.reset_index().melt(id_vars="density_bin", value_vars=[f"percent_{exp}" for exp in exps], var_name="routine"), x="density_bin", y="value", hue="routine", alpha=0.5)
    sns.despine()
    xticks = np.arange(bin_num -1) + 0.5
    xticklabels = [np.format_float_scientific(bin, precision=0, exp_digits=1) for bin in bin_edges[:-1]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    # plot histogram with one bar for each column
    
    plt.xlabel(global_label_dict[x_var])
    plt.ylabel("Fastest routine on % of the matrices")
    savename = f"{image_folder}/SpMM_hist_BEST_{B_cols}"
    plt.legend()
    plt.savefig(savename + ".png",  bbox_inches='tight', dpi = 300)
    plt.close()   


def make_histo_best_blocksize(df, image_folder,B_cols, x_var = 'density'):
    plt.figure()

    tmp_df = df.loc[(df["b_cols"] == B_cols) & (df["multiplication_algo"] == 6) & (df["blocking_algo"] == 5)]
    tmp_df = tmp_df.loc[df["row_block_size"] == df["col_block_size"]]

    idx = tmp_df.groupby('matrix')['avg_time_multiply'].idxmin()
    result = tmp_df.loc[idx, "row_block_size"]
    exps = ("VBR-reord", "CSR","BELLPACK-no-reord")
    vars = ["avg_time_multiply_" + exp for exp in exps]
    tmp_df['fastest_time'] = df[vars].min(axis=1)
    tmp_df['fastest_routine'] = df[vars].idxmin(axis=1)

    bin_num = 10  # Adjust bin size as needed
    bin_edges = np.logspace(np.log10(tmp_df[x_var].min()), np.log10(tmp_df[x_var].max()), bin_num)
    tmp_df["density_bin"] = pd.cut(tmp_df[x_var], bins=bin_edges)
    best = tmp_df.groupby("density_bin")
    best = best["fastest_routine"].value_counts()

    best = best.unstack().fillna(0)

    best["total"] = best[vars].sum(axis=1)
    print(best)

    for exp in exps: 
        var = "percent_" + exp
        best[var] = best["avg_time_multiply_" + exp]/best["total"] *100

    ax = sns.barplot(data=best.reset_index().melt(id_vars="density_bin", value_vars=[f"percent_{exp}" for exp in exps], var_name="routine"), x="density_bin", y="value", hue="routine", alpha=0.5)
    sns.despine()
    xticks = np.arange(bin_num -1) + 0.5
    xticklabels = [np.format_float_scientific(bin, precision=0, exp_digits=1) for bin in bin_edges[:-1]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    # plot histogram with one bar for each column
    
    plt.xlabel(global_label_dict[x_var])
    plt.ylabel("Fastest routine on % of the matrices")
    savename = f"{image_folder}/SpMM_hist_BEST_{B_cols}"
    plt.legend()
    plt.savefig(savename + ".png",  bbox_inches='tight', dpi = 300)
    plt.close()   

def make_heatmap(df,image_folder,B_cols,exp_name, colormap_variable = "speed-vs-cusparse"):     
    M_algo, B_algo = experiments[exp_name]
    heatmap_df = df[(df["multiplication_algo"]==M_algo) & (df["blocking_algo"]== B_algo) & (df["b_cols"] == B_cols)]
    table = heatmap_df.pivot_table(index="row_block_size", columns="col_block_size", values=colormap_variable, aggfunc='mean')
    table = table.sort_values(by=['row_block_size'], ascending=False)
    if colormap_variable in global_label_dict:
        c_label = global_label_dict[colormap_variable]
    else:
        c_label = colormap_variable
    sns.heatmap(table,annot = True, cbar_kws={'label': c_label})
    plt.ylabel("block height")
    plt.xlabel("block width")
    plt.savefig(f"{image_folder}/{exp_name}_heatmap_{colormap_variable}_{B_cols}.png",  bbox_inches='tight', dpi = 300)
    plt.close()


def make_scatter_block_size(df,B_cols = 1024,var_x = "dense-amp", var_y = "speed-vs-VBR-no-reord", col_block_size = 1024, row_block_size = 1024):
    plt.figure(figsize=(8,2))
    tmp_df = df.loc[(df["b_cols"] == B_cols) & (df["multiplication_algo"] == 6) & (df["blocking_algo"] == 5) & (df["col_block_size"]==col_block_size) & (df["row_block_size"] == row_block_size)]
    print("SCATTER!", tmp_df)
    print(np.max(tmp_df["dense-amp"]))
    print(tmp_df.head())
    ax = sns.scatterplot(data=tmp_df, x=var_x, y=var_y)
    ax.axhline(y=1, color='red',alpha = 0.7)
    plt.ylim(ymin=0)

    #point = (0.95,1)
    #transformed_point = ax.transAxes.transform(point)
    #transformed_point_data = ax.transData.inverted().transform(transformed_point)
    #ax.text(transformed_point_data[0], 1.05, 'density amplification resulted in speed-up', ha='center', va='bottom', transform=ax.transData)
    #ax.text(transformed_point_data[0], 0.95, 'density amplification resulted in slow-down', ha='center', va='top', transform=ax.transData)

    plt.xlabel(global_label_dict[var_x])
    plt.ylabel(global_label_dict[var_y])
    plt.savefig(f"{image_folder}/scatter_plot_{var_x}_vs_{var_y}_B_cols_{B_cols}_b_{col_block_size}_B_{row_block_size}",  bbox_inches='tight', dpi = 300)
    plt.close()


def make_scatter(df,B_cols,var_x = "dense-amp", var_y = "speed-vs-VBR-no-reord"):
    plt.figure(figsize=(8,4))
    tmp_df = df.loc[(df["b_cols"] == B_cols) & (df["multiplication_algo"] == 6) & (df["blocking_algo"] == 5)]
    print(tmp_df.columns)
    #print("SCATTER!", tmp_df)
    ax = sns.scatterplot(data=tmp_df, x=var_x, y=var_y, hue = "row_block_size")
    ax.axhline(y=1, color='red',alpha = 0.7)

    #point = (0.95,1)
    #transformed_point = ax.transAxes.transform(point)
    #transformed_point_data = ax.transData.inverted().transform(transformed_point)
    #ax.text(transformed_point_data[0], 1.05, 'density amplification resulted in speed-up', ha='center', va='bottom', transform=ax.transData)
    #ax.text(transformed_point_data[0], 0.95, 'density amplification resulted in slow-down', ha='center', va='top', transform=ax.transData)

    plt.xlabel(global_label_dict[var_x])
    plt.ylabel(global_label_dict[var_y])
    plt.savefig(f"{image_folder}/scatter_plot_{var_x}_vs_{var_y}_B_cols_{B_cols}",  bbox_inches='tight', dpi = 300)
    plt.close()


def make_scatter_all_blocks(df,B_cols,var_x = "blocked_density", var_y = "speed-vs-CSR", hue = "row_block_size", xscale = "log", yscale = "linear"):
    plt.figure(figsize=(8,4))
    tmp_df = df.loc[(df["b_cols"] == B_cols) & (df["multiplication_algo"] == 6) & (df["blocking_algo"] == 5)]
    print(tmp_df.columns)
    #print("SCATTER!", tmp_df)
    ax = sns.scatterplot(data=tmp_df, x=var_x, y=var_y, hue = "row_block_size")
    #ax.axhline(y=1, color='red',alpha = 0.7)


    #point = (0.95,1)
    #transformed_point = ax.transAxes.transform(point)
    #transformed_point_data = ax.transData.inverted().transform(transformed_point)
    #ax.text(transformed_point_data[0], 1.05, 'density amplification resulted in speed-up', ha='center', va='bottom', transform=ax.transData)
    #ax.text(transformed_point_data[0], 0.95, 'density amplification resulted in slow-down', ha='center', va='top', transform=ax.transData)ù

    #plt.ylim(ymin= 0, ymax = 10)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(global_label_dict[var_x])
    plt.ylabel(global_label_dict[var_y])
    plt.savefig(f"{image_folder}/scatter_plot_{var_x}_vs_{var_y}_B_cols_{B_cols}",  bbox_inches='tight', dpi = 300)
    plt.close()

parser = argparse.ArgumentParser(description='Plots for multiplication experiments')
parser.add_argument('-f',
                    '--file',
                    type=str,
                    default='results/test_suitsparse_3_multiplication.csv',
                    help='The csv with the results to be plotted')
parser.add_argument('-i',
                    '--imagedir',
                    type = str,
                    default='images/multiplication_images/',
                    help='The directory where to save the images')
args = vars(parser.parse_args())


data_file = args["file"]
image_folder = args["imagedir"]
try: os.mkdir(image_folder) 
except: 1

df = pd.read_csv(data_file)
#df = get_dataframe_folder("results/suitsparse_collection_3")


#df_CSR = df.loc[df["multiplication_algo"] == 2][["matrix","b_cols","avg_time_multiply"]]
df["block_density"] = df["nonzeros"]/df["VBR_nzcount"]
#df_VBR_no_reord = df.loc[(df["multiplication_algo"] == 6) & (df["blocking_algo"] == 2)][["matrix","b_cols","avg_time_multiply","col_block_size","row_block_size","block_density"]]


d = (0.000001, 0.1)
n = (20000,500000)

df["density"] = df["nonzeros"].values/(df["rows"].values * df["cols"].values)
df = df[(df["density"] >= d[0]) & (df["density"] <= d[1])]
df = df[(df["rows"] >= n[0]) & (df["rows"] <= n[1])]
df = df[(df["cols"] >= n[0]) & (df["cols"] <= n[1])]

print("MATRICES:", len(df["matrix"].unique()))

def count_failed(df, b_cols):
    failed_df = df.loc[df["avg_time_multiply"] < 0.001]
    completed_df = df.loc[(df["b_cols"] == b_cols) & (df["avg_time_multiply"] > 0.001)]
    completed = len(completed_df["matrix"].unique())
    return completed

def count_failed_block(df, b_cols, row_block_size):
    failed_df = df.loc[df["avg_time_multiply"] < 0.001]
    completed_df = df.loc[(df["b_cols"] == b_cols) & (df["avg_time_multiply"] > 0.001) & (df["row_block_size"] == row_block_size)]
    completed = len(completed_df["matrix"].unique())
    return completed


dfs = {}
for exp, params in experiments.items():
    if exp in ("VBR-no-reord","VBR-reord"): #if blocking is involved
        dfs[exp] = df.loc[(df["multiplication_algo"] == params["multiplication_algo"]) & (df["blocking_algo"] == params["blocking_algo"])][["matrix","b_cols","avg_time_multiply","col_block_size","row_block_size","block_density"]]
    elif exp in("BELLPACK-no-reord","CUTLASS-BELLPACK"): #if blocking is involved b ut needs to be taken out
        dfs[exp] = df.loc[(df["multiplication_algo"] == params["multiplication_algo"]) & (df["blocking_algo"] == params["blocking_algo"])][["matrix","b_cols","avg_time_multiply","col_block_size","row_block_size","block_density"]]
    else:
        dfs[exp] = df.loc[(df["multiplication_algo"] == params["multiplication_algo"])][["matrix","b_cols","avg_time_multiply"]]

for exp in experiments.keys():
    for b in df["b_cols"].unique():
        print(f"{exp} ***COMPLETED at block_size {b}: ", count_failed(dfs[exp],b))
        if "VBR" in exp:
            for row_block_size in df["row_block_size"].unique():
                print(row_block_size, count_failed_block(dfs[exp], b, row_block_size))



df = df.loc[(df["blocking_algo"] == 5) & (df["multiplication_algo"] == 6)]


#make_histo_best_blocksize(df, image_folder,1024, x_var = 'density')

for exp, params in experiments.items():
    print("collecting experiment:", exp, params)
    if exp in ("VBR-no-reord","VBR-reord"):
        df = pd.merge(df,dfs[exp], how = "left",on = ["matrix","b_cols","row_block_size","col_block_size"], suffixes=('','_' + exp) )
    else:
        df = pd.merge(df,dfs[exp], how = "left",on = ["matrix","b_cols"], suffixes=('','_' + exp) )

print(df.columns)
df["dense-amp"] = df["block_density"]/df["block_density_VBR-no-reord"]
df[df["dense-amp"] <= 1]["avg_time_multiply"] = df["avg_time_multiply_VBR-no-reord"]

print(df["dense-amp"])
print("MAAAAX", df.loc[df["dense-amp"].idxmax()])

for exp in experiments.keys():
    if exp == "VBR-reord": continue
    print("recording speed-up: ", exp)
    df["speed-vs-" + exp] = df["avg_time_multiply_" + exp]/df["avg_time_multiply"]




df["time-per-block"] = df["avg_time_multiply"]/df["VBR_nzblocks_count"]
df["time-per-area"] = df["avg_time_multiply"]/df["VBR_nzcount"]
df["GFLOPs"] = df["b_cols"]*df["VBR_nzcount"]/(df["avg_time_multiply"]*1000*1000*1000)

df["time-per-true-nonzero"] = df["avg_time_multiply"]/df["nonzeros"]
df = df[df["dense-amp"] > 1]


#df_BELLPACK = df[df["multiplication_algo"] == 3][["matrix","b_cols","avg_time_multiply","row_block_size","col_block_size"]]
#df = pd.merge(df,df_BELLPACK, how = "left",on = ["matrix","b_cols","row_block_size","col_block_size"], suffixes=('','_BELLPACK') )
#df_VBR_no_reord = df.loc[(df["multiplication_algo"] == 6) & df["blocking_algo"] == 2][["matrix","b_cols","avg_time_multiply","row_block_size","col_block_size"]]
#df = pd.merge(df,df_BELLPACK, how = "left",on = ["matrix","b_cols","row_block_size","col_block_size"], suffixes=('','_VBR_no_reorder') )


df["density"] = df["nonzeros"].values/(df["rows"].values * df["cols"].values)
#df["block_density"] = df["nonzeros"].values/df["VBR_nzcount"].values
#df["dense_amp"] = df["block_density"].values/df["density"].values
df.sort_values(by=['density','matrix'], inplace=True)

#PREPARE HEATMAP FOR ALL BLOCK-SIZES

print(df["speed-vs-GEMM"])

print("Making barplots")
for B_cols in (1024, 8192):
    try:
        make_histo_best(df, image_folder,B_cols)
        make_scatter(df,B_cols)
    except:
        print(f"FAILED GENERAL IMAGES for bcols {B_cols}")
    make_scatter_all_blocks(df,B_cols,var_x="VBR_nzblocks_count",var_y="GFLOPs")
    for exp in experiments.keys():
        if exp != "VBR-reord":
            make_scatter_all_blocks(df,B_cols,var_x="block_density",var_y="speed-vs-" + exp)
            make_scatter_all_blocks(df,B_cols,var_x="density",var_y="speed-vs-" + exp)
    
    print(f"***** for B_cols = {B_cols}")
    make_boxplot_best(df, image_folder,B_cols)
    for row_block_size in (32,64,128,256,512,1024):
        col_block_size = row_block_size
        make_scatter_block_size(df,B_cols,col_block_size=col_block_size,row_block_size=row_block_size)

"""

print("Making heatmaps")
for B_cols in df["b_cols"].unique():
        for exp_name in experiments:
            try:
                make_heatmap(df,image_folder,B_cols,exp_name, colormap_variable= "time-per-true-nonzero")
                make_heatmap(df,image_folder,B_cols,exp_name, colormap_variable= "speed-vs-cusparse")
                make_heatmap(df,image_folder,B_cols,exp_name, colormap_variable= "speed-vs-no-reord")
                make_heatmap(df,image_folder,B_cols,exp_name, colormap_variable= "time-per-block")
                make_heatmap(df,image_folder,B_cols,exp_name, colormap_variable= "time-per-area")
            except: 
                print(f"no heatmap for {exp_name}")


"""