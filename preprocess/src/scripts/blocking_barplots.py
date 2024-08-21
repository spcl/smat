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
from matplotlib import cm, colors
import matplotlib.ticker as mtick

import argparse



global_label_dict = {
    "block_density" : "Blocked density (IC)",
    "density" : "Original density",
    "nonzeros" : "# of nonzeros in the original matrix",
    "dense-amp" : "Density amplification (against unblocked)",
    "relative-dense-amp" : "Density amplification",
    "block ratio": "Shape (height / width)",
    "time_to_block" : "DenseAMP runtime (ms)",
    "rows" : "Sparse matrix rows (N)",
    "cols" : "Sparse matrix columns (M)",
    "block_density_no_reord" : "Blocked density (natural)"
}

global_exp_dict = {}

global_exp_dict["reordered"] = {
    "style" : {
        "hatch" : "//",
        "edgecolor" : "black",
        "color" : "orange",
        "label" : "IC blocking"
    }
}

global_exp_dict["no-reordered"] = {
    "style": {
    "hatch" : "//",
    "edgecolor" : "black",
    "color" : "blue",
    "label" : "natural blocking"
    }
}

global_exp_dict["original"] = {
    "style" : {
    "hatch" : "..",
    "edgecolor" : "black",
    "color" : "white",
    "label" : "orginal matrix"
    }
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

def barplot(x_labels, x_ax_label, ys, y_styles, y_ax_label, yscale = "log", savename = "test_barplot"):
    plt.figure()
    plt.xlabel(x_ax_label)
    plt.ylabel(y_ax_label)
    
    bars = len(ys)
    tot_width = 0.9
    barpos = -tot_width/2
    increment = tot_width/bars
    width = increment*0.95

    for y, style in zip(ys, y_styles):
        x_pos = np.arange(barpos,len(x_labels) + barpos)
        plt.bar(x_pos,y, width = width, **style)
        barpos += increment

    plt.legend()
    plt.yscale(yscale)
    plt.grid("both")
    plt.title(f"block height = {row_block_size}, block width = {col_block_size}")
    plt.xticks(range(len(x_labels)), x_labels, rotation=90)
    plt.savefig(savename + ".png",  bbox_inches='tight', dpi = 300)
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
                    default='images/blocking_images/test_images',
                    help='The directory where to save the images')
args = vars(parser.parse_args())


data_file = args["file"]
image_folder = args["imagedir"]
try: os.mkdir(image_folder) 
except: 1

df = pd.read_csv(data_file)
#df = get_dataframe_folder("results/suitsparse_collection_3")

d = (0.000001, 0.1)
n = (20000,500000)

print("*************SIZES:")
print("MIN:", df["rows"].min(), "MAX: ", df["rows"].max())

df = df.loc[df.groupby(["matrix","blocking_algo","col_block_size","row_block_size"])["VBR_nzblocks_count"].idxmin()]

df["density"] = df["nonzeros"].values/(df["rows"].values * df["cols"].values)
df = df[(df["density"] >= d[0]) & (df["density"] <= d[1])]
df = df[(df["rows"] >= n[0]) & (df["rows"] <= n[1])]
df = df[(df["cols"] >= n[0]) & (df["cols"] <= n[1])]

df["block_density"] = df["nonzeros"].values/df["VBR_nzcount"].values

df["dense-amp"] = df["block_density"].values/df["density"].values
df["block_area"] = df["row_block_size"].values*df["col_block_size"].values
df["block ratio"] = df["row_block_size"].values/df["col_block_size"].values


df_VBR_no_reord = df[df["blocking_algo"] == 2][["matrix","col_block_size","row_block_size","block_density"]]
df = pd.merge(df[df["blocking_algo"] == 5],df_VBR_no_reord, how = "left",on = ["matrix","row_block_size","col_block_size"], suffixes=('','_no_reord') )
df["relative-dense-amp"] = df["block_density"]/df["block_density_no_reord"]
df.loc[df["relative-dense-amp"] < 1, "relative-dense-amp"] = 1

print("MAX RELATIVE ")
print(df["relative-dense-amp"].max())


for block_size in (64,128,256,512,1024):
    print(f"BLOCK SIZE: {block_size}")
    bad = len(df.loc[(df["relative-dense-amp"] == 1) & (df["row_block_size"] == block_size)])
    good = len(df.loc[(df["relative-dense-amp"] > 1) & (df["row_block_size"] == block_size)])
    print("BAD", bad)
    print("GOOD", good)
    print("PERCENT:", good/(good+bad+0.00001))

df.sort_values(by=['density','matrix'], inplace=True)

np.set_printoptions(suppress = True)

#print(df["matrix"].unique())
for row_block_size in (32,64,128,256,512,1024):
    for col_block_size in (64,128,256,512,1024):
        print(f"row_block_size, col_block_size = {row_block_size},{col_block_size}")
        tmp_df = df.loc[(df["blocking_algo"] == 5) & (df["col_block_size"] == col_block_size) & (df["row_block_size"] == row_block_size)]
        #print(tmp_df["tau"].values)


def make_scatter(x_var = "density", y_var = "relative-dense-amp", row_block_size = 128, col_block_size = 128, xscale = "log", yscale = "linear", drawline = 0):
    blocking_algo = 5
    fig, ax = plt.subplots()
    tmp_df = df.loc[(df["blocking_algo"] == blocking_algo) & (df["row_block_size"] == row_block_size) & (df["col_block_size"] == col_block_size)]
    tmp_df.plot.scatter(
                    x= x_var,
                    y= y_var,
                    colormap='viridis',alpha=0.5, edgecolor = "black", ax = ax)
    plt.ylabel(global_label_dict[y_var])
    if drawline:
        plt.axhline(drawline,color = "red", linestyle= "--")
    #plt.xlim(0.00001,1)
    #plt.ylim(0.00001,1)
    plt.xscale(xscale)
    plt.yscale(yscale)

    # Compute histogram of y_var > 1 at each x_var bin
    #bins = np.logspace(np.log10(tmp_df[x_var].min()), np.log10(tmp_df[x_var].max()), 20)
    #mask = tmp_df[y_var] > 1
    #mask = mask / (mask + (tmp_df[y_var] <= 2))
    #weights = mask.astype(float)
    #hist, bin_edges = np.histogram(tmp_df[x_var], bins=bins, weights=weights)

    # Plot histogram
    #axs[1].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.5, edgecolor="black")
    #axs[1].set_ylabel('Count')

    #plt.yscale("log")
    plt.xlabel(global_label_dict[x_var])
    plt.savefig(f"{image_folder}/scatterplot_{x_var}_vs_{y_var}_b{row_block_size}_B{col_block_size}_algo_{blocking_algo}.png", bbox_inches='tight', dpi = 300)
    plt.close()


def make_scatter_all_block(x_var = "nonzeros", y_var = "time_to_block"):
    blocking_algo = 5
    fig, ax = plt.subplots()
    tmp_df = df.loc[(df["blocking_algo"] == blocking_algo)]
    sns.scatterplot(data=tmp_df,
                        x= x_var,
                        y= y_var,
                        alpha=0.5, edgecolor = "black", ax = ax, label = row_block_size, hue = "row_block_size")
    plt.ylabel(global_label_dict[y_var])
    plt.yscale("log")
    plt.xscale("log")

    plt.xlabel(global_label_dict[x_var])
    plt.yscale("log")
    plt.savefig(f"{image_folder}/scatterplot_{x_var}_vs_{y_var}.png", bbox_inches='tight', dpi = 300)
    plt.close()

def make_success_hist(x_var="density", y_var="relative-dense-amp", row_block_size=128, col_block_size=128, xscale="log", yscale="linear", drawline=0):
    blocking_algo = 5
    fig, ax = plt.subplots( figsize=(8, 2.5))
    tmp_df = df.loc[(df["blocking_algo"] == blocking_algo) & (df["row_block_size"] == row_block_size) & (df["col_block_size"] == col_block_size)]

    # Compute histogram of y_var > 1 at each x_var bin
    bins = np.logspace(np.log10(tmp_df[x_var].min()), np.log10(tmp_df[x_var].max()), 10)

    tmp_df["density_bin"] = pd.cut(tmp_df[x_var], bins=bins)
    success = tmp_df[tmp_df[y_var] > 1].groupby("density_bin").size() / tmp_df.groupby("density_bin").size() * 100

    # plot the results using seaborn barplot
    ax = sns.barplot(x=bins[:-1], y=success.values, color="blue", alpha=0.5)

    xticks = np.arange(len(bins)-1) + 0.5
    xticklabels = [np.format_float_scientific(bin, precision=0, exp_digits=1) for bin in bins[:-1]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel('% of successfully amplified')
    #ax.set_xscale(xscale)
    ax.set_xlabel(global_label_dict[x_var])
    ax.set_ylim([0, 100])

    plt.savefig(f"{image_folder}/dense_hist_{x_var}_vs_{y_var}_b{row_block_size}_B{col_block_size}_algo_{blocking_algo}.png",
                bbox_inches='tight', dpi=300)
    plt.close()

def make_success_hist_all_size(x_var="density", y_var="relative-dense-amp"):
    blocking_algo = 5
    fig, ax = plt.subplots(figsize=(8, 4))
    tmp_df = df.loc[(df["blocking_algo"] == blocking_algo)]

    # Create histogram
    bin_num = 10  # Adjust bin size as needed
    bin_edges = np.logspace(-5, -1, bin_num)


    tmp_df["density_bin"] = pd.cut(tmp_df[x_var], bins=bin_edges)
    total_counts = tmp_df.groupby(["density_bin", "row_block_size"]).size().reset_index(name="total_counts")
    success_counts = tmp_df[tmp_df[y_var] > 1].groupby(["density_bin", "row_block_size"]).size().reset_index(name="success_counts")

    # Merge total_counts and success_counts on density_bin and row_block_size
    merged_counts = pd.merge(total_counts, success_counts, on=["density_bin", "row_block_size"], how="left")

    # Compute percentage success as a new column
    merged_counts["percent_success"] = 100 * merged_counts["success_counts"] / merged_counts["total_counts"]

    print(merged_counts)
    # Plot histogram
    sns.barplot(data=merged_counts, x="density_bin", y="percent_success", ax=ax, edgecolor="black", hue = "row_block_size")
    ax.set_xlabel(global_label_dict[x_var])
    ax.set_ylabel("Percentage of successful denseAMP")
    # Set x-tick labels to the lower end of each bin
    xticks = np.arange(bin_num -1) - 0.5
    xticklabels = [np.format_float_scientific(bin, precision=0, exp_digits=1) for bin in bin_edges[:-1]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)


    #ax.set_xticks(np.arange(0.5,len(bin_edges), 1))  # Set xticks to the lower edge of each bin
    # Set the x-axis tick locations to the powers of 10
    #xticks_shifted = [tick-0.5 for tick in range(tick_labels)]
    #ax.set_xticks(xticks_shifted)  # Set xticks to the shifted values
    plt.legend (title = "Block size")

    plt.savefig(f"{image_folder}/hist_success_{x_var}_vs_{y_var}_all_block_size.png",
                bbox_inches='tight', dpi=300)
    plt.close()


def make_hist(x_var="density", y_var="relative-dense-amp", row_block_size=128, col_block_size=128, xscale="log", yscale="linear", drawline=0):
    blocking_algo = 5
    fig, ax = plt.subplots(figsize=(8, 3))
    tmp_df = df.loc[(df["blocking_algo"] == blocking_algo) & (df["row_block_size"] == row_block_size) & (df["col_block_size"] == col_block_size)]

    # Create histogram
    bin_size = 0.1  # Adjust bin size as needed
    bin_edges = [0.9999, 1.001]  # Define custom bin edges
    bin_edges += list(np.arange(1.1, 1.4*tmp_df["relative-dense-amp"].mean() + bin_size, bin_size))
    bin_edges += [tmp_df["relative-dense-amp"].max()]

    tmp_df["density_bin"] = pd.cut(tmp_df["relative-dense-amp"], bins=bin_edges)    
    density_counts = tmp_df.groupby("density_bin").size().reset_index(name="counts")
    print(density_counts)
    density_counts["percentage"] = density_counts["counts"] / len(tmp_df["relative-dense-amp"]) * 100

    # Plot histogram
    sns.barplot(data=density_counts, x="density_bin", y="percentage", ax=ax, edgecolor="black")
    ax.set_xlabel("Density amplification (against natural blocking)")
    ax.set_ylabel("Percentage of Points")
    #ax.set_xticks(np.arange(0.5,len(bin_edges), 1))  # Set xticks to the lower edge of each bin
    xticks = np.arange(len(bin_edges) -1) + 0.5
    xticklabels = [bin for bin in bin_edges[:-1]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    #xticks_shifted = [tick-0.5 for tick in range(tick_labels)]
    #ax.set_xticks(xticks_shifted)  # Set xticks to the shifted values
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.savefig(f"{image_folder}/hist_{x_var}_vs_{y_var}_b{row_block_size}_B{col_block_size}_algo_{blocking_algo}.png",
                bbox_inches='tight', dpi=300)
    plt.close()

def make_hist_all_block_sizes(x_var="density", y_var="relative-dense-amp"):
    blocking_algo = 5
    fig, ax = plt.subplots(figsize=(8, 4))
    tmp_df = df.loc[(df["blocking_algo"] == blocking_algo)]

    # Create histogram
    bin_size = 0.1  # Adjust bin size as needed
    bin_edges = [0.9999, 1.001]  # Define custom bin edges
    bin_edges += list(np.arange(1.1, 1.4*tmp_df["relative-dense-amp"].mean() + bin_size, bin_size))
    bin_edges += [tmp_df["relative-dense-amp"].max()]

    tmp_df["density_bin"] = pd.cut(tmp_df["relative-dense-amp"], bins=bin_edges)    
    density_counts = tmp_df.groupby(["density_bin", "row_block_size"]).size().reset_index(name="counts")
    print(density_counts)
    density_counts["percentage"] = density_counts.groupby("row_block_size")["counts"].transform(lambda x: x / x.sum() * 100)

    # Plot histogram
    sns.barplot(data=density_counts, x="density_bin", y="percentage", ax=ax, edgecolor="black", hue = "row_block_size")
    ax.set_xlabel("Density amplification (against natural blocking)")
    ax.set_ylabel("Percentage of Points")
    #ax.set_xticks(np.arange(0.5,len(bin_edges), 1))  # Set xticks to the lower edge of each bin
    xticks = np.arange(len(bin_edges) ) - 0.5
    xticklabels = [1,1] + [f"{bin:.2g}" for bin in bin_edges[2:-1]] + ["+inf",]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.legend (title = "Block size")

    plt.savefig(f"{image_folder}/hist_{x_var}_vs_{y_var}_all_block_size.png",
                bbox_inches='tight', dpi=300)
    plt.close()


print(df["time_to_block"])

try:
    make_hist_all_block_sizes(x_var="density", y_var="relative-dense-amp")
    make_scatter_all_block(x_var = "rows", y_var = "time_to_block")
    make_scatter_all_block(x_var = "nonzeros", y_var = "time_to_block")
    make_success_hist_all_size(x_var="density", y_var="relative-dense-amp")
    make_success_hist_all_size(x_var="block_density_no_reord", y_var="relative-dense-amp")
except: 
    "PROBLEMS!"

for block_size in (64,128,256,512,1024):
        try:
            make_success_hist(x_var ="density", y_var = "relative-dense-amp", row_block_size = block_size, col_block_size = block_size)
            make_success_hist(x_var ="block_density_no_reord", y_var = "relative-dense-amp", row_block_size = block_size, col_block_size = block_size)
            make_hist(x_var ="block_density_no_reord", y_var = "relative-dense-amp", row_block_size = block_size, col_block_size = block_size)
            make_hist(x_var ="density", y_var = "relative-dense-amp", row_block_size = block_size, col_block_size = block_size)
            make_scatter(x_var ="rows", y_var = "time_to_block", row_block_size = block_size, col_block_size = block_size, yscale="log")
            make_scatter(x_var ="density", y_var = "relative-dense-amp", row_block_size = block_size, col_block_size = block_size)
        except:
            print(f"COULD NOT MAKE SCATTER FOR {block_size} x {block_size}")


"""
x_var = "density"
y_var = "relative-dense-amp"
for row_block_size in (32,64,128,256,512,1024):
    for col_block_size in (32,64,128,256,512,1024):
        try:
            fig,ax = plt.subplots()
            tmp_df = df.loc[(df["blocking_algo"] == 5) & (df["row_block_size"] == row_block_size) & (df["col_block_size"] == row_block_size)]
            tmp_df.plot.scatter(
                            x= x_var,
                            y= y_var,
                            colormap='viridis',alpha=0.5, edgecolor = "black", marker = "s", ax = ax, label = "Blocking after reordering")
            plt.ylabel(global_label_dict[y_var])
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.xscale("log")
            plt.savefig(f"{image_folder}/scatterplot_{x_var}_vs_{y_var}_{row_block_size}x{col_block_size}.png", bbox_inches='tight', dpi = 300)
        except:
            print(f"FAILED TO PRODUCE SCATTER PLOT FOR {row_block_size} x {col_block_size}")
        plt.close()
"""


color_var = "relative-dense-amp"
color_label = "Density Amplification (against natural blocking)"

plt.figure()
vmin = 1
vmax = 2
table = df.loc[(df["blocking_algo"] == 5) & (df["relative-dense-amp"] > 1)]
table = table.pivot_table(index="row_block_size", columns="col_block_size", values=color_var, aggfunc='mean')
table = table.sort_values(by = ["row_block_size",], ascending = False)
sns.heatmap(table,annot = True, cbar_kws={'label': color_label},vmin = vmin, vmax = vmax)
plt.ylabel("block height")
plt.xlabel("block width")
plt.savefig(f"{image_folder}/heatmap_{color_var}.png",  bbox_inches='tight', dpi = 300)
plt.close()
    

"""
for algo in (2,5):
    heatmap_df = df[df["blocking_algo"]==algo].pivot_table(index="row_block_size", columns="col_block_size", values=colormap_variable, aggfunc='mean')
    heatmap_df = heatmap_df.sort_values(by=['row_block_size'], ascending=False)
    sns.heatmap(heatmap_df,annot = True, cbar_kws={'label': color_label},vmin = 1, vmax = max_var_value)
    plt.ylabel("block height")
    plt.xlabel("block width")
    plt.savefig(f"{image_folder}/heatmap_{colormap_variable}_algo_{algo}.png",  bbox_inches='tight', dpi = 300)
    plt.close()
"""




for row_block_size in sorted(df["row_block_size"].unique()):
    for col_block_size in sorted(df["col_block_size"].unique()):
        try:
            tmp_df = df.loc[(df["col_block_size"]==col_block_size) & (df["row_block_size"] == row_block_size)]
            matrices_names = [val.split("/")[-1].split(".")[0] for val in tmp_df["matrix"].unique()]

            reorder_df = tmp_df[tmp_df["blocking_algo"] == 5]
            no_reorder_df = tmp_df[tmp_df["blocking_algo"] == 2]
            try:
                assert np.equal(reorder_df["matrix"].values, no_reorder_df["matrix"].values).all() #check that there are the same n of values for all matrices
                assert len(reorder_df) > 0
            except:
                print(f"MISSING DATA for blocks {row_block_size} x {col_block_size}")
                continue

            print(f"FOUND DATA for {row_block_size} x {col_block_size}")
            data_lines = {}

            data_lines["blocking, no reordering"] = no_reorder_df["block_density"].values
            data_lines["blocking, reordering"] = reorder_df["block_density"].values
            data_lines["no blocking (original)"] = tmp_df["density"].unique()   

            savename = f"{image_folder}/Density_barplot_{row_block_size}_{col_block_size}"

            exps = ""
            styles = [exp["style"] for exp in global_exp_dict]

            barplot(
                    x_labels = matrices_names,
                    x_ax_label="graphs",
                    ys=data_lines.values(),
                    y_labels=data_lines.keys(),
                    y_ax_label = "Density", 
                    savename = savename)
        except:
            print(f"NO BARPLOT FOR {row_block_size} x {col_block_size}")
    plt.close()
