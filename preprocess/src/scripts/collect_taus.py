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
import argparse   

parser = argparse.ArgumentParser(description='Plots for multiplication experiments')
parser.add_argument('-f',
                    '--filein',
                    type=str,
                    default='results/test_suitsparse_3_multiplication.csv',
                    help='The csv with the results to be plotted')
parser.add_argument('-o',
                    '--outfile',
                    type = str,
                    default='results/tau.csv',
                    help='The directory where to save the taus csv')
args = vars(parser.parse_args())


data_file = args["filein"]
outfile = args["outfile"]

df = pd.read_csv(data_file)

df = df.loc[df.groupby(["matrix","blocking_algo","col_block_size","row_block_size"])["VBR_nzblocks_count"].idxmin()]
df_VBR_no_reord = df[df["blocking_algo"] == 2][["matrix","col_block_size","row_block_size","VBR_nzblocks_count"]]

df = pd.merge(df[df["blocking_algo"] == 5],df_VBR_no_reord, how = "left",on = ["matrix","row_block_size","col_block_size"], suffixes=('','_no_reord') )
df["relative-dense-amp"] = df["VBR_nzblocks_count"]/df["VBR_nzblocks_count_no_reord"]
df.loc[df["relative-dense-amp"] >= 1, "tau"] = -1

df[["matrix","row_block_size","col_block_size","tau"]].to_csv(outfile, index = False)