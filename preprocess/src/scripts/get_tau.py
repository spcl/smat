# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:35:55 2022

@author: Paolo
"""

import pandas as pd
import os as os
import argparse   

parser = argparse.ArgumentParser(description='Plots for multiplication experiments')
parser.add_argument('-t',
                    '--taufile',
                    type=str,
                    default='results/collected_experiments/suitsparse_all/tau.csv',
                    help='The csv with the taus')
parser.add_argument('-m',
                    '--matrix',
                    type = str,
                    default='NOMATRIXSELECTED',
                    help='The matrix')
parser.add_argument('-b',
                    '--colblocksize',
                    type = int,
                    default=64,
                    help='The column block size')
parser.add_argument('-B',
                    '--rowblocksize',
                    type = int,
                    default=64,
                    help='The row block size')
args = vars(parser.parse_args())


data_file = args["taufile"]
matrix = args["matrix"]
row_size = args["rowblocksize"]
col_size = args["colblocksize"]

try:
    df = pd.read_csv(data_file)
    df = df[df['matrix'].str.contains(matrix)]
    taus = df.loc[(df["row_block_size"] == row_size) & (df["col_block_size"] == col_size)]["tau"].values
    if len(taus) > 1:
        print(-2)
        exit()
    print(taus[0])
except:
    print(-3)