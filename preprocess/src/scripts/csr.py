# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:14:55 2022

@author: Paolo
"""


import argparse
import os
import numpy as np
import itertools
from numpy.random import default_rng

class CSR: 
    def __init__(self):
        self.clean()
        
    def clean(self):
        self.N = 0
        self.M = 0
        self.nzcount = []
        self.pos = []
        
    def __str__(self):
        print("N:", self.N, "M:", self.M)
        print("nzcount: ", self.nzcount)
        print("pos: ", self.pos)
        return " "
        
    
    def tot_nz(self):
        tot = 0
        for count in self.nzcount:
            tot += count
        return tot
    
    def density(self):
        return self.tot_nz()*1./(self.N*self.M)
    
    def fill_from_edgelist(self,edgelist_file, delimiter = " "):
        #must be a sorted edgelist (gaps allowed)
        self.clean()
        with open(edgelist_file) as f:
            for line in f:
                linesplit = line.split(delimiter)
                inc = int(linesplit[0])
                out = int(linesplit[1])
                
                while inc > self.N - 1:
                    #create new node;
                    self.add_node()
                    
                self.add_edge(row = -1, col = out)
                
        return self.N
    
    def fill_from_array(self,array):
        self.clean()
        self.M = len(array[0])
        for row_idx, row in enumerate(array):
            self.add_node()
            for col_idx, elem in enumerate(row):
                if elem != 0.:
                    self.nzcount[row_idx] +=1
                    self.pos[row_idx].append(col_idx)
                    

    def fill_uniform_random(n,m, density):
        self.clean()
        self.M = m
        self.N = n
        k = int(density*n*m)
        rng = default_rng()
        nz_pos = rng.choice(m*n, size=k, replace=False)
        j = 0
        for i in range(n):
            self.add_node()

        for j in nz_pos:
            self.add_edge(row = nz_pos[j]/m, col = nz_pos[j]%m)


    def add_node(self):
        self.N += 1
        self.nzcount.append(0)
        self.pos.append([])
    
    def add_edge(self, row, col):
        self.nzcount[row] += 1
        self.pos[row].append(col)
    
    def print_edgelist(separator = " "):
        for i, row in enumerate(self.pos):
            for j in row:
                print (f"{i}{separator}{j}")
            
#****************************************************************


def compressed_to_dense(row, width):
    out = np.zeros(width)
    for elem in row:
        out[elem] = 1
    return out
            

def print_blocked(cmat, grouping, block_size):
    groups = np.unique(grouping)
    for g in groups:
        for row, row_group in zip(cmat.pos, grouping):
            if row_group == g:
                print(compressed_to_dense(row,cmat.M))
        print("___"*cmat.M)

