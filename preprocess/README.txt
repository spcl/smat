Compile with 

'''make serial''' to compile without cuda  
or
'''make all''' to compile also the cuda test

run '''./programs/general/TEST_blocking_VBR''' to see an example of blocking; 

For example, run
./programs/general/TEST_blocking_VBR -b 3 -t 0.6
to produce a blocking of a test matrix, fixing the column size at 3 (-b 3) and the threshold distance tau at 0.6 (-t 0.6).

run again with
./programs/general/TEST_blocking_VBR -b 3 -t 0.6 -F 1 -B 3
to force fixed-height blocks (-F 1) of height 3 (-B 3)

add the option -f PATH/TO/MATRIX.el to load a matrix. 
some small matrices are available for testing in data/
you can use your own matrices, provided they are stored as an edgelist with space-separated, ordered values.

Find all the options below:


OPTIONS: 
-a: blocking algorithm selection:
		0: iterative, 
		1: iterative_structured, 
		2: fixed_size 
		3: iterative_clocked
		4: iterative_queue 
		5: iterative_max_size (BEST fixed block)

-b: column block size

-B: row block size (only for fixed-size blockings)

-c: number of columns in the matrix B (only used when running AB multiplication)

-f: filename of an edgelist to be read from memory

-F: force fixed size: 
		0: false. The blocking algorithm may creat blocks of uneven height
		1: true. Whatever is the result of the blocking algorithm, a fixed-size grid (see -b, -B) will be superimposed to the result.

-g: use group sized when calculating similarity.
		0: false
		1: true

-o: filename where to save the results of blocking and multiplication

-p: usage of "pattern" when calculating similarities:
		0: do not use pattern. similarities are calculated between a candidate row and the seed row.
		1: use patterns. similarities are calculated between a candidate and the entire cluster

-P: treat the matrix as weighted or not
		0: weights are ignored when reading a matrix from edgelist and during processing
		1: weights are loaded, stored, and processed

-m: similarity measure:
		0: Hamming
		1: Jaccard (default)

-M: spmm multiplication algorithm. Blocking must be appropriate to the chosen algorithm.
		0: no multiplication
		1: cuBLAS GEMM (blocking is ignored) 
		2: cuSparse CSR (blocking is ignored)
		3: cuSparse BELLPACK (blocks should be fixed-size and square)
		4: cuBLAS VBR (any blocking allowed);

-n: name of the experiment

-r: reorder the CSR matrix before processing/blocking/multiplying
	0: do nothing (default)
	1: reorder rows by nonzero count (descending)
	2: scramble rows

-R: matrix format (how each line in the edgelist looks like)
	0: row col (default)
	1: col row

-s: random seed

-S: number of cuda streams to be used in the VBR multiplication 
		16 (default)

-t: the distance threshold for merging similar rows:
	0.0: merge only identical rows
	0.x: only merge when distance < 0.x
	1.0: merge any nonzero row 

-v: verbose
	0: print minimum
	1: print infos, but not matrices
	2: print matrices

-w: how many warmup multiplication runs?

-x: how many repetition to average for multiplication?
