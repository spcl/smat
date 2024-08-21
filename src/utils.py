import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import statistics
import random

from format_converter import coo_matrix2mtx

LOG_LEVEL=1


def randomly_permute_rows(matrix, n):
    copied_matrix = matrix.copy()
    indices = np.arange(0, copied_matrix.shape[0])
    for _ in range(n):
        i, j = np.random.choice(indices, size=2, replace=False)
        copied_matrix[i, :], copied_matrix[j, :] = copied_matrix[j, :], copied_matrix[i, :].copy()
    return copied_matrix


def randomly_permute_columns(matrix, n):
    copied_matrix = matrix.copy()
    indices = np.arange(0, copied_matrix.shape[1])
    for _ in range(n):
        i, j = np.random.choice(indices, size=2, replace=False)
        copied_matrix[:, i], copied_matrix[:, j] = copied_matrix[:, j], copied_matrix[:, i].copy()
    return copied_matrix


def get_dense_matrix(name):
    matrix = sp.io.mmread(name)
    dense_array = np.array(matrix.todense())
    return dense_array


def get_num_permutations(matrix):
    return max(matrix.shape[0], matrix.shape[1])


def get_sparsity(matrix):
    sparsity = 1.0 - (np.count_nonzero(matrix) / float(matrix.size))
    return 100 * sparsity


def plot_and_print_sparsity_statistics(name):
    def get_row_sparsity(matrix):
        total_elements_in_each_row = matrix.shape[1]
        zero_elements_in_each_row = np.count_nonzero(matrix == 0, axis=1)
        percentage_zero_in_each_row = (zero_elements_in_each_row / total_elements_in_each_row) * 100
        return percentage_zero_in_each_row

    def get_column_sparsity(matrix):
        total_elements_in_each_column = matrix.shape[0]
        zero_elements_in_each_column = np.count_nonzero(matrix == 0, axis=0)
        percentage_zero_in_each_column = (zero_elements_in_each_column / total_elements_in_each_column) * 100
        return percentage_zero_in_each_column

    dense_array = get_dense_matrix(name)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(dense_array)
    axs[0].set_title('Original matrix')

    num_permutations = get_num_permutations(dense_array)
    permuted_rows_and_columns = randomly_permute_rows(randomly_permute_columns(matrix=dense_array, n=num_permutations),
                                                      n=num_permutations)
    # plot_matrix(permuted_rows_and_columns, title='Permuted rows and columns')
    axs[1].imshow(permuted_rows_and_columns)
    axs[1].set_title('Permuted rows and columns')

    # total_elements = permuted_rows_and_columns.size
    # zero_elements = np.count_nonzero(permuted_rows_and_columns == 0)
    # sparsity = (zero_elements / total_elements) * 100
    print(f'matrix shape: {permuted_rows_and_columns.shape[0]} x {permuted_rows_and_columns.shape[1]}')
    # print('sparsity: ', sparsity)

    # calculate average sparsity per row
    # percentage_zero_in_each_row_before_permuting = get_row_sparsity(dense_array)
    # print(f'row sparsity before permuting: {statistics.mean(percentage_zero_in_each_row_before_permuting)} \u00B1 {statistics.stdev(percentage_zero_in_each_row_before_permuting)}')
    percentage_zero_in_each_row = get_row_sparsity(permuted_rows_and_columns)
    print(
        f'row sparsity after permuting: {statistics.mean(percentage_zero_in_each_row)} \u00B1 {statistics.stdev(percentage_zero_in_each_row)}')

    # calculate average sparsity per column
    # percentage_zero_in_each_column_before_permuting = get_column_sparsity(dense_array)
    # print(f'column sparsity before permuting: {statistics.mean(percentage_zero_in_each_column_before_permuting)} \u00B1 {statistics.stdev(percentage_zero_in_each_column_before_permuting)}')
    percentage_zero_in_each_column = get_column_sparsity(permuted_rows_and_columns)
    print(
        f'column sparsity after permuting: {statistics.mean(percentage_zero_in_each_column)} \u00B1 {statistics.stdev(percentage_zero_in_each_column)}')


def printDivisorsOptimal(n):
    div = []
    print("The Divisors of", n, "are:")
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            # print(i, end=" ")
            div.append(i)
            if i != n / i:
                # print(int(n/i), end=" ")
                div.append(int(n / i))
    return sorted(div)


def add_grid(ax, matrix, p=16, b=8):
    rows, cols = matrix.shape
    # Add blocks and color edges
    for i in range(0, rows, p):
        for j in range(0, cols, b):
            # Create a rectangle patch
            block = matrix[i:(i + p) % (rows + 1), j:(j + b) % (cols + 1)]
            if is_block_empty(block):
                facecolor = 'black'
            elif is_block_dense(block):
                facecolor = 'red'
            else:
                facecolor = 'none'
            if block.size != 0:
                rect = plt.Rectangle((j, i), b, p, linewidth=0.5, edgecolor='black', facecolor=facecolor)
                # Add the rectangle patch to the axis
                ax.add_patch(rect)


def plot_original_and_transformed(original, transformed, p=16, b=8):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original)
    axs[0].set_title('Original matrix')

    axs[1].imshow(original)
    axs[1].set_title('Original matrix')
    add_grid(axs[1], original, p, b)

    axs[2].imshow(transformed)
    axs[2].set_title('Random permutations')
    add_grid(axs[2], transformed, p, b)
    plt.show()


def count_dense_blocks(matrix, p, b):
    """
    Counts the number of dense blocks with shape pxb
    :param b:
    :param p:
    :param matrix:
    :return:
    """
    pass


def is_block_dense(block):
    return not is_block_2_4(block)


def is_block_empty(block):
    """
    Returns True if all elements of a block are 0
    :param block:
    :return:
    """
    return np.all(block == 0)


def is_block_2_4(block):
    """
    Returns True if the given block satisfies the required 2:4 sparsity pattern.
    :param matrix:
    :return:
    """
    for row in block:
        for i in range(0, len(row), 4):
            part = row[i : (i+4) % (len(row) + 1)]
            if np.count_nonzero(part) > 2:
                return False
    return True


def count_fill_ins(block):
    """
    Counts the number of 0 fill-ins required for the 2:4 sparsity to be satisfied. Fill-ins can be 0, 1 or 2.
    :param block:
    :return:
    """
    total_fill_ins = 0
    for row in block:
        for i in range(0, len(row), 4):
            part = row[i : (i+4) % (len(row) + 1)]
            assert np.count_nonzero(part) <= 2
            total_fill_ins += (2 - np.count_nonzero(part))
    return total_fill_ins


def iterate_over_blocks(matrix, p=16, b=8):
    """
    Iterate over pxb blocks of matrix.
    p and b are determined according to Sparse Tensor Core input dimensions.
    :param matrix:
    :param p:
    :param b:
    :return:
    """
    zero_blocks, dense_blocks, fill_ins, sparse_blocks = 0, 0, 0, 0
    fill_ins_per_block = []
    rows, cols = matrix.shape
    for i in range(0, rows, p):
        for j in range(0, cols, b):
            block = matrix[i:(i + p), j:(j + b)]
            # do statistics for the block
            if is_block_empty(block):
                zero_blocks += 1
            elif is_block_2_4(block):
                current_fill_ins = count_fill_ins(block)
                fill_ins += current_fill_ins
                fill_ins_per_block.append(current_fill_ins)
                sparse_blocks += 1
            else:
                dense_blocks += 1

    if LOG_LEVEL > 1:
        print(f"0 blocks: {zero_blocks}")
    print(f"Num bocks: {dense_blocks + sparse_blocks}, dense: {dense_blocks}, sparse: {sparse_blocks}")
    # print(f"Dense blocks: {dense_blocks}")
    # print(f"Sparse blocks: {sparse_blocks}")
    if LOG_LEVEL > 1:
        print(f"Fill-ins in 2:4 sparse blocks: {fill_ins}")
        if len(fill_ins_per_block) > 0:
            print(f"Fill-ins per {p}x{b} block {statistics.mean(fill_ins_per_block)} \u00B1 {statistics.stdev(fill_ins_per_block)}"
            f", {statistics.mean(fill_ins_per_block) / (p * b) * 100}% \u00B1 "
            f"{statistics.stdev(fill_ins_per_block)  / (p * b) * 100}")


def pad_matrix(matrix, p=16, b=8):
    if matrix.shape[0] % p != 0 :
        num_rows = (matrix.shape[0] // p + 1) * p - matrix.shape[0]
        # matrix = np.resize(matrix, (matrix.shape[0] + num_rows, matrix.shape[1]))
        # matrix[-num_rows:, :] = 0
        zeros_to_append = np.zeros((num_rows, matrix.shape[1]))
        matrix = np.concatenate((matrix, zeros_to_append), axis=0)
        if LOG_LEVEL > 1:
            print(f'Padded {num_rows} 0-rows.')

    if matrix.shape[1] % b != 0:
        num_cols = (matrix.shape[1] // b + 1) * b - matrix.shape[1]
        # matrix = np.resize(matrix, (matrix.shape[0], matrix.shape[1] + num_cols))
        # matrix[:, -num_cols:] = 0
        zeros_to_append = np.zeros((matrix.shape[0], num_cols))
        matrix = np.concatenate((matrix, zeros_to_append), axis=1)
        if LOG_LEVEL > 1:
            print(f'Padded {num_cols} 0-columns.')

    return matrix


def coo_matrix2mtx(sparse_matrix, mtx_path):
    """
    Convert a sparse matrix in COO format as as scipy.sparse object to .mtx format.
    Mtx is just a standard EL (edgelist) format with additional header metadata.
    All we need to do is add the header and save the rest as a .mtx file.
    """
    mtx_lines = []
    M,N = sparse_matrix.shape
    lines = list(zip(sparse_matrix.row, sparse_matrix.col, sparse_matrix.data))
    # Add the header
    mtx_lines.append(f'%%MatrixMarket matrix coordinate real general\n% Generated by el2mtx\n{M} {N} {len(lines)}')
    # Convert the 0-based indexing to 1-based indexing
    for line in lines:
        mtx_lines.append(
            ' '.join([str(x)
                 for x in line])
        )
        # if there are no edge values (only two values per line), add a 1
        if len(line) == 2:
            mtx_lines[-1] += ' 1'
    
    # Save the .mtx file
    with open(mtx_path, 'w') as f:
        f.write('\n'.join(mtx_lines))

def pad_to_row_vect(filename: str, row_vect: int, filename_suffix: str = "_padded"):
    csr_matrix = sp.io.mmread(filename).tocsr()
    coo_matrix = sp.io.mmread(filename)
    # padded_edgelist = list(zip(coo_matrix.row, coo_matrix.col))
    padded_rows = list(coo_matrix.row)
    padded_cols = list(coo_matrix.col)
    n_rows = len(csr_matrix.indptr) - 1
    # iterate over blocks of sparse matrix in CSR format
    for r in range(0, n_rows, row_vect):
        block_cols = np.empty(0, dtype=np.int32)
        for rr in range(r, min(r+row_vect, n_rows)):
            block_cols = np.union1d(block_cols, csr_matrix.getrow(rr).indices)

        # add nonzeros to each row in a row_vect block
        for c in block_cols:
            for rr in range(r, min(r+row_vect, n_rows)):
                # padded_edgelist.append((rr, c))
                padded_rows.append(rr)
                padded_cols.append(c)

    padded_filename = filename.split('.')[0] + f"_{filename_suffix}_v{row_vect}.mtx"
    padded_coo = sp.sparse.coo_matrix((np.ones(len(padded_rows)), (padded_rows, padded_cols)))

    # write back padded_coo_matrix to file
    coo_matrix2mtx(padded_coo, padded_filename) 
    return padded_coo

# import sys
# import os.path as path
# if __name__ == "__main__":
#     n = len(sys.argv)
#     if n < 3:    
#         v = 8
#         matrix_dir = "test"
#     else:
#         v = int(sys.argv[1])
#         matrix_dir = sys.argv[2]
#     orginal_matrices = [f for f in os.listdir(matrix_dir) if f.endswith('.mtx') and "reordered" not in f]
#     for original in orginal_matrices: 
#         print(f"padding matrix {original} to {v} rows")
#         pad_to_row_vect(f'{path.join(matrix_dir,original)}', v)


def analyze_sparse_matrix(filename: str, row_block=16, col_block=8):
    csr_matrix = sp.io.mmread(filename).tocsr()

    n_rows = len(csr_matrix.indptr) - 1

    tot_num_blocks = 0
    # iterate over blocks of sparse matrix in CSR format
    for r in range(0, n_rows, row_block):
        block_cols = np.empty(0, dtype=np.int32)
        for rr in range(r, min(r+row_block, n_rows)):
            block_cols = np.union1d(block_cols, csr_matrix.getrow(rr).indices)
        max_col = max(block_cols) if len(block_cols) > 0 else 0
        max_num_blocks = (max_col-1)//col_block + 2
        blocks = np.zeros(max_num_blocks)
        for c in block_cols:
            blocks[c//col_block] += 1
        num_blocks = np.count_nonzero(blocks)
        tot_num_blocks += num_blocks       

    print(f"Total number of blocks: {tot_num_blocks}")    
    return tot_num_blocks


def analyze_matrix(data):
    # load matrix
    if LOG_LEVEL > 1:
        print(f"Matrix name: {data}")
    dense_array = get_dense_matrix(data)
    dense_array = pad_matrix(dense_array)
    if LOG_LEVEL > 1:
        print(f"Sparsity: {get_sparsity(dense_array)}")

    # do random permutations
    # num_permutations = get_num_permutations(dense_array)
    # permuted_rows_and_columns = randomly_permute_rows(randomly_permute_columns(matrix=dense_array, n=num_permutations),
    #                                                   n=num_permutations)
    # plot matrix
    # plot_original_and_transformed(dense_array, permuted_rows_and_columns)

    # print statistics for number sparse blocks, dense blocks, number of fill_ins, average number of sparse blocks
    if LOG_LEVEL > 1:
        print("Original:")
    iterate_over_blocks(dense_array)
    print()
    # print("Random permutations")
    # iterate_over_blocks(permuted_rows_and_columns)
    # print()


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    set_seed(seed=42)

    # iterate over all pairs of matrices: original and reordered using Paolo's method
    orginal_matrices = [f for f in os.listdir("test") if f.endswith('.mtx') and "reordered" not in f]
    for original in orginal_matrices: 
        pad_to_row_vect(f'test/{original}', 8)
        # if "ca-HepPh" not in original: 
        #     continue
        # if any(m in original for m in ["ia-wikiquote-user", "pdb1HYS"]):
        #     continue
        print(f"\n\n\n--------{original}--------\n\n\n")
        # try:
        print("Original:")
        # analyze_matrix(f'test/{original}')
        analyze_sparse_matrix(f'test/{original}')
        print("Reordered:")
        # analyze_matrix(f'test/{original.split(".")[0]}_reordered.mtx')
        analyze_sparse_matrix(f'test/{original.split(".")[0]}_reordered.mtx')

        print("Column Reordered:")
        analyze_sparse_matrix(f'test/{original.split(".")[0]}_reordered_transposed_reordered.mtx')
        # except Exception as e:
        #     print(f"Error: {e}")
        #     continue
        a = 1