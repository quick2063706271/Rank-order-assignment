import numpy as np

def hungarian(matrix):
    '''

    :param matrix:
    :return:
    '''
    dimension = matrix.shape[0]
    cur_matrix = matrix

    # 1. subtract internal minimum for each column and row
    for row_idx in range(matrix.shape[0]):
        cur_matrix[row_idx] = cur_matrix[row_idx] - np.min(cur_matrix[row_idx])

    for col_idx in range(matrix.shape[1]):
        cur_matrix[:, col_idx] = cur_matrix[:, col_idx] - np.min(cur_matrix[:, col_idx])
    zero_count = 0
    while zero_count < dimension:
