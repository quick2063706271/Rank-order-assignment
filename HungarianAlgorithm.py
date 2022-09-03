import numpy as np


# Reference (Code develped from): https://python.plainenglish.io/hungarian-algorithm-introduction-python-implementation-93e7c0890e15

def mark_min_zero_row(zero_mat, mark_zero):
    """
    The function can be split into two steps:
    #1 The function is used to find the row which containing the fewest 0.
    #2 Select the zero number on the row, and then marked the element corresponding row and column as False
    :param zero_mat:
    :param mark_zero:
    """

    min_zero_row = [10000000, -1]  # Store [(number of 0), (row idx)]
    # 1. find row that has least number of zeros
    for row_num in range(zero_mat.shape[0]):
        num_zero = np.sum(zero_mat[row_num] == True)
        if 0 < num_zero < min_zero_row[0]:
            min_zero_row = [num_zero, row_num]

    # Store the marked coordinates into mark_zero
    zero_index = np.where(zero_mat[min_zero_row[1]] == True)[0][0]
    mark_zero.append((min_zero_row[1], zero_index))

    # 2. Marked both row and column as False
    zero_mat[min_zero_row[1], :] = False
    zero_mat[:, zero_index] = False


def mark_matrix(matrix):
    """
    Finding the returning possible solutions for LAP problem.
    :param matrix:
    """

    # Transform the matrix to boolean matrix(0 = True, others = False)
    cur_matrix = matrix
    zero_boolean_matrix = (cur_matrix == 0)
    zero_boolean_matrix_copy = zero_boolean_matrix.copy()

    # Store possible answer positions by marked_zero
    marked_zero = []  # store [(idxes that marked zero), (row Idx)]
    while True in zero_boolean_matrix_copy:
        mark_min_zero_row(zero_boolean_matrix_copy, marked_zero)

    # Recording the row and column positions separately.
    marked_zero_row = []
    marked_zero_col = []
    for i in range(len(marked_zero)):
        marked_zero_row.append(marked_zero[i][0])
        marked_zero_col.append(marked_zero[i][1])

    # Store indexes of row that do not contain marked 0 elements
    non_marked_row = list(set(range(cur_matrix.shape[0])) - set(marked_zero_row))

    marked_cols = []
    check_switch = True
    while check_switch:
        check_switch = False
        # Search for any unmarked 0 elements in their column
        for i in range(len(non_marked_row)):
            row_array = zero_boolean_matrix[non_marked_row[i], :]

            for j in range(row_array.shape[0]):
                # Step 2-2-2
                if row_array[j] == True and j not in marked_cols:
                    # Step 2-2-3
                    marked_cols.append(j)
                    check_switch = True

        for row_num, col_num in marked_zero:
            # Step 2-2-4
            if row_num not in non_marked_row and col_num in marked_cols:
                # Step 2-2-5
                non_marked_row.append(row_num)
                check_switch = True
    # Step 2-2-6
    marked_rows = list(set(range(matrix.shape[0])) - set(non_marked_row))

    return marked_zero, marked_rows, marked_cols


def adjust_matrix(mat, cover_rows, cover_cols):
    """
    1. Find the minimum value that is not in marked_rows and marked_cols
    2. Subtract the elements which are not in marked_rows nor marked_cols form the minimum values
    3. Add the element in marked_rows to the min value
    :param mat:
    :param cover_rows:
    :param cover_cols:
    :return:
    """
    cur_mat = mat
    non_zero_element = []

    # 1. Find the minimum value that is not in marked_rows and marked_cols
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    non_zero_element.append(cur_mat[row][i])
    min_val = min(non_zero_element)

    # 2. Subtract the elements which are not in marked_rows nor marked_cols form the minimum values
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    cur_mat[row, i] = cur_mat[row, i] - min_val
    # 3. Add the element in marked_rows to the min value
    for row in range(len(cover_rows)):
        for col in range(len(cover_cols)):
            cur_mat[cover_rows[row], cover_cols[col]] = cur_mat[cover_rows[row], cover_cols[col]] + min_val
    return cur_mat


def hungarian_algorithm(matrix):
    """Return the result of linear assignment from matrix using hungarian algorithm

    :param matrix:
    :return:
    """

    dimension = matrix.shape[0]
    cur_matrix = matrix

    # 1. Subtract every column and every row with its internal minimum
    for row_idx in range(dimension):
        min_val = np.min(cur_matrix[row_idx])
        cur_matrix[row_idx] = cur_matrix[row_idx] - min_val
    for col_idx in range(matrix.shape[1]):
        min_val = np.min(cur_matrix[:, col_idx])
        cur_matrix[:, col_idx] = cur_matrix[:, col_idx] - min_val

    zero_count = 0
    # Repeat step 2 and 3 until the zero_count == dimension
    result = None
    while zero_count < dimension:
        # 2
        result, marked_rows, marked_cols = mark_matrix(cur_matrix)
        zero_count = len(marked_cols) + len(marked_rows)
        # 3
        if zero_count < dimension:
            cur_matrix = adjust_matrix(cur_matrix, marked_rows, marked_cols)

    return result
