import numpy as np


# Reference (Code developed from):
# https://python.plainenglish.io/hungarian-algorithm-introduction-python-implementation-93e7c0890e15

def mark_min_zero_row(zero_mat, mark_zero):
    """
    The function can be split into two steps:
    #1 The function is used to find the row which containing the fewest 0.
    #2 Select the zero number on the row, and then marked the element corresponding row and column as False
    :param zero_mat: a boolean matrix
    :param mark_zero: a list store (row_idx, col_idx) that are marked zero
    """

    min_num_zero = 1000000
    min_row_idx = -1
    # 1. find row that has least number of zeros
    for i in range(zero_mat.shape[0]):
        num_zero = np.sum(zero_mat[i, ] == True)
        if 0 < num_zero < min_num_zero:
            min_num_zero = num_zero
            min_row_idx = i

    # Store the marked coordinates into mark_zero
    zero_idxes = np.where(zero_mat[min_row_idx, ] == True)[0][0]
    mark_zero.append((min_row_idx, zero_idxes))

    # 2. Marked both row and column as False
    zero_mat[min_row_idx, :] = False
    zero_mat[:, zero_idxes] = False
    return


def mark_matrix(matrix):
    """
    Finding the possible solutions for hungarian algorithm.
    :param matrix: a numpy matrix
    :return (marked_zero, marked_rows, marked_cols)
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
                if row_array[j] == True and j not in marked_cols:
                    # Store them in the marked_cols
                    marked_cols.append(j)
                    check_switch = True

        for row_num, col_num in marked_zero:
            # Compare column indexes stored in marked_zero and marked_cols
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
    :param mat: a numpy matrix
    :param cover_rows: an integer list
    :param cover_cols: an integer list
    :return: a matrix after adjustment made above
    """

    # 1. Find the minimum value that is not in marked_rows and marked_cols
    min_val = 10000000
    cur_mat = mat
    not_cover_rows = set(range(len(cur_mat))) - set(cover_rows)
    not_cover_cols = set(range(len(cur_mat[0]))) - set(cover_cols)
    for i in not_cover_rows:
        for j in not_cover_cols:
            if min_val > cur_mat[i][j]:
                min_val = cur_mat[i][j]

    # 2. Subtract the elements which are not in marked_rows nor marked_cols from the minimum values
    for i in not_cover_rows:
        for j in not_cover_cols:
            cur_mat[i, j] = cur_mat[i, j] - min_val
    # 3. Add the element in marked_rows to the min value
    for i in range(len(cover_rows)):
        for j in range(len(cover_cols)):
            cur_mat[cover_rows[i], cover_cols[j]] = cur_mat[cover_rows[i], cover_cols[j]] + min_val
    return cur_mat


def hungarian_algorithm(matrix):
    """Return the result of linear assignment from matrix using hungarian algorithm

    :param matrix: a numpy matrix
    :return: indices that optimizes the the linear assignment problem
    """

    cur_matrix = matrix

    # 1. Subtract every column and every row with its internal minimum
    for row_idx in range(matrix.shape[0]):
        min_val = np.min(cur_matrix[row_idx])
        cur_matrix[row_idx] = cur_matrix[row_idx] - min_val
    for col_idx in range(matrix.shape[1]):
        min_val = np.min(cur_matrix[:, col_idx])
        cur_matrix[:, col_idx] = cur_matrix[:, col_idx] - min_val

    zero_count = 0
    # Repeat step 2 and 3 until the zero_count == dimension
    result = None
    dimension = matrix.shape[0]
    while zero_count < dimension:
        # 2
        result, marked_rows, marked_cols = mark_matrix(cur_matrix)
        zero_count = len(marked_cols) + len(marked_rows)
        # 3
        if zero_count < dimension:
            cur_matrix = adjust_matrix(cur_matrix, marked_rows, marked_cols)

    return result


def load_preference(matrix, doctors_capacity):
    """Replicate columns in matrix according to different doctors_capacity

    :param doctors_capacity:
    :param matrix: a numpy matrix
    :return: a modified matrix
    """
    hungarian_matrix = matrix.copy()
    cur_index = 0
    for i in range(len(doctors_capacity)):
        capacity = doctors_capacity[i]
        hungarian_matrix = np.insert(hungarian_matrix,
                           (capacity - 1) * [cur_index],
                           matrix[:, [cur_index]],
                           axis=1)
        cur_index += capacity
    return hungarian_matrix


if __name__ == "__main__":
    doctors_capacity = [2, 3, 1, 5]
    preference = np.array([[0, 1, 2, 3], [0, 3, 1, 2], [1, 3, 2, 0], [3, 1, 0, 2], [3, 2, 0, 1]])
    print(preference)
    hungarian_matrix = preference[:, 0]
    cur_index = 0
    for i in range(len(doctors_capacity)):
        capacity = doctors_capacity[i]
        preference = np.insert(preference,
                               (capacity - 1) * [cur_index],
                               preference[:, [cur_index]],
                               axis=1)
        cur_index += capacity
    result = hungarian_algorithm(preference)
    sorted_by_second = sorted(result, key=lambda tup: tup[0])
    print(sorted_by_second)
