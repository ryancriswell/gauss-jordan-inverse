import numpy as np

# Constant used as a cutoff point for numbers that should be treated as zero.
# Used for round off error in calculating the determinant to check if the matrix is invertible.
ROUND_OFF = 1.0e-12


# Generate a matrix from input.
def input_matrix():
    n = int(input('Enter the number of rows and columns for the nxn matrix:\n'))
    matrix = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            matrix[i][j] = input(f'Enter a value for position {i+1}, {j+1}:\n')
    return matrix


# Input matrix and indices for the two rows you want to swap.
def swap_rows(matrix, r1, r2):
    temp = matrix[r1]
    matrix[r1] = matrix[r2]
    matrix[r2] = temp
    return matrix


# Calculates the inverse of the passed in matrix using Gauss-Jordan Elimination
def inverse(matrix):
    matrix = np.array(matrix, float)

    # If the determinant of the matrix is less than the round off we assume it's 0 this not invertible
    if np.linalg.det(matrix) < ROUND_OFF:
        return 'Not Invertible'

    n = len(matrix[0])
    identity = np.identity(len(matrix), float)

    for k in range(n):
        # Partial pivoting checking the absolute value of the diagonal element (pivots).
        if np.fabs(matrix[k, k]) < ROUND_OFF:
            # Iterate through the elements under the pivot element (column k).
            for i in range(k + 1, n):
                # Interchange rows to get the largest absolute value in the pivot (diagonal).
                if np.fabs(matrix[i, k]) > np.fabs(matrix[k, k]):
                    for j in range(k, n):
                        swap_rows(matrix, i, k)
                        swap_rows(identity, i, k)
                        break

        # Divide the lower rows by the pivot row to get 1 or 0 as the leading entry.
        pivot = matrix[k, k]
        for j in range(n):
            # Divide the rows below the pivot.
            matrix[k, j] /= pivot
            identity[k, j] /= pivot

        # Forward elimination
        for i in range(n):
            if i != k and matrix[i, k] != 0:
                scalar = matrix[i, k]
                # Subtract row i by a scalar multiple of row k.
                for j in range(n):
                    matrix[i, j] -= scalar * matrix[k, j]
                    identity[i, j] -= scalar * identity[k, j]
    return identity
