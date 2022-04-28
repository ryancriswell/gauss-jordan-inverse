import GaussJordanInverse


matrix = GaussJordanInverse.input_matrix()

print('The original matrix is: \n', matrix, '\n')
print('The Inverse of the matrix is: \n', GaussJordanInverse.inverse(matrix))
