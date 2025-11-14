import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_1d(input_array, kernel_array):
    return len(input_array) - len(kernel_array) + 1


# -----------------------------------------------
# Example:
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(f"Task 1 Output: {compute_output_size_1d(input_array, kernel_array)}")


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------

def convolve_1d(input_array, kernel_array):
    out_size = compute_output_size_1d(input_array, kernel_array)
    output = np.zeros(out_size)
    for i in range(out_size):
        output[i] = np.sum(input_array[i:i+len(kernel_array)] * kernel_array)
    return output
    # Tip: start by initializing an empty output array (you can use your function above to calculate the correct size).
    # Then fill the cells in the array with a loop.
    

# -----------------------------------------------
# Another tip: write test cases like this, so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 6])
kernel_array = np.array([1, 0, -1])
print(f"Task 2 Output: {convolve_1d(input_array, kernel_array)}")

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_2d(input_matrix, kernel_matrix):
    h_in, w_in = input_matrix.shape
    h_k, w_k = kernel_matrix.shape
    return (h_in - h_k + 1, w_in - w_k + 1)


# -----------------------------------------------


# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------
def convolute_2d(input_matrix, kernel_matrix):
    out_h, out_w = compute_output_size_2d(input_matrix, kernel_matrix)
    output = np.zeros((out_h, out_w))
    kh, kw = kernel_matrix.shape
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(input_matrix[i:i+kh, j:j+kw] * kernel_matrix)
    return output   # Tip: same tips as above, but you might need a nested loop here in order to
    # define which parts of the input matrix need to be multiplied with the kernel matrix.



# -----------------------------------------------
# Example:
input_matrix = np.array([[2, 2, 3], [4, 5, 6], [7, 8, 10]])

kernel_matrix = np.array([[1, 0],
                          [0, -1]])

print(f"Task 3 Output: {compute_output_size_2d(input_matrix, kernel_matrix)}")
print("Task 4 Output:")
print(convolute_2d(input_matrix, kernel_matrix))