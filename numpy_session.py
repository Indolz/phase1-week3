import numpy as np

# Python list — stores different types, scattered in memory
py_list = [1, 2, 3, 4, 5]

# NumPy array — same type, contiguous memory, vectorised operations
arr = np.array([1, 2, 3, 4, 5])

# The key difference — no loop needed with NumPy
print("List x 2:", [x * 2 for x in py_list]) # needs a loop
print("Array x 2:", arr * 2)                 # operates on whole array

# Creating arrays different ways
zeros = np.zeros((3, 4))    # 3 rows, 4 cols, all zeros
ones = np.ones((2, 3))      # 2 rows, 3 cols, all ones
rng = np.arange(0, 10, 2)   # [0, 2, 4, 6, 8] - like range()
lin = np.linspace(0, 1, 5)  # 5 values evenly spaced from 0 to 1

print("\nZeros:\n", zeros)
print("\nOnes:\n", ones)
print("\nArange:", rng)
print("\nnLinspace:", lin)

# Three things to always check about an array
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print("\n--- Array info ---")
print("Shape:", matrix.shape)           # (2, 3) — rows, columns
print("Dimensions:", matrix.ndim)       # 2 — it's a 2D array
print("Total elements:", matrix.size)   # 6
print("Data type:", matrix.dtype)       # int64


print("\n" + "="*50)
print("BLOCK 2 - Indexiing and Slicing")
print("="*50)

data = np.array([[10, 20, 30, 40],
                 [50, 60, 70, 80],
                 [90, 100, 110, 120]])

print("Full array:\n", data)
print("Shape:", data.shape)

# Single elements - [row, column]
print("\nRow 0, Column 2:", data[0, 2])
print("Last row, last column:", data[-1, -1])

# Entire rows and columns
print("\nFirst row:", data[0, :])
print("First column:", data[:, 0])

# Slices - [start:stop] where stop is excluded
print("\nFirst 2 rows, first 2 columns:\n", data[:2, :2])

# Boolean indexing - filter by condiction
print("\nAll values greater than 60:", data[data > 60])

# Copy before modifying - very important habit
copy = data.copy()
copy[copy < 50] = 0
print("\nAfter setting values below 50 to zero:\n", copy)

# Reshaping - same data, different shape
flat = np.arange(12)
print("\nFlat array", flat)
print("Reshaped to (3,4):\n", flat.reshape(3, 4))
print("Reshape with -1 (auto):", flat.reshape(2, -1).shape)


print("\n" + "="*50)
print("BLOCK 3 - Broadcasting")
print("="*50)

# Broadcasting means NumPy can do maths between arrays
# of different shapes — it stretches the smaller one automatically

scores = np.array([[85, 90, 78],
                   [92, 88, 95],
                   [70, 75, 80]])

print("Original scores:\n", scores)
print("Shape:", scores.shape)   #(3, 3)

# Add 5 points to every single score — no loop needed
curved = scores + 5
print("\nAfter adding 5 to everything:\n", curved)

# Subtract each column's mean from that column
# This is called "normalisation" — you'll use it constantly in ML
col_means = scores.mean(axis=0)     # mean of each column
print("\nColumn means:", col_means)

normalised = scores - col_means     # NumPy broadcasts col_means across rows
print("Normalised (subtract column mean):\n", normalised)

# Common maths operations on arrays
print("\n--- Array maths ---")
print("Sum of all:", scores.sum())
print("Sum per column:",scores.sum(axis=0))
print("Sum per row:", scores.sum(axis=1))
print("Max value:", scores.max())
print("Max per row:", scores.max(axis=1))
print("Mean:", scores.mean())
print("Standard deviation:", scores.std().round(2))


print("\n" + "="*50)
print("BLOCK 4 - Real ML Operations")
print("="*50)

# In ML, data usually comes as a 2D array
# rows = samples, columns = features
# This represents 5 students with 3 exam scores each
students = np.array([
    [85, 90, 78],
    [92, 88, 95],
    [70, 75, 80],
    [88, 82, 91],
    [65, 70, 72]
])

# Calculate each student's average score — axis=1 means across columns
student_averages = students.mean(axis=1)
print("\nEach student's average:", student_averages.round(1))

# Find which student has the highest average
best_student = student_averages.argmax()  # returns the INDEX of max value
print("Best student index:", best_student)
print("Their scores:", students[best_student])

# Normalise - subtract mean, divide by std
# This puts all values on the same scale, - very common in ML
mean = students.mean(axis=0)  # mean of each column
std = students.std(axis=0)    # std of each column
normalised = (students - mean) / std

print("\nNormalised scores:\n", normalised.round(2))
print("Check - normalised column means:", normalised.mean(axis=0).round(10))  # should be ~0
print("Check - normalised column std:", normalised.std(axis=0).round(2))    # should be ~1

# Dot product - a fundamental to neural networks
# Every layer of a neural network is: output = input . weights
weights = np.array([0.5, 0.3, 0.2]) # importance of each exam
weighted_scores = students.dot(weights)  # shape (5,) - one score per student
print("\nWeighted final scores:", weighted_scores.round(1))