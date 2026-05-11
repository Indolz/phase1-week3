import numpy as np

# ── The problem with Python lists ─────────────────────────────────
py_list = [1, 2, 3, 4, 5]

# To multiply every element by 2 in a list you need a loop
doubled_list = [x * 2 for x in py_list]
print("List doubled:", doubled_list)

# ── NumPy array — operates on entire array at once ────────────────
arr = np.array([1, 2, 3, 4, 5])
doubled_arr = arr * 2          # no loop needed
print("Array doubled:", doubled_arr)

# ── Creating arrays ───────────────────────────────────────────────
zeros = np.zeros((3, 4))    # 3 rows, 4 columns of zeros
ones = np.ones((2,3))       # 2 rows, 3 columns of ones
rng = np.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
lin = np.linspace(0, 1, 5)  # 5 evenly spaced values from 0 to 1

print("\nZeros shape:", zeros.shape)
print("Ones:\n", ones)
print("Arange:", rng)
print("Linspace", lin)

# ── Key attributes every ML engineer checks ───────────────────────
matrix = np.array([[1, 2, 3],
                  [4, 5, 6]])
print("\nShape:", matrix.shape)  # (2, 3) — 2 rows, 3 columns
print("Dimensions:", matrix.ndim) # 2
print("Total elements:", matrix.size) # 6
print("Data type:", matrix.dtype)   # int64


