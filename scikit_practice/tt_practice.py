from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_iris

X , y = load_iris(return_X_y=True)

# use the shuffle parameter to shuffle the data before splitting without randomness
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
print("X_train: ", X_train)
print("X_test: ", X_test)
print("y_train:", y_train)
print("y_test: ", y_test)
print("\n" + "="*50 + "\n")

# use the shuffle parameter to shuffle the data before splitting with randomness
X_train_rs, X_test_rs, y_train_rs, y_test_rs = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=55)
print("X_train_rs: ", X_train_rs)
print("X_test_rs: ", X_test_rs)
print("y_train_rs:", y_train_rs)
print("y_test_rs: ", y_test_rs)
print("\n" + "="*50 + "\n")

#not shuffling the data before splitting
X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(X, y, test_size=0.3,shuffle=False)
print("X_train without shuffling: ", X_train_ns)
print("X_test without shuffling: ", X_test_ns)
print("y_train without shuffling:", y_train_ns)
print("y_test without shuffling: ", y_test_ns)
print("\n" + "="*50 + "\n")

# Manual shuffling before splitting the dataset
from sklearn.utils import shuffle as sk_shuffle
X_shuffled, y_shuffled = sk_shuffle(X, y, random_state=55)
X_train_msh, X_test_msh, y_train_msh, y_test_msh = train_test_split(X_shuffled, y_shuffled, train_size=0.7)
print("X_train after manual shuffling: ", X_train_msh)
print("X_test after manual shuffling: ", X_test_msh)
print("y_train after manual shuffling:", y_train_msh)
print("y_test after manual shuffling: ", y_test_msh)
print("\n" + "="*50 + "\n")


print("##############")
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.utils import shuffle

# Dense array - the regular numpy array
X_dense = np.array([[1., 0.], 
                     [2., 1.], 
                     [0., 0.]])

print("Dense Array:")
print(X_dense)
print(f"\nShape: {X_dense.shape}")
print(f"Total elements: {X_dense.size}")
print(f"Memory used: {X_dense.nbytes} bytes")
print(f"Data type: {X_dense.dtype}")
print("\nHow it's stored in memory:")
print("All 6 values are stored: [1., 0., 2., 1., 0., 0.]")
print("Even the zeros take up space!")

print("\n" + "="*70)
print("PART 2: UNDERSTANDING SPARSE ARRAYS (COO FORMAT)")
print("="*70)

# Convert to sparse (COO = Coordinate format)
X_sparse_coo = coo_matrix(X_dense)

print("Sparse Array (COO format):")
print(X_sparse_coo)
print(f"\nShape: {X_sparse_coo.shape}")
print(f"Non-zero elements: {X_sparse_coo.nnz}")
print(f"Sparsity: {(1 - X_sparse_coo.nnz / X_dense.size) * 100:.1f}%")

print("\nHow COO format works:")
print("It stores 3 arrays:")
print(f"  row indices:    {X_sparse_coo.row}")
print(f"  column indices: {X_sparse_coo.col}")
print(f"  values:         {X_sparse_coo.data}")

print("\nInterpretation:")
for i in range(len(X_sparse_coo.data)):
    print(f"  Position ({X_sparse_coo.row[i]}, {X_sparse_coo.col[i]}) "
          f"has value {X_sparse_coo.data[i]}")

print("\nZeros are NOT stored at all!")

print("\n" + "="*70)
print("PART 3: SPARSE ARRAY FORMATS (CSR)")
print("="*70)

# CSR = Compressed Sparse Row (more efficient for operations)
X_sparse_csr = csr_matrix(X_dense)

print("Sparse Array (CSR format):")
print(X_sparse_csr)
print("\nCSR is more efficient for:")
print("  - Row slicing")
print("  - Matrix-vector products")
print("  - Machine learning operations")

print("\nConvert back to dense:")
print(X_sparse_csr.toarray())

print("\n" + "="*70)
print("PART 4: SHUFFLING DENSE AND SPARSE TOGETHER")
print("="*70)

X_dense = np.array([[1., 0.], [2., 1.], [0., 0.]])
X_sparse = coo_matrix(X_dense)
y = np.array([0, 1, 2])

print("Before shuffle:")
print("X_dense:\n", X_dense)
print("\nX_sparse (as dense):\n", X_sparse.toarray())
print("y:", y)

# Shuffle all three together - they stay synchronized!
X_dense_shuffled, X_sparse_shuffled, y_shuffled = shuffle(
    X_dense, X_sparse, y, random_state=0
)

print("\nAfter shuffle (random_state=0):")
print("X_dense_shuffled:\n", X_dense_shuffled)
print("\nX_sparse_shuffled (as dense):\n", X_sparse_shuffled.toarray())
print("y_shuffled:", y_shuffled)

print("\nNotice: All three were shuffled in the SAME order!")
print("Row 2 [0., 0.] with label 2 moved to position 0")
print("Row 1 [2., 1.] with label 1 moved to position 1")
print("Row 0 [1., 0.] with label 0 moved to position 2")

print("\n" + "="*70)
print("PART 5: MEMORY COMPARISON WITH LARGER EXAMPLE")
print("="*70)

# Create a large sparse matrix
size = 1000
density = 0.01  # Only 1% non-zero

# Dense version
large_dense = np.random.choice([0, 1], size=(size, size), p=[1-density, density])
print(f"Large matrix: {size}x{size} = {size*size:,} elements")
print(f"Density: {density*100}% non-zero")
print(f"Dense memory: {large_dense.nbytes:,} bytes ({large_dense.nbytes/1024/1024:.2f} MB)")

# Sparse version
large_sparse = csr_matrix(large_dense)
# Approximate sparse memory (data + indices + indptr)
sparse_memory = (large_sparse.data.nbytes + 
                 large_sparse.indices.nbytes + 
                 large_sparse.indptr.nbytes)
print(f"Sparse memory: {sparse_memory:,} bytes ({sparse_memory/1024/1024:.2f} MB)")
print(f"Memory saved: {(1 - sparse_memory/large_dense.nbytes)*100:.1f}%")

print("\n" + "="*70)
print("PART 6: WHEN TO USE EACH FORMAT")
print("="*70)

print("""
USE DENSE when:
  ✓ Matrix is small (< 10,000 elements)
  ✓ Matrix has few zeros (< 50% sparsity)
  ✓ You need fast random access to elements
  ✓ Doing many element-wise operations

USE SPARSE when:
  ✓ Matrix is large (> 100,000 elements)
  ✓ Matrix has many zeros (> 50-70% sparsity)
  ✓ Memory is limited
  ✓ Working with text data, networks, or recommendation systems

REAL EXAMPLES:
  • Text analysis (TF-IDF): 95-99% sparse
  • Netflix recommendations: 99.9% sparse (users rate <1% of movies)
  • Social networks: 99.99% sparse (you're not friends with everyone)
  • Image processing: Varies (depends on image content)
""")

print("="*70)
print("PART 7: OPERATIONS ON SPARSE MATRICES")
print("="*70)

X_sparse = csr_matrix([[1, 0, 0],
                       [0, 2, 0],
                       [0, 0, 3]])

print("Original sparse matrix:")
print(X_sparse.toarray())

# Matrix multiplication
result = X_sparse.dot(X_sparse.T)
print("\nMatrix multiplication (X @ X.T):")
print(result.toarray())

# Element access (careful - this is slow for sparse!)
print(f"\nElement at (1,1): {X_sparse[1,1]}")

# Sum
print(f"Sum of all elements: {X_sparse.sum()}")

# Get non-zero elements
print(f"Non-zero elements: {X_sparse.data}")

print("\n" + "="*70)
print("KEY TAKEAWAY")
print("="*70)
print("""
Dense Array: Stores EVERYTHING (like a full spreadsheet)
  [1, 0, 0]
  [0, 2, 0]  → Stores: 1,0,0,0,2,0,0,0,3 (9 numbers)
  [0, 0, 3]

Sparse Array: Stores ONLY non-zeros with positions
  [1, 0, 0]
  [0, 2, 0]  → Stores: [(0,0,1), (1,1,2), (2,2,3)] (3 coordinates)
  [0, 0, 3]

The shuffle function keeps both formats synchronized!
""")