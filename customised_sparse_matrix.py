import sys
import time
import numpy as np
import matplotlib.pyplot as plt

'''
Customised multi-dimensional sparse array for Q tables.

Functions:
1. Converting to dense arrays.
2. Finding certain values according to the index.
3. Updating certain values.
4. Averaging along certain (multiple) axes.
'''

class MSA():
    def __init__(self, size, sparse=None) -> None:
        '''
        size -- A 1D tuple, e.g., (2,3,4,5)
        '''
        if sparse is None:
            self.sparse = np.array([])
        else:
            self.sparse = sparse
        if type(size) != tuple:
            raise Exception("Non-tuple type input for size.")
        self.size = size
    
    # Return the dense shape of sparse array
    def shape(self):
        return self.size

    # Load an existing sparse
    def load_sparse(self, sparse):
        self.sparse = sparse
    
    # Convert the sparse array to a dense array
    def to_dense(self):
        dense = np.zeros(self.size)
        if self.sparse.shape[0] == 0:
            return dense
        elif len(self.sparse.shape) == 1:
            dense[tuple(self.sparse[:-1].astype(int))] = self.sparse[-1]
        else:
            for entry in self.sparse:
                dense[tuple(entry[:-1].astype(int))] = entry[-1]
        return dense

    # Return a certain value of the sparse array given the index
    def get(self, idx):
        '''
        idx -- A 1D list or Numpy array or tuple
        '''
        idx = np.array(idx)
        if len(idx) != len(self.size):
            raise Exception("Input index size doesn't match.")
        if np.any(idx < 0) or np.any(idx >= np.array(self.size)):
            raise Exception("Given index {} output range of size {}.".format(idx, self.size))
        # Find entry idx
        if self.sparse.shape[0] == 0:
            return 0
        elif len(self.sparse.shape) == 1:
            entry_idx = np.where(np.all(self.sparse[:-1]==idx, axis=0))[0]
        else:
            entry_idx = np.where(np.all(self.sparse[:,:-1]==idx, axis=1))[0]
        # Return value
        if entry_idx.size == 0:
            return 0
        else:
            if len(self.sparse.shape) == 1:
                return self.sparse[-1]
            else:
                return self.sparse[entry_idx[0]][-1]

    # Update a certain value of the sparse array given the index
    def update(self, idx, value):
        '''
        idx -- A 1D list or Numpy array
        '''
        idx = np.array(idx)
        if len(idx) != len(self.size):
            raise Exception("Input index size doesn't match.")
        if np.any(idx < 0) or np.any(idx >= np.array(self.size)):
            raise Exception("Given index {} out of range of size {}.".format(idx, self.size))
        # Find entry index
        if self.sparse.shape[0] == 0:
            self.sparse = np.append(self.sparse, np.append(idx, value))
            return
        elif len(self.sparse.shape) == 1:
            entry_idx = np.where(np.all(self.sparse[:-1]==idx, axis=0))[0]
        else:
            entry_idx = np.where(np.all(self.sparse[:,:-1]==idx, axis=1))[0]
        # Update value
        if entry_idx.size == 0:
            if value != 0:
                self.sparse = np.vstack([self.sparse, np.append(idx, value)])
        elif entry_idx.size == 1:
            if value == 0:
                self.sparse = np.delete(self.sparse, entry_idx[0], 0)
            else:
                self.sparse[entry_idx[0]][-1] = value
        else:
            raise Exception("Repeated entry detected, check the sparse.")

    # Average the sparse along one certain axis and return the mean sparse as a new sparse matrix
    def mean_single_axis(self, axis):
        '''
        axis -- An integer
        '''
        if axis < 0 or axis >= len(self.size):
            raise Exception("Axis {} out of range of size {}".format(axis, self.size))
        reduced_size = tuple(np.delete(np.array(self.size), axis))
        if self.sparse.shape[0] == 0:
            return type(self)(reduced_size, self.sparse)
        elif len(self.sparse.shape) == 1:
            reduced_sparse = np.delete(self.sparse, axis)
            return type(self)(reduced_size, reduced_sparse)
        else:
            reduced_sparse = np.delete(self.sparse, axis, axis=1)
            unique_entries, unique_idx = np.unique(reduced_sparse[:, :-1], axis=0, return_inverse=True)
            mean_values = np.zeros(unique_entries.shape[0])
            np.add.at(mean_values, unique_idx, self.sparse[:, -1])
            mean_values /= np.bincount(unique_idx)
            mean_sparse = np.column_stack((unique_entries, mean_values))
            return type(self)(reduced_size, mean_sparse)

    # Average the sparse along multiple axes and return the mean sparse as a new sparse matrix
    def mean(self, axes):
        '''
        axes - A list, array or tuple of axes to average along with
        '''
        # Convert axes to array and sort from low to high
        axes = np.array(axes)
        axes.sort()
        # Average the first axis
        axis = axes[0]
        axes = axes[1:] - 1
        mean_sparse = self.mean_single_axis(axis)
        # Average the rest axes
        while len(axes) > 0:
            axis = axes[0]
            axes = axes[1:] - 1
            mean_sparse = mean_sparse.mean_single_axis(axis)
        
        return mean_sparse
    
    # Save the sparse to path
    def save(self, path):
        np.save(path, self.sparse)


class new_MSA:
    def __init__(self, size, sparse=None):
        if sparse is None:
            self.sparse = np.empty((0, len(size) + 1))
        else:
            self.sparse = sparse
        if not isinstance(size, tuple):
            raise ValueError("Non-tuple type input for size.")
        self.size = size
    
    def shape(self):
        return self.size
    
    def load_sparse(self, sparse):
        self.sparse = sparse
    
    def to_dense(self):
        dense = np.zeros(self.size)
        for entry in self.sparse:
            dense[tuple(entry[:-1].astype(int))] = entry[-1]
        return dense

    def get(self, idx):
        idx = np.array(idx)
        if len(idx) != len(self.size) or np.any((idx < 0) | (idx >= np.array(self.size))):
            raise ValueError("Input index size doesn't match or is out of range.")
        entry_idx = np.where(np.all(self.sparse[:, :-1] == idx, axis=1))[0]
        return 0 if entry_idx.size == 0 else self.sparse[entry_idx[0], -1]

    def update(self, idx, value):
        idx = np.array(idx)
        if len(idx) != len(self.size) or np.any((idx < 0) | (idx >= np.array(self.size))):
            raise ValueError("Input index size doesn't match or is out of range.")
        entry_idx = np.where(np.all(self.sparse[:, :-1] == idx, axis=1))[0]
        if entry_idx.size == 0:
            if value != 0:
                self.sparse = np.vstack([self.sparse, np.append(idx, value)])
        elif entry_idx.size == 1:
            if value == 0:
                self.sparse = np.delete(self.sparse, entry_idx[0], axis=0)
            else:
                self.sparse[entry_idx[0], -1] = value
        else:
            raise ValueError("Repeated entry detected, check the sparse.")

    def mean_single_axis(self, axis):
        if axis < 0 or axis >= len(self.size):
            raise ValueError("Axis out of range.")
        reduced_size = tuple(np.delete(np.array(self.size), axis))
        if self.sparse.shape[0] == 0:
            return type(self)(reduced_size, self.sparse)
        reduced_sparse = np.delete(self.sparse, axis, axis=1)
        unique_entries, unique_idx = np.unique(reduced_sparse[:, :-1], axis=0, return_inverse=True)
        mean_values = np.zeros(unique_entries.shape[0])
        np.add.at(mean_values, unique_idx, self.sparse[:, -1])
        mean_values /= np.bincount(unique_idx)
        mean_sparse = np.column_stack((unique_entries, mean_values))
        return type(self)(reduced_size, mean_sparse)

    def mean(self, axes):
        # Convert axes to array and sort from low to high
        axes = np.array(axes)
        axes.sort()
        # Average the first axis
        axis = axes[0]
        axes = axes[1:] - 1
        mean_sparse = self.mean_single_axis(axis)
        # Average the rest axes
        while len(axes) > 0:
            axis = axes[0]
            axes = axes[1:] - 1
            mean_sparse = mean_sparse.mean_single_axis(axis)
        
        return mean_sparse
    
    def save(self, path):
        np.save(path, self.sparse)


if __name__ == '__main__':
    # # Basic function test
    # q = refined_MSA((2,3,2))
    # q.update((0,0,0), 3)
    # q.update((0,1,0), 2)
    # q.update((0,2,1), 4)
    # q.update((0,0,1), 6)
    # q.update((0,2,0), 3)
    # q.update((1,0,0), 1)
    # q.update((1,0,1), 3)
    # q.update((0,1,1), 1)
    # print(q.sparse)
    # print(q.shape())
    # print(q.to_dense())
    # print("Sparse memosize: {} bytes".format(sys.getsizeof(q.sparse)))
    # print("Dense memosize: {} bytes".format(sys.getsizeof(q.to_dense())))
    # mq = q.mean((1,2))
    # print(mq.sparse)
    # print(mq.shape())
    # print(mq.to_dense())
    # print("Sparse memosize: {} bytes".format(sys.getsizeof(mq.sparse)))
    # print("Dense memosize: {} bytes".format(sys.getsizeof(mq.to_dense())))

    # Efficiency test
    dim = [3,5,3,5,3,5,3,5,3,5,3,5,3]
    q = MSA(tuple(dim))
    update_t = []
    get_t = []
    mean_t = []
    msize = []
    for i in range(300):
        print(i)
        random_dim = np.random.randint(0, dim)

        start_t = time.time()
        q.update(tuple(random_dim), np.random.randn())
        update_t.append(time.time() - start_t)

        start_t = time.time()
        q.get(tuple(random_dim))
        get_t.append(time.time() - start_t)

        start_t = time.time()
        q.mean((0,1,2,3,4,5))
        mean_t.append(time.time() - start_t)

        msize.append(sys.getsizeof(q.sparse))

    plt.figure()
    plt.plot(update_t, label="Update() time")
    plt.plot(get_t, label="Get() time")
    plt.plot(mean_t, label="Mean() time")
    plt.ylabel("Time (s)")
    plt.legend()

    plt.figure()
    plt.plot(msize, label="Memory size")
    plt.ylabel("Memory size (bytes)")
    plt.legend()

    plt.show()
