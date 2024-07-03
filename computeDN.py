import numpy as np
import itertools


def compute_corr(wX, wY):
    """
    Input: weight matrices of networks X and Y
    Output: list containing all possible correspondences between X and Y.
    Each correspondence has r rows and c columns, where r = |X| and c=|Y|.
    """

    # Number of rows and columns
    r = wX.shape[0]
    c = wY.shape[0]

    # Get all binary matrices of size r*c. There are 2^(r*c) combinations
    num_combinations = 2 ** (r * c)
    M = np.array([list(np.binary_repr(i, width=r * c)) for i in range(num_combinations)], dtype=int)
    M = M.reshape((num_combinations, r, c))

    # Filter this list to obtain actual correspondences
    good = np.array(
        [np.all(np.sum(M[i], axis=0) > 0) and np.all(np.sum(M[i], axis=1) > 0) for i in range(num_combinations)])

    M = M[good]
    C = [M[i] for i in range(len(M))]

    return C


def compute_dis(wX, wY, R):
    """
    Input: weight matrices of networks X and Y, and a correspondence
    Output: distortion of correspondence
    """

    # Find nonzero elements in R as linear indices
    nz = np.flatnonzero(R)

    # Obtain pairs of elements in R; lack of symmetry means need to use permutations
    p = list(itertools.product(nz, repeat=2))
    pp = np.zeros(len(p))

    c_values = []

    for i, (z1, z2) in enumerate(p):
        x1, y1 = np.unravel_index(z1, R.shape, order='F')
        x2, y2 = np.unravel_index(z2, R.shape, order='F')

        c = wX[x1, x2] / wY[y1, y2]
        c_values.append(c)

    C = min(c_values)

    for i, (z1, z2) in enumerate(p):
        x1, y1 = np.unravel_index(z1, R.shape, order='F')
        x2, y2 = np.unravel_index(z2, R.shape, order='F')

        pp[i] = wX[x1, x2] / (C * wY[y1, y2])

    dis = max(pp)

    return dis

def compute_DN(wX, wY):
    """
    Input: weight matrices of networks X and Y
    Output: network distance between X and Y
    """

    C = compute_corr(wX, wY)
    cost = np.empty(len(C))
    cost.fill(np.nan)

    for i, corr in enumerate(C):
        cost[i] = compute_dis(wX, wY, corr)

    dN = np.min(np.log(cost))

    return dN


wX =  np.array([[2, 4, 8], [5, 6, 3], [10, 3, 6]])
wY =  np.array([[7, 6, 3], [9, 2, 9], [10, 2, 3]])
print(compute_DN(wX, wY))