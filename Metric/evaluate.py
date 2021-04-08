import numpy as np

def distance(v1, v2, d_type='d1'):
    # check same shape
    assert v1.shape == v2.shape, "shape of two vectors need to be same!"

    if d_type == 'd1':
        return np.sum(np.absolute(v1 - v2))
    elif d_type == 'd2':
        return np.sum((v1 - v2) ** 2)
    elif d_type == 'd2-norm':
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == 'cosine':
        return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    elif d_type == 'square':
        return np.sum((v1 - v2) ** 2)