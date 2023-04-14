import numpy as np

def load_calib(dir):
    with open(dir, 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        p = np.reshape(params, (3,4))
        k = p[0:3, 0:3]

    return k, p

def _load_poses(filepath):
    """
    Loads the GT poses
    Parameters
    ----------
    filepath (str): The file path to the poses file
    Returns
    -------
    poses (ndarray): The GT poses
    """
    poses = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
            print(T)
            print("/////////////")
    return poses

_load_poses('datasets/00.txt')
    
# k, p = load_calib('datasets/00/calib.txt')

# print(k)

# print(p)


