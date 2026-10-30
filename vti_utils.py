import numpy as np

def thomsen_to_voigt(vp0: float, vs0: float, epsilon: float, delta: float, gamma: float, rho: float) -> np.ndarray:
    c33 = rho * (vp0 ** 2)
    c44 = rho * (vs0 ** 2)
    c11 = c33 * (1.0 + 2.0 * epsilon)
    c66 = c44 * (1.0 + 2.0 * gamma)
    
    val = 2.0 * delta * c33 * (c33 - c44) + (c33 - c44) ** 2
    c13 = np.sqrt(max(val, 0.0)) - c44
    c12 = c11 - 2.0 * c66
    
    C = np.zeros((6, 6))
    C[0, 0] = c11; C[1, 1] = c11; C[2, 2] = c33
    C[3, 3] = c44; C[4, 4] = c44; C[5, 5] = c66
    C[0, 1] = c12; C[1, 0] = c12
    C[0, 2] = c13; C[2, 0] = c13
    C[1, 2] = c13; C[2, 1] = c13
    return C

def voigt_to_tensor4d(C_voigt: np.ndarray, rho: float) -> np.ndarray:
    A = np.zeros((3, 3, 3, 3))
    voigt_map = {
        (0, 0): 0, (1, 1): 1, (2, 2): 2,
        (1, 2): 3, (2, 1): 3, (0, 2): 4,
        (2, 0): 4, (0, 1): 5, (1, 0): 5
    }
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    A[i, j, k, l] = C_voigt[voigt_map[(i, j)], voigt_map[(k, l)]] / rho
    return A