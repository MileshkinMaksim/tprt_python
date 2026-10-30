import numpy as np
import scipy.optimize as opt
from layers import HomogeneousAnisotropicLayer
from kinematics_3d import solve_christoffel

def compute_refracted_slowness(p_inc: np.ndarray, layer_from: HomogeneousAnisotropicLayer, 
                               layer_to: HomogeneousAnisotropicLayer, mode_idx: int) -> np.ndarray | None:
    """
    Рассчитывает вектор медленности преломленного луча на горизонтальной границе VTI-сред.
    Возвращает None при полном внутреннем отражении.
    """
    px, py = p_inc[0], p_inc[1]
    A_to = layer_to.get_tensors()
    sign_z = 1.0 if p_inc[2] >= 0 else -1.0

    def christoffel_error(pz_val):
        p_vec = np.array([px, py, pz_val[0]])
        p_mag = np.linalg.norm(p_vec)
        if p_mag == 0:
            return 1.0
        n_vec = p_vec / p_mag
        V, _, _ = solve_christoffel(A_to, n_vec, mode_idx)
        return (p_mag * V) - 1.0

    pz_guess = p_inc[2]
    res = opt.root(christoffel_error, [pz_guess], method='hybr')
    
    if not res.success or np.isnan(res.x[0]):
        return None  # Полное внутреннее отражение

    pz_refracted = abs(res.x[0]) * sign_z
    
    if abs(christoffel_error([pz_refracted])) > 1e-5:
        return None

    return np.array([px, py, pz_refracted])


def compute_reflected_slowness(p_inc: np.ndarray) -> np.ndarray:
    """
    Рассчитывает вектор медленности отраженного луча от горизонтальной границы Z=const.
    """
    return np.array([p_inc[0], p_inc[1], -p_inc[2]])