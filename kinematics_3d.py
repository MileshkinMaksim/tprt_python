import numpy as np
import scipy.optimize as opt
from layers import HomogeneousAnisotropicLayer

def solve_christoffel(A_tensor: np.ndarray, n_vec: np.ndarray, mode_idx: int):
    """
    Решает уравнение Кристоффеля для заданного единичного вектора фазовой нормали n_vec.
    
    Parameters:
        A_tensor: 4D-тензор плотности модулей размера (3, 3, 3, 3).
        n_vec: Единичный вектор фазовой нормали [nx, ny, nz].
        mode_idx: Индекс моды волны (0 - qS1, 1 - qS2, 2 - qP).
        
    Returns:
        V: Фазовая скорость (float).
        w: Вектор поляризации (3,).
        v_group: Вектор групповой скорости (3,).
    """
    # Матрица Кристоффеля Gamma_jl = A_ijkl * n_i * n_k
    Gamma = np.einsum('ijkl,i,k->jl', A_tensor, n_vec, n_vec)
    
    eigvals, eigvecs = np.linalg.eigh(Gamma)
    sorted_indices = np.argsort(eigvals)
    selected_mode = sorted_indices[mode_idx]
    
    V2 = eigvals[selected_mode]
    V = np.sqrt(max(V2, 1e-12))
    w = eigvecs[:, selected_mode]
    
    # Вектор групповой скорости: v_g^r = A_rjkl * n_k * n_l * w_j * w_k / V
    v_group = np.einsum('rjkl,k,l,j->r', A_tensor, n_vec, w, w) / V
    return V, w, v_group


def get_group_velocity_for_direction(dir_geom: np.ndarray, layer: HomogeneousAnisotropicLayer, mode_idx: int) -> np.ndarray:
    """
    Находит вектор групповой скорости v_group, соответствующий заданному
    геометрическому направлению dir_geom в VTI-среде.
    """
    norm_dir = np.linalg.norm(dir_geom)
    if norm_dir == 0:
        return np.zeros(3)
    d_hat = dir_geom / norm_dir
    
    phi = np.arctan2(d_hat[1], d_hat[0])
    A = layer.get_tensors()

    def objective(theta_arr):
        th = theta_arr[0]
        n_guess = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
        _, _, v_calc = solve_christoffel(A, n_guess, mode_idx)
        v_norm = np.linalg.norm(v_calc)
        if v_norm == 0:
            return 1.0
        return 1.0 - np.dot(v_calc / v_norm, d_hat)

    # Оценка начального зенитного угла из геометрии
    theta_guess = np.arccos(np.clip(d_hat[2], -1.0, 1.0))
    res = opt.minimize(objective, [theta_guess], method='Nelder-Mead', options={'xatol': 1e-7, 'fatol': 1e-7})
    
    opt_theta = res.x[0]
    n_opt = np.array([np.sin(opt_theta) * np.cos(phi), np.sin(opt_theta) * np.sin(phi), np.cos(opt_theta)])
    _, _, v_group_calc = solve_christoffel(A, n_opt, mode_idx)
    
    return v_group_calc