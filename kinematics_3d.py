import numpy as np
import scipy.optimize as opt
from layers import HomogeneousAnisotropicLayer

def solve_christoffel(A_tensor: np.ndarray, n_vec: np.ndarray, mode_idx):
    """
    Решает уравнение Кристоффеля с точной идентификацией мод по поляризации:
    - mode_idx = 2 ('qP'): квазипродольная волна (поляризация вдоль n_vec)
    - mode_idx = 1 ('qSV' / 'S1'): поляризация в вертикальной плоскости падения
    - mode_idx = 0 ('SH' / 'S2'): поляризация строго горизонтальная (перпендикулярна плоскости падения)
    """
    n = n_vec / np.linalg.norm(n_vec)
    
    # Решение тензорного уравнения Кристоффеля
    Gamma = np.einsum('ijkl,i,k->jl', A_tensor, n, n)
    eigvals, eigvecs = np.linalg.eigh(Gamma)
    
    # 1. Находим qP-волну (вектор поляризации максимально параллелен фазовой нормали n)
    p_projections = np.abs(np.dot(n, eigvecs))  # |n · w_i|
    p_idx = int(np.argmax(p_projections))

    # 2. Находим SH-волну (поляризация перпендикулярна плоскости падения: w = [-sin(phi), cos(phi), 0])
    phi = np.arctan2(n[1], n[0])
    e_sh = np.array([-np.sin(phi), np.cos(phi), 0.0])
    sh_projections = np.abs(np.dot(e_sh, eigvecs))
    
    # Если нормировка дает околонулевой вектор в зените (n = [0, 0, 1]), берем проекцию на Y
    if np.linalg.norm(e_sh) < 1e-6:
        sh_projections = np.abs(eigvecs[1, :])
        
    sh_idx = int(np.argmax(sh_projections))

    # 3. qSV-волна — оставшийся индекс в плоскости падения
    sv_idx = [i for i in range(3) if i not in (p_idx, sh_idx)]
    sv_idx = sv_idx[0] if len(sv_idx) > 0 else (0 if p_idx != 0 else 1)

    # Отображение запрошенного mode_idx
    if isinstance(mode_idx, str):
        mode_map = {'qP': p_idx, 'P': p_idx, 'qSV': sv_idx, 'S1': sv_idx, 'SH': sh_idx, 'qS1': sv_idx, 'qS2': sh_idx, 'S2': sh_idx}
        sel_idx = mode_map.get(mode_idx, p_idx)
    else:
        # Для числовых индексов: 2 -> qP, 0 -> qSV (S1), 1 -> SH (S2)
        idx_map = {2: p_idx, 0: sv_idx, 1: sh_idx}
        sel_idx = idx_map.get(int(mode_idx), p_idx)

    V2 = max(eigvals[sel_idx], 1e-12)
    V = np.sqrt(V2)
    w = eigvecs[:, sel_idx]

    # Групповая скорость (скорость энергии)
    v_group = np.einsum('rjkl,k,l,j->r', A_tensor, n, w, w) / V
    return V, w, v_group


def get_group_velocity_for_direction(dir_geom: np.ndarray, layer: HomogeneousAnisotropicLayer, mode_idx: int) -> np.ndarray:
    """
    Находит вектор групповой скорости v_group, соответствующий заданному
    направлению Луча dir_geom в VTI-среде.
    """
    norm_dir = np.linalg.norm(dir_geom)
    if norm_dir == 0:
        return np.zeros(3)
    d_hat = dir_geom / norm_dir
    phi = np.arctan2(d_hat[1], d_hat[0])
    A = layer.get_tensors()

    def objective(th_val):
        th = float(np.atleast_1d(th_val)[0])
        n_guess = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
        _, _, v_calc = solve_christoffel(A, n_guess, mode_idx)
        v_norm = np.linalg.norm(v_calc)
        if v_norm == 0:
            return 1.0
        return 1.0 - np.dot(v_calc / v_norm, d_hat)

    # Одномерная оптимизация фазового угла theta на отрезке [0, pi]
    res = opt.minimize_scalar(objective, bounds=(0.0, np.pi), method='bounded')
    opt_theta = res.x

    n_opt = np.array([np.sin(opt_theta) * np.cos(phi), np.sin(opt_theta) * np.sin(phi), np.cos(opt_theta)])
    _, _, v_group_calc = solve_christoffel(A, n_opt, mode_idx)
    return v_group_calc