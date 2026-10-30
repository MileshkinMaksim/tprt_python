import numpy as np
import scipy.optimize as opt
from layers import LayeredVelocityModel, HomogeneousAnisotropicLayer
from kinematics_3d import get_group_velocity_for_direction, solve_christoffel


class TwoPointBendingTracer:
    """
    Решатель двухточечной задачи лучевого трассирования методом изгиба (Bending Method)
    на основе принципа Ферма для многослойных VTI-сред.
    """
    def __init__(self, model: LayeredVelocityModel, mode_idx: int):
        self.model = model
        self.mode_idx = mode_idx

    def _get_crossed_layers_and_interfaces(self, z_src: float, z_rec: float):
        """
        Определяет последовательность слоев и промежуточных границ,
        пересекаемых лучом от источника к приемнику.
        """
        layers = []
        interfaces_z = []
        
        # Направление движения по Z
        going_down = z_src < z_rec
        
        if going_down:
            sorted_layers = sorted(self.model.layers, key=lambda l: l.z_top)
            for layer in sorted_layers:
                if layer.z_bot > z_src and layer.z_top < z_rec:
                    layers.append(layer)
            for i in range(len(layers) - 1):
                interfaces_z.append(layers[i].z_bot)
        else:
            sorted_layers = sorted(self.model.layers, key=lambda l: l.z_top, reverse=True)
            for layer in sorted_layers:
                if layer.z_top < z_src and layer.z_bot > z_rec:
                    layers.append(layer)
            for i in range(len(layers) - 1):
                interfaces_z.append(layers[i].z_top)
                
        return layers, interfaces_z

    def _segment_travel_time(self, p1: np.ndarray, p2: np.ndarray, layer: HomogeneousAnisotropicLayer) -> float:
        """
        Вычисляет время пробега по прямолинейному сегменту [p1, p2] в однородном VTI-слое.
        """
        dir_vec = p2 - p1
        dist = np.linalg.norm(dir_vec)
        if dist == 0:
            return 0.0
            
        v_group = get_group_velocity_for_direction(dir_vec, layer, self.mode_idx)
        v_mag = np.linalg.norm(v_group)
        if v_mag == 0:
            return 1e9
        return dist / v_mag

    def solve(self, source: np.ndarray, receiver: np.ndarray) -> dict:
        """
        Находит оптимальные координаты узлов на границах слоев и строит финальный луч.
        
        Parameters:
            source: Координаты источника [Xs, Ys, Zs].
            receiver: Координаты приемника [Xr, Yr, Zr].
            
        Returns:
            dict с ключами: 'path', 'travel_time', 'segments', 'slowness_vectors'
        """
        source = np.array(source, dtype=float)
        receiver = np.array(receiver, dtype=float)
        
        crossed_layers, interface_zs = self._get_crossed_layers_and_interfaces(source[2], receiver[2])
        num_interfaces = len(interface_zs)

        # Если источник и приемник в одном слое — прямая линия
        if num_interfaces == 0:
            layer = crossed_layers[0] if crossed_layers else self.model.get_layer_at_depth(source[2])
            t_total = self._segment_travel_time(source, receiver, layer)
            return {
                'path': np.array([source, receiver]),
                'travel_time': t_total,
                'layers': [layer]
            }

        # Начальное приближение — прямая линия S -> R
        init_params = []
        for z_int in interface_zs:
            frac = (z_int - source[2]) / (receiver[2] - source[2])
            x_int = source[0] + frac * (receiver[0] - source[0])
            y_int = source[1] + frac * (receiver[1] - source[1])
            init_params.extend([x_int, y_int])
        
        init_params = np.array(init_params)

        # Целевая функция минимизации суммарного времени
        def objective(params):
            nodes = [source]
            for i, z_int in enumerate(interface_zs):
                nodes.append(np.array([params[2 * i], params[2 * i + 1], z_int]))
            nodes.append(receiver)
            
            t_sum = 0.0
            for i in range(len(crossed_layers)):
                t_sum += self._segment_travel_time(nodes[i], nodes[i + 1], crossed_layers[i])
            return t_sum

        # Оптимизация методом L-BFGS-B
        res = opt.minimize(objective, init_params, method='L-BFGS-B', options={'gtol': 1e-6, 'ftol': 1e-9})

        # Формирование финальной траектории
        opt_params = res.x
        path = [source]
        for i, z_int in enumerate(interface_zs):
            path.append(np.array([opt_params[2 * i], opt_params[2 * i + 1], z_int]))
        path.append(receiver)
        
        path_arr = np.array(path)
        
        # Расчет сегментов и векторов медленности
        segments = []
        slowness_vectors = []
        for i in range(len(crossed_layers)):
            p_start, p_end = path_arr[i], path_arr[i + 1]
            seg_dir = p_end - p_start
            v_g = get_group_velocity_for_direction(seg_dir, crossed_layers[i], self.mode_idx)
            
            # Нахождение вектора фазовой нормали n
            phi = np.arctan2(seg_dir[1], seg_dir[0])
            theta = np.arccos(np.clip(seg_dir[2] / np.linalg.norm(seg_dir), -1.0, 1.0))
            n_vec = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
            
            V_phase, _, _ = solve_christoffel(crossed_layers[i].get_tensors(), n_vec, self.mode_idx)
            p_vec = n_vec / V_phase
            
            slowness_vectors.append(p_vec)
            segments.append({
                'start': p_start,
                'end': p_end,
                'v_group': v_g,
                'travel_time': np.linalg.norm(seg_dir) / np.linalg.norm(v_g),
                'layer': crossed_layers[i]
            })

        return {
            'path': path_arr,
            'travel_time': res.fun,
            'segments': segments,
            'slowness_vectors': slowness_vectors
        }