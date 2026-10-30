import numpy as np
import scipy.optimize as opt
from scipy.interpolate import CubicSpline
from layers import LayeredVelocityModel, HomogeneousAnisotropicLayer
from kinematics_3d import solve_christoffel


class FastVTILayerGroupSlowness:
    """
    Быстрый интерполятор вектора медленности S(psi) = 1 / V_group(psi)
    для VTI-слоя на основе кубического сплайна.
    """
    def __init__(self, layer: HomogeneousAnisotropicLayer, mode_idx: int, num_points: int = 361):
        self.layer = layer
        self.mode_idx = mode_idx
        A = layer.get_tensors()
        
        psi_grid = np.linspace(0.0, np.pi / 2.0, num_points)
        slowness_grid = []
        
        for psi_ray in psi_grid:
            def angle_diff(theta):
                n_vec = np.array([np.sin(theta), 0.0, np.cos(theta)])
                _, _, vg_2d = solve_christoffel(A, n_vec, mode_idx)
                psi_calc = np.arctan2(abs(vg_2d[0]), abs(vg_2d[2]))
                return psi_calc - psi_ray

            try:
                th_opt = opt.brentq(angle_diff, 0.0, np.pi / 2.0, xtol=1e-8)
            except Exception:
                res = opt.minimize_scalar(lambda th: abs(angle_diff(th)), bounds=(0.0, np.pi / 2.0), method='bounded')
                th_opt = float(res.x)

            n_opt = np.array([np.sin(th_opt), 0.0, np.cos(th_opt)])
            _, _, vg_2d = solve_christoffel(A, n_opt, mode_idx)
            v_mag = np.linalg.norm(vg_2d)
            slowness_grid.append(1.0 / v_mag if v_mag > 0 else 1e-9)

        self.spline = CubicSpline(psi_grid, slowness_grid)

    def get_segment_travel_time(self, p1: np.ndarray, p2: np.ndarray) -> float:
        dir_vec = p2 - p1
        dist = np.linalg.norm(dir_vec)
        if dist == 0:
            return 0.0
        
        dz = abs(dir_vec[2])
        psi = np.arccos(np.clip(dz / dist, 0.0, 1.0))
        return dist * float(self.spline(psi))


class TwoPointBendingTracer:
    """
    Решатель двухточечной задачи лучевого трассирования методом изгиба 
    для многослойных VTI-сред с 1D-параметризацией узлов.
    """
    def __init__(self, model: LayeredVelocityModel, mode_idx: int):
        self.model = model
        self.mode_idx = mode_idx
        self.fast_layers = [FastVTILayerGroupSlowness(l, mode_idx) for l in model.layers]

    def _get_crossed_layers_and_interfaces(self, z_src: float, z_rec: float):
        layers = []
        fast_layers = []
        interfaces_z = []
        going_down = z_src < z_rec
        
        if going_down:
            sorted_indices = sorted(range(len(self.model.layers)), key=lambda idx: self.model.layers[idx].z_top)
            for idx in sorted_indices:
                layer = self.model.layers[idx]
                if layer.z_bot > z_src and layer.z_top < z_rec:
                    layers.append(layer)
                    fast_layers.append(self.fast_layers[idx])
            for i in range(len(layers) - 1):
                interfaces_z.append(layers[i].z_bot)
        else:
            sorted_indices = sorted(range(len(self.model.layers)), key=lambda idx: self.model.layers[idx].z_top, reverse=True)
            for idx in sorted_indices:
                layer = self.model.layers[idx]
                if layer.z_top < z_src and layer.z_bot > z_rec:
                    layers.append(layer)
                    fast_layers.append(self.fast_layers[idx])
            for i in range(len(layers) - 1):
                interfaces_z.append(layers[i].z_top)
                
        return layers, fast_layers, interfaces_z

    def solve(self, source: np.ndarray, receiver: np.ndarray) -> dict:
        source = np.array(source, dtype=float)
        receiver = np.array(receiver, dtype=float)
        
        crossed_layers, crossed_fast_layers, interface_zs = self._get_crossed_layers_and_interfaces(source[2], receiver[2])
        num_interfaces = len(interface_zs)

        if num_interfaces == 0:
            fast_layer = crossed_fast_layers[0] if crossed_fast_layers else self.fast_layers[0]
            t_total = fast_layer.get_segment_travel_time(source, receiver)
            return {
                'path': np.array([source, receiver]),
                'travel_time': t_total
            }

        dx = receiver[0] - source[0]
        dy = receiver[1] - source[1]
        
        # Начальное приближение
        init_alphas = np.array([(i + 1) / (num_interfaces + 1) for i in range(num_interfaces)], dtype=float)

        def objective(alphas):
            nodes = [source]
            for i, z_int in enumerate(interface_zs):
                a = alphas[i]
                nodes.append(np.array([source[0] + a * dx, source[1] + a * dy, z_int]))
            nodes.append(receiver)
            
            t_sum = 0.0
            for i in range(len(crossed_fast_layers)):
                t_sum += crossed_fast_layers[i].get_segment_travel_time(nodes[i], nodes[i + 1])
            return t_sum

        bounds = [(0.0, 1.0)] * num_interfaces
        res = opt.minimize(objective, init_alphas, method='L-BFGS-B', bounds=bounds,
                           options={'gtol': 1e-7, 'ftol': 1e-10, 'maxiter': 100})

        opt_alphas = res.x
        path = [source]
        for i, z_int in enumerate(interface_zs):
            a = opt_alphas[i]
            path.append(np.array([source[0] + a * dx, source[1] + a * dy, z_int]))
        path.append(receiver)

        return {
            'path': np.array(path),
            'travel_time': float(res.fun)
        }