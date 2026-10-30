import os
import json
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# =============================================================================
# 1. МАТЕМАТИЧЕСКИЙ БЛОК: VTI-СРЕДА И ФАЗОВЫЕ / ГРУППОВЫЕ СКОРОСТИ
# =============================================================================

def thomsen_to_voigt(vp0, vs0, epsilon, delta, gamma, rho):
    """Преобразование параметров Томпсена в тензор жесткости Фойгта C_ij (Па)."""
    c33 = rho * (vp0 ** 2)
    c44 = rho * (vs0 ** 2)
    c11 = c33 * (1.0 + 2.0 * epsilon)
    c66 = c44 * (1.0 + 2.0 * gamma)
    
    val = 2.0 * delta * c33 * (c33 - c44) + (c33 - c44)**2
    c13 = np.sqrt(val) - c44 if val >= 0 else 0.0
    c12 = c11 - 2.0 * c66
    
    C = np.zeros((6, 6))
    C[0, 0] = c11; C[1, 1] = c11; C[2, 2] = c33
    C[3, 3] = c44; C[4, 4] = c44; C[5, 5] = c66
    C[0, 1] = c12; C[1, 0] = c12
    C[0, 2] = c13; C[2, 0] = c13
    C[1, 2] = c13; C[2, 1] = c13
    return C

def voigt_to_tensor(C_voigt, rho):
    """Преобразование матрицы C_ij в 4-ранговый тензор A_ijkl = C_ijkl / rho."""
    A = np.zeros((3, 3, 3, 3))
    voigt_map = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 1): 3,
                 (0, 2): 4, (2, 0): 4, (0, 1): 5, (1, 0): 5}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    A[i, j, k, l] = C_voigt[voigt_map[(i, j)], voigt_map[(k, l)]] / rho
    return A

def compute_group_velocity(A_tensor, np_phase, mode_idx):
    """
    Вычисление групповой скорости v_group по фазовой нормали np_phase.
    mode_idx: 0 -> qS1, 1 -> qS2, 2 -> qP (по возрастанию фазовых скоростей).
    """
    n = np_phase / np.linalg.norm(np_phase)
    Gamma = np.einsum('ijkl,i,k->jl', A_tensor, n, n)
    eigvals, eigvecs = np.linalg.eigh(Gamma)
    
    sorted_idx = np.argsort(eigvals)
    selected_mode = sorted_idx[mode_idx]
    
    V_phase = np.sqrt(eigvals[selected_mode])
    w = eigvecs[:, selected_mode]
    
    v_group = np.einsum('rjkl,k,l,j->r', A_tensor, n, w, w) / V_phase
    return v_group, V_phase

def calculate_ray_travel_time(A_tensor, src_pos, rec_pos, mode_idx):
    """Определяет точное время пробега луча от источника до приёмника в VTI-среде."""
    ray_dir = rec_pos - src_pos
    distance = np.linalg.norm(ray_dir)
    if distance == 0:
        return 0.0
    ray_dir_norm = ray_dir / distance
    
    theta_geom = np.arctan2(ray_dir_norm[0], ray_dir_norm[2])

    def objective(theta):
        n_p = np.array([np.sin(theta[0]), 0.0, np.cos(theta[0])])
        v_g, _ = compute_group_velocity(A_tensor, n_p, mode_idx)
        v_g_norm = v_g / np.linalg.norm(v_g)
        return 1.0 - np.dot(v_g_norm, ray_dir_norm)

    res = opt.minimize(objective, [theta_geom], method='Nelder-Mead', tol=1e-12)
    best_theta = res.x[0]
    
    n_p_opt = np.array([np.sin(best_theta), 0.0, np.cos(best_theta)])
    v_g_opt, _ = compute_group_velocity(A_tensor, n_p_opt, mode_idx)
    
    v_g_mag = np.linalg.norm(v_g_opt)
    return distance / v_g_mag


# =============================================================================
# 2. ОСНОВНОЙ КЛАСС АВТОТЕСТЕРА
# =============================================================================

class SeismicRayTracingAutoTester:
    def __init__(self, base_dir="Autotest_Data"):
        self.base_dir = base_dir
        self.source_pos = np.array([1000.0, 0.0, 1000.0])
        self.wave_modes = {'qP': 2, 'qS1': 0, 'qS2': 1}
        self.wave_keys_true = {'qP': 'P', 'qS1': 'S1', 'qS2': 'S2'}

    def load_json(self, filename):
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run_all_tests(self, num_tests=5):
        print("=" * 80)
        print(" ЗАПУСК АВТОМАТИЧЕСКОГО ТЕСТИРОВАНИЯ СИМУЛЯЦИИ ЛУЧЕВОГО ТРАССИРОВАНИЯ")
        print(f" Директория с данными: {os.path.abspath(self.base_dir)}")
        print(f" Координаты источника: X={self.source_pos[0]}, Y={self.source_pos[1]}, Z={self.source_pos[2]} м")
        print("=" * 80)

        receivers_data = self.load_json("receivers.json")
        receivers_coords = [
            np.array([r["location"]["x"], r["location"]["y"], r["location"]["z"]])
            if "location" in r else
            np.array([r["coordinates"]["x"], r["coordinates"]["y"], r["coordinates"]["z"]])
            for r in receivers_data
        ]

        summary_results = {}

        for test_idx in range(1, num_tests + 1):
            config_file = f"model_config_test_{test_idx}.json"
            times_file = f"Test_times_{test_idx}.json"

            config_path = os.path.join(self.base_dir, config_file)
            times_path = os.path.join(self.base_dir, times_file)

            if not (os.path.exists(config_path) and os.path.exists(times_path)):
                print(f"\n[ПРОПУСК] Тест {test_idx}: отсутствуют файлы {config_file} или {times_file}")
                continue

            print(f"\n---> Прогон Теста #{test_idx} ({config_file})")
            
            config_data = self.load_json(config_file)
            layer_params = config_data["layers"][0]["thomsen_parameters"]
            rho = config_data["layers"][0].get("rho", 1000.0)

            C_voigt = thomsen_to_voigt(
                layer_params["vp0"], layer_params["vs0"],
                layer_params["epsilon"], layer_params["delta"],
                layer_params["gamma"], rho
            )
            A_tensor = voigt_to_tensor(C_voigt, rho)

            true_times_data = self.load_json(times_file)
            test_metrics = {}

            for mode_name, mode_idx in self.wave_modes.items():
                true_key = self.wave_keys_true[mode_name]
                calc_times = []
                true_times = []

                for i, rec_pos in enumerate(receivers_coords):
                    t_calc = calculate_ray_travel_time(A_tensor, self.source_pos, rec_pos, mode_idx)
                    t_true = true_times_data[i]["times"][true_key]
                    
                    calc_times.append(t_calc)
                    true_times.append(t_true)

                calc_times = np.array(calc_times)
                true_times = np.array(true_times)

                abs_errors = np.abs(calc_times - true_times)
                rel_errors = (abs_errors / true_times) * 100.0

                mae = np.mean(abs_errors)
                rmse = np.sqrt(np.mean(abs_errors ** 2))
                max_ae = np.max(abs_errors)
                mre = np.mean(rel_errors)
                max_re = np.max(rel_errors)

                test_metrics[mode_name] = {
                    "MAE_sec": mae,
                    "RMSE_sec": rmse,
                    "MaxAE_sec": max_ae,
                    "MRE_pct": mre,
                    "MaxRE_pct": max_re,
                    "calc_times": calc_times.tolist(),
                    "true_times": true_times.tolist(),
                    "abs_errors": abs_errors.tolist()
                }

                print(f"  Волна {mode_name:4s} | MAE: {mae*1e6:8.3f} мкс | RMSE: {rmse*1e6:8.3f} мкс | MaxAE: {max_ae*1e6:8.3f} мкс | MRE: {mre:6.4f}%")

            summary_results[f"Test_{test_idx}"] = test_metrics

        self._save_report(summary_results)
        self._plot_results(summary_results, receivers_coords)
        self._print_executive_summary(summary_results)

    def _save_report(self, summary_results):
        report_path = os.path.join(self.base_dir, "autotest_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, ensure_ascii=False, indent=4)
        print(f"\n[INFO] Детальный отчет сохранен в: {report_path}")

    def _print_executive_summary(self, summary_results):
        print("\n" + "=" * 80)
        print(" ИТОГОВАЯ СВОДКА ТОЧНОСТИ АВТОТЕСТОВ")
        print("=" * 80)
        header = f"{'Тест':<8} | {'Волна':<6} | {'MAE (с)':<12} | {'RMSE (с)':<12} | {'MaxAE (с)':<12} | {'MRE (%)':<10}"
        print(header)
        print("-" * len(header))

        for test_name, modes in summary_results.items():
            for mode_name, m in modes.items():
                print(f"{test_name:<8} | {mode_name:<6} | {m['MAE_sec']:<12.6e} | {m['RMSE_sec']:<12.6e} | {m['MaxAE_sec']:<12.6e} | {m['MRE_pct']:<10.4f}")

    def _plot_results(self, summary_results, receivers_coords):
        if not summary_results:
            return
            
        x_rec = [r[0] for r in receivers_coords]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Результаты автотестирования трассирования лучей (Годографы и Ошибки)", fontsize=14)

        colors = {'qP': 'crimson', 'qS1': 'blue', 'qS2': 'forestgreen'}

        if "Test_1" in summary_results:
            ax1 = axes[0, 0]
            for mode, data in summary_results["Test_1"].items():
                ax1.plot(x_rec, data["true_times"], '--', label=f"{mode} (Истинное)", color=colors[mode], alpha=0.7)
                ax1.plot(x_rec, data["calc_times"], ':', label=f"{mode} (Расчетное)", color=colors[mode])
            ax1.set_title("Годографы времен пробега (Тест 1)")
            ax1.set_xlabel("X координата приёмника (м)")
            ax1.set_ylabel("Время (с)")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.5)

            ax2 = axes[0, 1]
            for mode, data in summary_results["Test_1"].items():
                ax2.plot(x_rec, np.array(data["abs_errors"]) * 1e6, label=f"{mode}", color=colors[mode])
            ax2.set_title("Абсолютная ошибка времени пробега (Тест 1)")
            ax2.set_xlabel("X координата приёмника (м)")
            ax2.set_ylabel("Ошибка (микросекунды)")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)

        ax3 = axes[1, 0]
        tests = list(summary_results.keys())
        x_indices = np.arange(len(tests))
        width = 0.25

        for i, mode in enumerate(['qP', 'qS1', 'qS2']):
            maes = [summary_results[t][mode]["MAE_sec"] * 1e6 for t in tests if mode in summary_results[t]]
            ax3.bar(x_indices + i * width, maes, width, label=f"Волна {mode}", color=colors[mode])

        ax3.set_title("Средняя абсолютная ошибка (MAE) по тестам")
        ax3.set_xticks(x_indices + width)
        ax3.set_xticklabels(tests)
        ax3.set_ylabel("MAE (микросекунды)")
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.5)

        ax4 = axes[1, 1]
        for i, mode in enumerate(['qP', 'qS1', 'qS2']):
            mres = [summary_results[t][mode]["MRE_pct"] for t in tests if mode in summary_results[t]]
            ax4.bar(x_indices + i * width, mres, width, label=f"Волна {mode}", color=colors[mode])

        ax4.set_title("Средняя относительная ошибка (MRE %) по тестам")
        ax4.set_xticks(x_indices + width)
        ax4.set_xticklabels(tests)
        ax4.set_ylabel("Относительная ошибка (%)")
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plot_path = os.path.join(self.base_dir, "autotest_results.png")
        plt.savefig(plot_path, dpi=300)
        print(f"[INFO] График сравнения результатов сохранен в: {plot_path}")

# =============================================================================
# 3. ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "Autotest_Data")

    tester = SeismicRayTracingAutoTester(base_dir=data_dir)
    tester.run_all_tests(num_tests=5)