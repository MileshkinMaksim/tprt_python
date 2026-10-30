# autotester.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from layers import LayeredVelocityModel, HomogeneousAnisotropicLayer
from bending_solver import TwoPointBendingTracer
from vti_utils import thomsen_to_voigt

class SeismicRayTracingAutoTester:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.source_pos = np.array([1000.0, 0.0, 1000.0])
        # Mapping wave modes to bending tracer mode indices
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
                print(f"\n[ПРОПУСК] Тест #{test_idx}: отсутствуют файлы {config_file} или {times_file}")
                continue

            print(f"\n---> Прогон Теста #{test_idx} ({config_file})")
            
            config_data = self.load_json(config_file)
            model = LayeredVelocityModel()

            # Сборка модели среды через vti_utils и layers
            for l_idx, l_cfg in enumerate(config_data["layers"]):
                p = l_cfg["thomsen_parameters"]
                rho = l_cfg.get("rho", 1000.0)
                
                C_voigt = thomsen_to_voigt(
                    vp0=p["vp0"], 
                    vs0=p["vs0"], 
                    epsilon=p["epsilon"], 
                    delta=p["delta"], 
                    gamma=p["gamma"], 
                    rho=rho
                )
                
                layer = HomogeneousAnisotropicLayer(
                    name=l_cfg.get("name", f"Layer_{l_idx + 1}"),
                    z_top=l_cfg["z_top"],
                    z_bot=l_cfg["z_bot"],
                    C_voigt=C_voigt,
                    rho=rho
                )
                model.add_layer(layer)

            true_times_data = self.load_json(times_file)
            test_metrics = {}


            for mode_name in ['qP', 'qS1', 'qS2']:
                mode_idx = self.wave_modes[mode_name]
                tracer = TwoPointBendingTracer(model=model, mode_idx=mode_idx)
                
                calc_times = []
                true_times = []
                
                for i, rec_pos in enumerate(receivers_coords):
                    res = tracer.solve(self.source_pos, rec_pos)
                    t_calc = res['travel_time']
                    t_true = true_times_data[i]["times"][self.wave_keys_true[mode_name]]
                    
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
        print(f"\n[INFO] Отчет сохранен в: {report_path}")

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
        fig.suptitle("Результаты автотестирования трассирования лучей", fontsize=14)

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
        print(f"[INFO] График сохранен в: {plot_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "Autotest_Data")

    tester = SeismicRayTracingAutoTester(base_dir=data_dir)
    tester.run_all_tests(num_tests=5)