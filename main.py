import json
import os
import numpy as np
import matplotlib.pyplot as plt

from vti_utils import thomsen_to_voigt
from layers import HomogeneousAnisotropicLayer, LayeredVelocityModel
from bending_solver import TwoPointBendingTracer


def create_sample_json(filename="model_config.json"):
    """
    Создает базовую двухслойную VTI-модель.
    """
    data = {
        "layers": [
            {
                "type": "homogeneous",
                "name": "Upper Layer (Thomsen)",
                "z_top": 0.0,
                "z_bot": 1000.0,
                "rho": 2400.0,
                "thomsen_parameters": {
                    "vp0": 2600.0,
                    "vs0": 1300.0,
                    "epsilon": 0.20,
                    "delta": 0.10,
                    "gamma": 0.15
                }
            },
            {
                "type": "homogeneous",
                "name": "Lower Layer (Voigt GPa)",
                "z_top": 1000.0,
                "z_bot": 2500.0,
                "rho": 2600.0,
                "C_voigt_GPa": [
                    [24.0,  8.0,  6.5,  0.0,  0.0,  0.0],
                    [ 8.0, 24.0,  6.5,  0.0,  0.0,  0.0],
                    [ 6.5,  6.5, 15.0,  0.0,  0.0,  0.0],
                    [ 0.0,  0.0,  0.0,  5.0,  0.0,  0.0],
                    [ 0.0,  0.0,  0.0,  0.0,  5.0,  0.0],
                    [ 0.0,  0.0,  0.0,  0.0,  0.0,  8.0]
                ]
            }
        ]
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_model_from_json(json_filepath: str) -> LayeredVelocityModel:
    """
    Загружает конфигурацию слоев из JSON-файла.
    """
    model = LayeredVelocityModel()
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for layer_data in data.get("layers", []):
        name = layer_data.get("name")
        z_top = layer_data.get("z_top")
        z_bot = layer_data.get("z_bot")
        rho = layer_data.get("rho")

        if "thomsen_parameters" in layer_data:
            tp = layer_data["thomsen_parameters"]
            C_voigt = thomsen_to_voigt(tp["vp0"], tp["vs0"], tp["epsilon"], tp["delta"], tp["gamma"], rho)
        else:
            C_voigt = np.array(layer_data.get("C_voigt_GPa", np.zeros((6, 6)))) * 1e9

        model.add_layer(HomogeneousAnisotropicLayer(name, z_top, z_bot, C_voigt, rho))

    return model


def main():
    json_filename = "model_config.json"
    if not os.path.exists(json_filename):
        create_sample_json(json_filename)

    model = load_model_from_json(json_filename)

    # Параметры задачи
    SOURCE = np.array([1000.0, 0.0, 1000.0])
    WAVE_TYPE = 'qP'                      # Тип волны ('qP', 'qS1', 'qS2')
    mode_index = {'qS1': 0, 'qS2': 1, 'qP': 2}[WAVE_TYPE]

    # Линия приемников на поверхности
    receiver_x = np.linspace(200.0, 3000.0, 15)
    receivers = [np.array([x, x * 0.2, 0.0]) for x in receiver_x]

    tracer = TwoPointBendingTracer(model, mode_idx=mode_index)

    plt.figure(figsize=(10, 7))
    print(f"=== Расчет двухточечного лучевого трассирования ({WAVE_TYPE}) ===")
    print(f"Источник S: {SOURCE}")
    print("-" * 65)

    for i, rec in enumerate(receivers):
        result = tracer.solve(SOURCE, rec)
        path = result['path']
        t_total = result['travel_time']

        print(f"Приемник R{i+1:02d} ({rec[0]:7.1f}, {rec[1]:6.1f}, {rec[2]:3.1f} м) | "
              f"Время пробега t = {t_total:.4f} с | Узлов = {len(path)}")

        # Отображение профиля X-Z
        plt.plot(path[:, 0], path[:, 2], 'o-', label=f'R{i+1}' if i % 3 == 0 else "")

    # Отображение границ слоев
    for layer in model.layers:
        plt.axhline(y=layer.z_top, color='gray', linestyle='--', alpha=0.6)
        plt.axhline(y=layer.z_bot, color='gray', linestyle='--', alpha=0.6)

    # Точки источника и приемников
    plt.plot(SOURCE[0], SOURCE[2], 'r*', markersize=15, label='Источник (S)')
    rec_arr = np.array(receivers)
    plt.plot(rec_arr[:, 0], rec_arr[:, 2], 'v', color='black', markersize=8, label='Приемники (R)')

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.xlabel('Координата X (м)')
    plt.ylabel('Глубина Z (м)')
    plt.title(f'Двухточечное лучевое трассирование методом изгиба ({WAVE_TYPE}-волна)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    output_file = "two_point_ray_tracing.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("-" * 65)
    print(f"[УСПЕХ] Результаты расчетов сохранены в файл: {output_file}")
    
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()