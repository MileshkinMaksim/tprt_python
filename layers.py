import numpy as np
from vti_utils import voigt_to_tensor4d

class HomogeneousAnisotropicLayer:
    """
    Однородный анизотропный VTI-слой.
    """
    def __init__(self, name: str, z_top: float, z_bot: float, C_voigt: np.ndarray, rho: float):
        self.name = name
        self.z_top = z_top
        self.z_bot = z_bot
        self.rho = rho
        self.C_voigt = C_voigt
        self.A_tensor = voigt_to_tensor4d(C_voigt, rho)

    def is_inside(self, z: float) -> bool:
        return self.z_top <= z < self.z_bot

    def get_tensors(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
        return self.A_tensor


class LayeredVelocityModel:
    """
    Модель среды, состоящая из набора горизонтальных однородных слоев.
    """
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: HomogeneousAnisotropicLayer):
        self.layers.append(layer)

    def get_layer_at_depth(self, z: float) -> HomogeneousAnisotropicLayer:
        for layer in self.layers:
            if layer.is_inside(z):
                return layer
        # Если глубина на нижней границе последнего слоя
        if self.layers and np.isclose(z, self.layers[-1].z_bot):
            return self.layers[-1]
        raise ValueError(f"Глубина Z={z} выходит за пределы модели.")