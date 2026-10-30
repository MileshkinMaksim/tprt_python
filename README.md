# Seismic Ray Tracing

High-performance numerical engine for 2D/3D seismic ray tracing and traveltime calculations in heterogeneous velocity models.

## Overview

`seismic-ray-tracing` is a computational framework designed to simulate seismic wave propagation using ray theory approximations. It provides algorithms for solving the Eikonal equation and kinematic ray systems, supporting both continuous velocity continua and layered geological models.

### Key Capabilities
* **Shooting Solver**: Solves initial-value problems (IVP) using $4^{\text{th}}$-order Runge-Kutta (RK4) ray integration.
* **Bending Solver**: Solves two-point boundary value problems (TPBVP) using Fermat's principle optimization.
* **Eikonal FMM**: Fast Marching Method for first-arrival traveltime field computation on regular spatial grids.
* **Multi-threading**: OpenMP acceleration for multi-source/multi-receiver geometry configurations.

## Mathematical Formulation

Seismic ray trajectories are calculated based on the high-frequency approximation of the acoustic wave equation, where traveltimes $T(\mathbf{x})$ satisfy the Eikonal equation:

$$ |\nabla T(\mathbf{x})|^2 = \frac{1}{v(\mathbf{x})^2} = s(\mathbf{x})^2 $$

where $v(\mathbf{x})$ represents the local P-wave or S-wave velocity field and $s(\mathbf{x})$ represents slowness.

The canonical ray system in Hamiltonian formulation is integrated as:

$$ \frac{d\mathbf{x}}{d\tau} = \mathbf{p}, \quad \frac{d\mathbf{p}}{d\tau} = \frac{1}{2} \nabla \left( \frac{1}{v^2(\mathbf{x})} \right) $$

where $\mathbf{x} = (x, y, z)^T$ is the spatial coordinate vector, $\mathbf{p} = \nabla T$ is the slowness vector, and $\tau$ is the ray parameter.

## Prerequisites

* **Compiler**: C++17 compliant (`g++` $\ge 9.0$, `clang++` $\ge 10.0$, or MSVC 2019+)
* **Build System**: CMake $\ge 3.18$
* **Dependencies**: OpenMP (optional, for parallel execution)

## Building from Source

```bash
git clone https://github.com/username/seismic-ray-tracing.git
cd seismic-ray-tracing
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

## Quick Start

### C++ Interface

```cpp
#include "raytracer/velocity_model.hpp"
#include "raytracer/shooting.hpp"

int main() {
    // Initialize continuous velocity grid
    VelocityModel2D model("velocity_grid.bin");

    // Source and receiver setup
    Point2D source{0.0, 0.0};
    Point2D receiver{2000.0, 1000.0};

    // Configuration parameters
    RayConfig config;
    config.step_size = 1.0;
    config.tolerance = 1e-5;

    // Perform two-point ray tracing
    RayPath ray = ShootingSolver::find_ray(model, source, receiver, config);

    ray.export_csv("ray_output.csv");
    return 0;
}
```

### Command Line Interface

Execute ray tracing via CLI configuration:

```bash
./bin/seismic_raytracer --config config.json --output results.dat
```

#### Example `config.json`

```json
{
  "model": {
    "grid_file": "velocity_grid.raw",
    "nx": 500,
    "nz": 300,
    "dx": 10.0,
    "dz": 10.0
  },
  "source": [0.0, 0.0],
  "receivers": {
    "x_min": 500.0,
    "x_max": 2500.0,
    "count": 51,
    "depth": 0.0
  },
  "solver": {
    "method": "shooting",
    "rk_order": 4,
    "max_steps": 10000
  }
}
```

## Data Specifications

Output trajectory files include the following parameters along each discretized ray segment:

| Parameter | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `x, y, z` | `double` | $m$ | Spatial position coordinates |
| `t` | `double` | $s$ | Cumulative traveltime from source |
| `px, py, pz` | `double` | $s/m$ | Slowness vector components |
| `v` | `double` | $m/s$ | Local wave velocity |
| `s_arc` | `double` | $m$ | Integrated arc length |

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.