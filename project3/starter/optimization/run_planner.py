#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from optimization.obstacles import default_obstacle_scene, cory105_obstacle_scene
from optimization.unicycle_planner import (
    UnicyclePlanner, PlannerParams,
    UnicycleTrackingPlanner, TrackingParams,
)
from optimization.plot_trajectory import plot_trajectory
import numpy as np

def save_trajectory(result, filename):
    full_filename = filename + ".npz"
    np.savez(full_filename, x=result.x, y=result.y, theta=result.theta, v=result.v, omega=result.omega, total_time=result.total_time, dt=result.dt)
    print(f"Trajectory saved to {full_filename}")

def main():
    parser = argparse.ArgumentParser(description="Unicycle trajectory planner")
    parser.add_argument("--N", type=int, default=100, help="Number of discretization intervals")
    parser.add_argument(
        "--mode", choices=["min_time", "tracking"], default="min_time",
        help="Planner mode: 'min_time' (minimize T) or 'tracking' (quadratic LQR cost)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="default",
        choices=["default", "cory105", "cory"],
        help="Obstacle scene to use: 'default' or 'cory105' (alias: 'cory')",
    )
    parser.add_argument("--obstacle_buffer", type=float, default=0.1, help="Obstacle buffer")
    parser.add_argument("--v_min", type=float, default=None, help="Minimum linear velocity bound")
    parser.add_argument("--v_max", type=float, default=None, help="Maximum linear velocity bound")
    parser.add_argument("--omega_min", type=float, default=None, help="Minimum angular velocity bound")
    parser.add_argument("--omega_max", type=float, default=None, help="Maximum angular velocity bound")
    parser.add_argument("--max_lin_accel", type=float, default=None, help="Maximum linear acceleration |dv/dt|")
    parser.add_argument("--max_ang_accel", type=float, default=None, help="Maximum angular acceleration |domega/dt|")
    parser.add_argument("--dt", type=float, default=None, help="Tracking mode only: fixed timestep")
    parser.add_argument("--dt_min", type=float, default=None, help="Min-time mode only: minimum timestep")
    parser.add_argument("--dt_max", type=float, default=None, help="Min-time mode only: maximum timestep")
    args = parser.parse_args()

    scene_name = "cory105" if args.scene in ("cory105", "cory") else args.scene
    hardware_profile_active = scene_name == "cory105" and args.mode == "min_time"

    if scene_name == "default":
        start = (0.0, 0.0, 0.0)
        goal = (5.0, 5.0, math.pi / 2)
        obstacles = default_obstacle_scene()
    elif scene_name == "cory105":
        start = (0.0, 0.0, 0.0)
        goal = (2.5781, 0.0, 0.0)
        obstacles = cory105_obstacle_scene()

    if args.mode == "tracking":
        params = TrackingParams(N=args.N, obstacle_buffer=args.obstacle_buffer)
        if args.v_min is not None:
            params.v_min = args.v_min
        if args.v_max is not None:
            params.v_max = args.v_max
        if args.omega_min is not None:
            params.omega_min = args.omega_min
        if args.omega_max is not None:
            params.omega_max = args.omega_max
        if args.max_lin_accel is not None:
            params.max_lin_accel = args.max_lin_accel
        if args.max_ang_accel is not None:
            params.max_ang_accel = args.max_ang_accel
        if args.dt is not None:
            params.dt = args.dt
        planner = UnicycleTrackingPlanner(params)
        buf = params.obstacle_buffer
    else:
        params = PlannerParams(N=args.N, obstacle_buffer=args.obstacle_buffer)
        if args.v_min is not None:
            params.v_min = args.v_min
        if args.v_max is not None:
            params.v_max = args.v_max
        if args.omega_min is not None:
            params.omega_min = args.omega_min
        if args.omega_max is not None:
            params.omega_max = args.omega_max
        if args.max_lin_accel is not None:
            params.max_lin_accel = args.max_lin_accel
        if args.max_ang_accel is not None:
            params.max_ang_accel = args.max_ang_accel
        if args.dt_min is not None:
            params.dt_min = args.dt_min
        if args.dt_max is not None:
            params.dt_max = args.dt_max

        # Hardware profile for TurtleBot runs in Cory 105.
        if hardware_profile_active:
            params.N = max(params.N, 220)
            params.obstacle_buffer = max(params.obstacle_buffer, 0.3)
            params.v_min = max(params.v_min, 0.0)
            params.v_max = min(params.v_max, 0.22)
            params.max_lin_accel = min(params.max_lin_accel, 0.18)
            params.max_ang_accel = min(params.max_ang_accel, 1.2)

            params.dt_min = max(params.dt_min, 0.04)
            params.dt_max = min(params.dt_max, 0.2)
            if params.dt_max < params.dt_min:
                params.dt_max = params.dt_min

            omega_cap = min(0.45, 2.0 * max(params.v_max, 0.05))
            params.omega_max = min(params.omega_max, omega_cap)
            params.omega_min = max(params.omega_min, -omega_cap)

        planner = UnicyclePlanner(params)
        buf = params.obstacle_buffer

    if args.mode == "tracking":
        print(
            f"Solving with N={params.N}, mode={args.mode}, dt={params.dt}, "
            f"v:[{params.v_min}, {params.v_max}], omega:[{params.omega_min}, {params.omega_max}], "
            f"accel:[{params.max_lin_accel}, {params.max_ang_accel}] ..."
        )
    else:
        if hardware_profile_active:
            print(
                "Applied cory105 hardware profile: "
                f"N={params.N}, buffer={params.obstacle_buffer}, v:[{params.v_min}, {params.v_max}], "
                f"omega:[{params.omega_min}, {params.omega_max}], "
                f"accel:[{params.max_lin_accel}, {params.max_ang_accel}], "
                f"dt range:[{params.dt_min}, {params.dt_max}]"
            )
        print(
            f"Solving with N={params.N}, mode={args.mode}, dt range:[{params.dt_min}, {params.dt_max}], "
            f"v:[{params.v_min}, {params.v_max}], omega:[{params.omega_min}, {params.omega_max}], "
            f"accel:[{params.max_lin_accel}, {params.max_ang_accel}] ..."
        )
    result = planner.solve(start, goal, obstacles)

    if result.success:
        print(f"Success! Total time T = {result.total_time:.4f} s, dt = {result.dt:.6f} s")
    else:
        print(f"Solver failed. Debug trajectory T = {result.total_time:.4f} s")

    plot_trajectory(result, obstacles, buf)
    if result.success:
        save_trajectory(result, f"optimization_trajectory_{scene_name}_{args.mode}")
    else:
        print("Not saving trajectory because solver did not converge to a feasible solution.")

if __name__ == "__main__":
    main()
