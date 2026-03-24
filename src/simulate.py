from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .simulation import SimulationConfig, run_simulation
except ImportError:
    from simulation import SimulationConfig, run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the robust MPC MuJoCo simulation.")
    parser.add_argument("--model-path", type=Path, default=SimulationConfig.model_path)
    parser.add_argument("--output-dir", type=Path, default=SimulationConfig.output_dir)
    parser.add_argument("--duration", type=float, default=SimulationConfig.duration)
    parser.add_argument("--horizon", type=int, default=SimulationConfig.horizon)
    parser.add_argument("--framerate", type=int, default=SimulationConfig.framerate)
    parser.add_argument("--disturbance-bound", type=float, default=SimulationConfig.disturbance_bound)
    parser.add_argument("--tube-disturbance-bound", type=float, default=SimulationConfig.tube_disturbance_bound)
    parser.add_argument("--seed", type=int, default=SimulationConfig.seed)
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SimulationConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        duration=args.duration,
        horizon=args.horizon,
        framerate=args.framerate,
        render_video=not args.no_render,
        save_plots=not args.no_plots,
        disturbance_bound=args.disturbance_bound,
        tube_disturbance_bound=args.tube_disturbance_bound,
        seed=args.seed,
    )
    result = run_simulation(config)

    print(f"Simulation finished with {len(result.time_steps)} steps.")
    if result.video_path is not None:
        print(f"Video: {result.video_path}")
    for label, path in result.plot_paths.items():
        print(f"{label.title()} plot: {path}")


if __name__ == "__main__":
    main()
