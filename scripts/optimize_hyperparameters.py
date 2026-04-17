"""Script to run hyperparameter optimization for SA-ZD-NIDS."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sa_zd_nids.optimization.optuna_tuner import run_optimization


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for SA-ZD-NIDS")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--input", default="data/processed.csv", help="Input data path")
    parser.add_argument("--output", default="optimization_results", help="Output directory for results")
    parser.add_argument("--trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--mode", choices=["classifier", "autoencoder", "joint", "all"], 
                       default="all", help="Optimization mode")
    
    args = parser.parse_args()
    
    print(f"Starting hyperparameter optimization...")
    print(f"Config: {args.config}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Mode: {args.mode}")
    
    if args.mode == "all":
        run_optimization(args.config, args.input, args.output)
    else:
        # Run specific optimization mode
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        from sa_zd_nids.optimization.optuna_tuner import OptunaTuner
        
        tuner = OptunaTuner(config)
        tuner.prepare_data(args.input)
        
        if args.mode == "classifier":
            study = tuner.optimize_classifier(args.trials)
            tuner.save_best_params(study, f"{args.output}/best_classifier_params.json")
        elif args.mode == "autoencoder":
            study = tuner.optimize_autoencoder(args.trials)
            tuner.save_best_params(study, f"{args.output}/best_autoencoder_params.json")
        elif args.mode == "joint":
            study = tuner.optimize_joint(args.trials)
            tuner.save_best_params(study, f"{args.output}/best_joint_params.json")
        
        print(f"Optimization completed for {args.mode} mode")


if __name__ == "__main__":
    main()
