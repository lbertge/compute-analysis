import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description='Run benchmarks for data mixing methods')
    parser.add_argument('--model', type=str, required=True, 
                      help='HuggingFace model name')
    parser.add_argument('--dataset', type=str, required=True, 
                      help='HuggingFace dataset name')
    parser.add_argument('--context_length', type=int, default=2048, 
                      help='Context window length (default: 2048)')
    parser.add_argument('--num_rounds', type=int, default=10,
                      help='T: Number of training rounds (default: 10)')
    parser.add_argument('--num_domains', type=int, default=10, 
                      help='m: Number of domains in the dataset (default: 10)')
    parser.add_argument('--delta', type=float, default=0.1, 
                      help='delta: Hyperparameter used in Skill-It and DGA methods (default: 0.1)')
    parser.add_argument('--training_iterations', type=int, default=None,
                      help='Number of training iterations (overrides dataset-based estimation)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--eval_fraction', type=float, default=0.05,
                      help='Fraction of training data to use for evaluation if validation split not available (default: 0.05)')
    args = parser.parse_args()
    
    # Run the benchmark script
    print("Running benchmark...")
    cmd = ["python", "benchmark_compute_cost.py"]
    
    # Add required arguments
    cmd.extend(["--model", args.model])
    cmd.extend(["--dataset", args.dataset])
    
    # Add optional arguments if provided
    if args.context_length:
        cmd.extend(["--context_length", str(args.context_length)])
    if args.num_rounds:
        cmd.extend(["--num_rounds", str(args.num_rounds)])
    if args.delta:
        cmd.extend(["--delta", str(args.delta)])
    if args.training_iterations:
        cmd.extend(["--training_iterations", str(args.training_iterations)])
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.eval_fraction:
        cmd.extend(["--eval_fraction", str(args.eval_fraction)])
    if args.num_domains:
        cmd.extend(["--num_domains", str(args.num_domains)])
    
    # Execute benchmark script
    subprocess.run(cmd)
    
    # Check if benchmark produced output
    expected_filename = f"benchmark_results_T{args.num_rounds}_m{args.num_domains}.csv"
    if not os.path.exists(expected_filename):
        print("Error: Benchmark did not produce results. Check for errors above.")
        return
    
    print("\nBenchmark complete!")
    print("\nResults files:")
    print(f"- {expected_filename}: Raw benchmark data")
    print(f"- benchmark_results_T{args.num_rounds}_m{args.num_domains}.png: Bar chart of FLOPs by method and dataset")

if __name__ == "__main__":
    main()