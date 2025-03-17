import argparse
import math
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt

def check_validation_split_exists(dataset_name):
    """Check if a validation split exists for the dataset"""
    try:
        # Try loading just the dataset info first
        dataset_info = load_dataset(dataset_name, split=None)
        splits = dataset_info.keys()
        
        # Check if validation split exists (might be called 'validation', 'val', 'dev', etc.)
        valid_split_names = ['validation', 'val', 'dev', 'valid', 'test']
        for split_name in valid_split_names:
            if split_name in splits:
                return split_name
                
        return None
    except Exception as e:
        print(f"Error checking validation split for {dataset_name}: {e}")
        return None

def get_model_params(model_name):
    """Get the number of parameters (N) from a HuggingFace model"""
    try:
        model = AutoModel.from_pretrained(model_name)
        params = sum(p.numel() for p in model.parameters())
        print(f"Model {model_name} has {params:,} parameters")
        return params
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

def estimate_tokens(dataset_name, split="train", context_length=2048):
    """Estimate the total number of tokens in a dataset"""
    try:
        dataset = load_dataset(dataset_name, split=split)
        # Estimate based on text field (adjust as needed for specific datasets)
        text_field = 'text' if 'text' in dataset.column_names else dataset.column_names[0]
        # Sample to estimate average token length if dataset is large
        if len(dataset) > 1000:
            sample_size = 1000
            sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
            # Convert numpy.int64 to Python int to avoid indexing issues
            sample_texts = [dataset[int(i)][text_field] for i in sample_indices]
            avg_length = sum(len(text) for text in sample_texts) / sample_size
            total_chars = avg_length * len(dataset)
        else:
            total_chars = sum(len(item[text_field]) for item in dataset)
        
        # Rough estimation: ~4 chars per token on average
        estimated_tokens = total_chars / 4
        total_tokens = estimated_tokens * context_length
        
        print(f"Dataset {dataset_name} ({split} split) estimated to have {estimated_tokens:,.0f} tokens")
        print(f"With context length {context_length}, total {split} tokens: {total_tokens:,.0f}")
        
        return total_tokens
    except Exception as e:
        print(f"Error loading dataset {dataset_name} ({split} split): {e}")
        return None

def calculate_flops(method, N, D_t, D_e, T, m, delta=0.1):
    """
    Calculate FLOPs based on the formulas from the paper
    
    Args:
        method: One of "skill-it", "aioli", "dga", "r&b"
        N: Number of model parameters
        D_t: Size of training dataset (tokens)
        D_e: Size of evaluation dataset (tokens)
        T: Number of training rounds
        m: Number of domains
        delta: Delta parameter used in some methods (default 0.1)
    
    Returns:
        FLOPs count
    """
    method = method.lower()
    
    if method == "skill-it":
        return 6 * (1 + delta) * D_t * N + 2 * (T + m) * D_e * N
    elif method == "aioli":
        return 6 * D_t * N + 2 * (T * m) * D_e * N
    elif method == "dga":
        return 6 * (1 + delta) * D_t * N + 6 * T * (D_e + m) * N
    elif method == "r&b":  # R&B (Ours)
        return 6 * D_t * N + T * m**2 * N
    else:
        raise ValueError(f"Unknown method: {method}")

def plot_results(results_df, output_file="benchmark_results.png"):
    """Plot the benchmark results"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    datasets = results_df['Dataset'].unique()
    methods = results_df['Method'].unique()
    
    x = np.arange(len(datasets))
    width = 0.2
    multiplier = 0
    
    for method in methods:
        method_data = results_df[results_df['Method'] == method]
        flops = [method_data[method_data['Dataset'] == d]['FLOPs'].values[0] for d in datasets]
        offset = width * multiplier
        rects = ax.bar(x + offset, flops, width, label=method)
        multiplier += 1
    
    ax.set_ylabel('FLOPs')
    ax.set_title('Computational Cost by Method and Dataset')
    ax.set_xticks(x + width, datasets)
    ax.legend(loc='best')
    ax.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Benchmark compute costs of data mixing methods')
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name')
    parser.add_argument('--dataset', type=str, required=True, help='HuggingFace dataset name')
    parser.add_argument('--context_length', type=int, default=2048, help='Context window length')
    parser.add_argument('--num_rounds', type=int, default=10,
                        help='T: Number of training rounds (default: 10)')
    parser.add_argument('--num_domains', type=int, default=10, 
                        help='m: Number of domains in the dataset (default: 10)')
    parser.add_argument('--delta', type=float, default=0.1, 
                        help='delta: Hyperparameter used in Skill-It and DGA methods (default: 0.1)')
    parser.add_argument('--training_iterations', type=int, default=None, 
                        help='Number of training iterations (if specified, overrides dataset-based token estimation)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training (default: 32)')
    parser.add_argument('--eval_fraction', type=float, default=0.05, 
                        help='Fraction of training data to use for evaluation if validation split not available (default: 0.05)')
    args = parser.parse_args()
    
    # Get model parameters (N)
    N = get_model_params(args.model)
    if N is None:
        print("Failed to load model. Exiting.")
        return
    
    results = []
    
    # Calculate D_t based on training iterations and batch size if provided
    if args.training_iterations is not None:
        D_t = args.training_iterations * args.batch_size * args.context_length
        print(f"Using training-based estimation: {args.training_iterations:,} iterations × {args.batch_size} batch size × {args.context_length} context length = {D_t:,} tokens")
        
        # Calculate D_e based on eval_fraction if no specific eval iterations are provided
        D_e = D_t * args.eval_fraction
        print(f"Evaluation tokens: {D_e:,} ({args.eval_fraction:.1%} of training tokens)")
        
    # If no training iterations provided, estimate from dataset
    else:
        D_t = estimate_tokens(args.dataset, split="train", context_length=args.context_length)
        if D_t is None:
            print("Failed to load dataset. Exiting.")
            return
        
        # Check if validation split exists
        validation_split = check_validation_split_exists(args.dataset)
        if validation_split:
            print(f"Found validation split: {validation_split}")
            D_e = estimate_tokens(args.dataset, split=validation_split, context_length=args.context_length)
            if D_e is None:
                print(f"Failed to load validation split. Using {args.eval_fraction:.1%} of training data instead.")
                D_e = D_t * args.eval_fraction
        else:
            print(f"No validation split found. Using {args.eval_fraction:.1%} of training data for evaluation.")
            D_e = D_t * args.eval_fraction
    
    # Use user-specified number of rounds (T)
    T = args.num_rounds
    print(f"Using T={T} training rounds as specified")
    
    # Use user-specified number of domains (m)
    m = args.num_domains
    print(f"Using m={m} domains in the dataset")
    
    methods = ["Skill-It", "Aioli", "DGA", "R&B"]
    for method in methods:
        flops = calculate_flops(method, N, D_t, D_e, T, m, args.delta)
        results.append({
            "Dataset": args.dataset,
            "Method": method,
            "FLOPs": flops,
            "N": N,
            "D_t": D_t,
            "D_e": D_e,
            "T": T,
            "m": m,
            "Dataset_config": f"{args.dataset}_T{T}_m{m}"  # Include both T and m in dataset identifier
        })
    
    # Create DataFrame and save results
    results_df = pd.DataFrame(results)
    
    # Include both T and m in the filename to avoid overwriting
    results_filename = f"benchmark_results_T{T}_m{m}.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"\nResults saved to {results_filename}")
    
    # Plot results with T and m values in filename
    plot_filename = f"benchmark_results_T{T}_m{m}.png"
    plot_results(results_df, output_file=plot_filename)
    print(f"Plot saved to {plot_filename}")
    
    # Print summary
    print("\nSummary:")
    summary = results_df.pivot(index="Dataset", columns="Method", values="FLOPs")
    print(summary)
    
    # Find the most efficient method for each dataset
    best_methods = []
    for dataset in results_df['Dataset'].unique():
        dataset_results = results_df[results_df['Dataset'] == dataset]
        best_method = dataset_results.loc[dataset_results['FLOPs'].idxmin()]
        best_methods.append({
            "Dataset": dataset,
            "Best Method": best_method['Method'],
            "FLOPs": best_method['FLOPs'],
            "Improvement over worst (%)": (1 - best_method['FLOPs'] / dataset_results['FLOPs'].max()) * 100
        })
    
    best_df = pd.DataFrame(best_methods)
    print("\nBest method per dataset:")
    print(best_df)

if __name__ == "__main__":
    main()