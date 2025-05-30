import argparse
import math
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import torch.nn as nn

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
        if 'gpt-neo' in model_name.lower():
            from transformers import GPTNeoForCausalLM
            model = GPTNeoForCausalLM.from_pretrained(model_name)
        elif 'qwen' in model_name.lower():
            from transformers import Qwen2ForCausalLM
            model = Qwen2ForCausalLM.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)
        params = sum(p.numel() for p in model.parameters())
        print(f"Model {model_name} has {params:,} parameters")
        return params, model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

def estimate_tokens(dataset_name, split="train", context_length=2048, return_num_examples=False):
    """Estimate the total number of tokens in a dataset
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        split: Dataset split to use (default: 'train')
        context_length: Context length for each example
        return_num_examples: Whether to also return the number of examples in the dataset
        
    Returns:
        total_tokens: Estimated total tokens
        num_examples: Number of examples in the dataset (if return_num_examples=True)
    """
    try:
        dataset = load_dataset(dataset_name, split=split)
        num_examples = len(dataset)
        
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
        
        print(f"Dataset {dataset_name} ({split} split) contains {num_examples:,} examples")
        print(f"Estimated to have {estimated_tokens:,.0f} tokens")
        print(f"With context length {context_length}, total {split} tokens: {total_tokens:,.0f}")
        
        if return_num_examples:
            return total_tokens, num_examples
        else:
            return total_tokens
    except Exception as e:
        print(f"Error loading dataset {dataset_name} ({split} split): {e}")
        if return_num_examples:
            return None, None
        else:
            return None

def calculate_flops(method, N, D_t, D_e, T, m, delta=0.1, gradient_fraction=0.1):
    """
    Calculate FLOPs based on the formulas from the paper
    
    Args:
        method: One of "skill-it", "aioli", "dga", "r&b", "stratified"
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
        return 6 * (1 + m * delta) * D_t * N + 2 * (T + m) * D_e * N
    elif method == "aioli":
        return 6 * D_t * N + 2 * (T * m) * D_e * N
    elif method == "dga":
        return 6 * (1 + m * delta) * D_t * N + 6 * T * (delta * D_e) * N
    elif method == "r&b":  # R&B (Ours)
        return 6 * D_t * N + T * m**2 * N * gradient_fraction
    elif method == "stratified":  # Stratified - just the training cost
        return 6 * D_t * N
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
    # Required parameters
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name')
    parser.add_argument('--dataset', type=str, required=True, help='HuggingFace dataset name')
    
    # Method parameters
    parser.add_argument('--num_rounds', type=int, default=10,
                        help='T: Number of training rounds (default: 10)')
    parser.add_argument('--num_domains', type=int, default=10, 
                        help='m: Number of domains in the dataset (default: 10)')
    parser.add_argument('--delta', type=float, default=0.1, 
                        help='delta: Hyperparameter used in Skill-It and DGA methods (default: 0.1)')
    parser.add_argument('--num_layers', type=int, default=1, 
                        help='Number of layers in the model to use as gradients (default: 1)')
    
    # Training parameters
    parser.add_argument('--context_length', type=int, default=2048, 
                        help='Context window length (default: 2048)')
    parser.add_argument('--training_iterations', type=int, default=None, 
                        help='Number of training iterations (if specified, overrides dataset-based token estimation)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training (default: 32)')
    parser.add_argument('--eval_fraction', type=float, default=0.05, 
                        help='Fraction of training examples to use for evaluation if validation split not available (default: 0.05)')
    
    args = parser.parse_args()
    
    # Get model parameters (N)
    N, model = get_model_params(args.model)
    if N is None:
        print("Failed to load model. Exiting.")
        return

    # calculate the number of layers in the model

    linear_layers = []
    linear_type = (nn.Linear, )
    for name, module in model.named_modules():
        if isinstance(module, linear_type):
            linear_layers.append(name)
    
    print(f"Number of linear layers in the model: {len(linear_layers)}")

    new_N = 0
    for layer in linear_layers[-args.num_layers:]:
        if f"{layer}.weight" in model.state_dict():
            new_N += model.state_dict()[f"{layer}.weight"].numel()
        else:
            print(f"Layer {layer} not found in model state dict")
    
    print(f"Number of parameters used in gradient computation: {new_N:,}")

    gradient_fraction = new_N / N

    results = []
    
    # Calculate D_t based on training iterations and batch size if provided
    if args.training_iterations is not None:
        # For training tokens
        D_t = args.training_iterations * args.batch_size * args.context_length
        print(f"Using training-based estimation: {args.training_iterations:,} iterations × {args.batch_size} batch size × {args.context_length} context length = {D_t:,} tokens")
        
        # For evaluation tokens, use the dataset to get example counts
        try:
            # Check if validation split exists
            validation_split = check_validation_split_exists(args.dataset)
            if validation_split:
                print(f"Found validation split: {validation_split}")
                D_e, _ = estimate_tokens(args.dataset, split=validation_split, 
                                      context_length=args.context_length,
                                      return_num_examples=True)
                if D_e is None:
                    raise Exception("Failed to load validation split")
            else:
                # No validation split, calculate based on training examples
                _, num_train_examples = estimate_tokens(args.dataset, split="train", 
                                                     context_length=args.context_length, 
                                                     return_num_examples=True)
                
                if num_train_examples is None:
                    raise Exception("Failed to get training examples count")
                
                # Use a fraction of training examples for evaluation
                num_eval_examples = int(num_train_examples * args.eval_fraction)
                
                # Calculate tokens per batch and use the same token-per-batch estimate for evaluation
                tokens_per_batch = D_t / args.training_iterations
                tokens_per_example = tokens_per_batch / args.batch_size
                
                # Total evaluation tokens
                D_e = tokens_per_example * num_eval_examples
                
                print(f"No validation split found. Using {args.eval_fraction:.1%} of training examples for evaluation.")
                print(f"Training has {num_train_examples:,} examples, using {num_eval_examples:,} examples for evaluation")
                print(f"Evaluation tokens: {D_e:,.0f}")
                
        except Exception as e:
            print(f"Error determining evaluation examples: {e}")
            print(f"Falling back to using {args.eval_fraction:.1%} of training tokens for evaluation")
            D_e = D_t * args.eval_fraction
            print(f"Evaluation tokens: {D_e:,}")
        
        
    # If no training iterations provided, estimate from dataset
    else:
        # Get training tokens and number of examples
        D_t, num_train_examples = estimate_tokens(args.dataset, split="train", 
                                                context_length=args.context_length, 
                                                return_num_examples=True)
        if D_t is None:
            print("Failed to load dataset. Exiting.")
            return
        
        # Check if validation split exists
        validation_split = check_validation_split_exists(args.dataset)
        if validation_split:
            print(f"Found validation split: {validation_split}")
            D_e, _ = estimate_tokens(args.dataset, split=validation_split, 
                                  context_length=args.context_length,
                                  return_num_examples=True)
            if D_e is None:
                print(f"Failed to load validation split. Using {args.eval_fraction:.1%} of training examples instead.")
                # Use a fraction of training examples for evaluation
                num_eval_examples = int(num_train_examples * args.eval_fraction)
                # Calculate tokens per example and multiply by number of eval examples
                tokens_per_example = D_t / num_train_examples
                D_e = tokens_per_example * num_eval_examples
                print(f"Using {num_eval_examples:,} examples for evaluation (from {num_train_examples:,} training examples)")
                print(f"Evaluation tokens: {D_e:,.0f}")
        else:
            print(f"No validation split found. Using {args.eval_fraction:.1%} of training examples for evaluation.")
            # Use a fraction of training examples for evaluation
            num_eval_examples = int(num_train_examples * args.eval_fraction)
            # Calculate tokens per example and multiply by number of eval examples
            tokens_per_example = D_t / num_train_examples
            D_e = tokens_per_example * num_eval_examples
            print(f"Using {num_eval_examples:,} examples for evaluation (from {num_train_examples:,} training examples)")
            print(f"Evaluation tokens: {D_e:,.0f}")
    
    # Use user-specified number of rounds (T)
    T = args.num_rounds
    print(f"Using T={T} training rounds as specified")
    
    # Use user-specified number of domains (m)
    m = args.num_domains
    print(f"Using m={m} domains in the dataset")
    
    methods = ["Skill-It", "Aioli", "DGA", "R&B", "Stratified"]
    for method in methods:
        flops = calculate_flops(method, N, D_t, D_e, T, m, args.delta, gradient_fraction)
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
    # Calculate and print computational overhead compared to stratified baseline
    print("\nComputational Overhead (compared to Stratified baseline):")
    overhead_df = results_df.copy()
    
    # Get stratified baseline for each dataset
    stratified_baseline = overhead_df[overhead_df['Method'] == 'Stratified'].set_index('Dataset')['FLOPs']
    
    # Calculate overhead for each method
    overhead_results = []
    for dataset in overhead_df['Dataset'].unique():
        dataset_results = overhead_df[overhead_df['Dataset'] == dataset]
        baseline = stratified_baseline[dataset]
        
        for _, row in dataset_results.iterrows():
            if row['Method'] != 'Stratified':
                overhead_results.append({
                    'Dataset': dataset,
                    'Method': row['Method'],
                    'Overhead FLOPs': row['FLOPs'] - baseline,
                    'Overhead %': ((row['FLOPs'] - baseline) / baseline) * 100
                })

    overhead_summary = pd.DataFrame(overhead_results)
    print("\nAbsolute Overhead (FLOPs):")
    print(overhead_summary.pivot(index='Dataset', columns='Method', values='Overhead FLOPs'))
    print("\nRelative Overhead (%):")
    print(overhead_summary.pivot(index='Dataset', columns='Method', values='Overhead %'))

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