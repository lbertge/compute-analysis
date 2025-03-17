import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_comparison_table(results_df, output_file="method_comparison.csv"):
    """Create a table similar to Table 2 in the paper"""
    # Pivot the data to get a format similar to the table
    datasets = results_df['Dataset'].unique()
    methods = results_df['Method'].unique()
    
    # Extract domain counts for each dataset
    domain_counts = {
        "Dolly-15k": {"original": 8, "regroup": 13},
        "NI-38": {"original": 38, "regroup": 100},
        "NI-OOD": {"original": 60, "regroup": 100},
        "SlimPajama-6B": {"original": 7, "regroup": 10},
        "SI-Full-59K": {"original": 54, "regroup": 54},
        "CLIP": {"original": None, "regroup": 1000}
    }
    
    # Create a table with FLOPs(×10^18) like in the paper
    table_data = []
    for method in methods:
        row = {"Method": method}
        for dataset in datasets:
            flops = results_df[(results_df['Method'] == method) & 
                              (results_df['Dataset'] == dataset)]['FLOPs'].values[0]
            # Convert to FLOPs(×10^18)
            flops_in_10_18 = flops / 1e18
            row[f"{dataset}"] = flops_in_10_18
        table_data.append(row)
    
    comparison_df = pd.DataFrame(table_data)
    comparison_df.to_csv(output_file, index=False)
    
    return comparison_df

def create_cost_formulas_table(output_file="cost_formulas.csv"):
    """Create a table similar to Table 1 in the paper"""
    methods = [
        "Skill-It (Chen et al., 2023)",
        "Aioli (Chen et al., 2024)",
        "DGA (Fan et al., 2024a)",
        "R&B (Ours)"
    ]
    
    compute_costs = [
        "6(1+δ)DₜN+2(T+m)DₑN",
        "6DₜN+2(Tm)DₑN",
        "6(1+δ)DₜN+6T(Dₑ+m)N",
        "6DₜN+Tm²N"
    ]
    
    asymptotic_costs = [
        "O(DₜN+(T+m)DₑN)",
        "O(DₜN+TmDₑN)",
        "O(DₜN+T(Dₑ+m)N)",
        "O(DₜN+Tm²N)"
    ]
    
    cost_df = pd.DataFrame({
        "Method": methods,
        "Total Compute Cost (FLOPs)": compute_costs,
        "Asymptotic Cost": asymptotic_costs
    })
    
    cost_df.to_csv(output_file, index=False)
    return cost_df

def plot_heatmap(results_df, output_file="flops_heatmap.png"):
    """Create a heatmap of FLOPs for each method and dataset"""
    # Pivot the data
    pivot_data = results_df.pivot(index="Method", columns="Dataset", values="FLOPs")
    
    # Normalize by column (dataset) for better visualization
    normalized_data = pivot_data.apply(lambda x: x / x.max(), axis=0)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(normalized_data, annot=True, fmt=".2f", cmap="YlGnBu_r")
    plt.title("Normalized Computational Cost (lower is better)")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return normalized_data

def parameter_sensitivity(benchmark_script, output_file="parameter_sensitivity.png"):
    """Analyze how the m parameter affects the FLOPs for each method"""
    import subprocess
    import os
    
    m_values = [2, 5, 10, 20, 50]
    results = []
    
    for m in m_values:
        # Run the benchmark script with different m values
        cmd = f"python {benchmark_script} --model gpt2 --m {m}"
        subprocess.run(cmd, shell=True)
        
        # Read the results
        results_df = pd.read_csv("benchmark_results.csv")
        
        # Take average across datasets for each method
        for method in results_df['Method'].unique():
            avg_flops = results_df[results_df['Method'] == method]['FLOPs'].mean()
            results.append({
                "m": m,
                "Method": method,
                "Avg_FLOPs": avg_flops
            })
    
    # Create DataFrame and plot
    sensitivity_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    
    for method in sensitivity_df['Method'].unique():
        method_data = sensitivity_df[sensitivity_df['Method'] == method]
        plt.plot(method_data['m'], method_data['Avg_FLOPs'], marker='o', label=method)
    
    plt.xlabel('m parameter value')
    plt.ylabel('Average FLOPs')
    plt.title('Sensitivity to m parameter')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return sensitivity_df

def main():
    # Read the benchmark results
    try:
        results_df = pd.read_csv("benchmark_results.csv")
    except FileNotFoundError:
        print("Benchmark results not found. Run the benchmark script first.")
        return
    
    # Create tables
    print("Creating comparison table...")
    comparison_df = create_comparison_table(results_df)
    print(comparison_df)
    
    print("\nCreating cost formulas table...")
    cost_df = create_cost_formulas_table()
    print(cost_df)
    
    # Create visualizations
    print("\nCreating heatmap visualization...")
    normalized_data = plot_heatmap(results_df)
    
    # Optional: Run parameter sensitivity analysis
    # print("\nRunning parameter sensitivity analysis...")
    # sensitivity_df = parameter_sensitivity("benchmark_compute_cost.py")
    
    print("\nAll visualizations and tables created successfully!")

if __name__ == "__main__":
    main()