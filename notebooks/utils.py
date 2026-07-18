import glob
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import stats


def parse_experiment_string(s: str) -> Dict[str, str]:
    """
    Parse an experiment identifier string (name of the files) into a named dictionary.

    The expected format of the string is:
        "device/dataset_raw_experiment_model_executionmode_frequency_timestamp"

    Notes:
        - Some of the results do not have a timestamp at the end, so we will ignore that part for now.
        - The execution mode is a combination of two parts in the filename. Since the files 
          are unfortunately named with `_` in the execution mode, we will combine those 
          two parts to get the execution mode (e.g. always_device or always_cloud).
    """
    # Normalize path separator
    s = s.replace("\\", "/")

    # Split path and filename
    device, rest = s.split("/", 1)
    parts = rest.split("_")

    if len(parts) < 7:
        raise ValueError("Unexpected experiment string format")

    dataset = parts[0]
    model = parts[3]
    # Workaround due to inconsistent file naming: always_device to always-device 
    # or always-cloud to always-cloud
    execution_mode = parts[4] + "-" + parts[5]  
    frequency = parts[6]

    return {
        "device": device,
        "dataset": dataset,
        "model": model,
        "execution_mode": execution_mode,
        "frequency": frequency,
    }


def plot_characters_vs_inference_time(
    experiment_data: List[List[pd.DataFrame]], 
    labels: List[List[str]], 
    subplot_names: List[str], 
    model: str
) -> None:
    """
    Plots scatter plots of number of input characters vs inference time with shared axes.
    """
    # Okabe-Ito Palette Colors
    COLOR_MAP = {
        'spam': '#56B4E9',        # Sky Blue
        'ag-news': '#009E73',     # Bluish Green
        'boolq': '#D55E00',       # Vermillion
        'lorem-ipsum': '#E69F00'  # Orange
    }
    DEFAULT_COLOR = 'black'
    MARKER_MAP = {'spam': 'o', 'ag-news': '^', 'boolq': 's', 'lorem-ipsum': 'D'}
    
    # Custom size mapping to compensate for marker geometry and visual weight
    SIZE_MAP = {'spam': 35, 'ag-news': 60, 'boolq': 30, 'lorem-ipsum': 25}
    DEFAULT_SIZE = 20

    sns.set(style="whitegrid")

    rows = len(experiment_data) // 2 + len(experiment_data) % 2
    
    # Enable shared axes: share x across columns, share y across rows
    fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(14, rows * 4.5), sharex='col', sharey='row')
    axes_flat = axes.flatten()

    # Loop through each model/device combination
    for idx, (ax, model_experiments, model_labels, model_name) in enumerate(zip(axes_flat, experiment_data, labels, subplot_names)):
        
        # Determine subplot positioning in the grid
        is_left_col = (idx % 2 == 0)
        is_bottom_row = (idx >= (rows - 1) * 2)

        # Sort model_experiments and model_labels so smaller datasets are plotted last (on top)
        sorted_zip = sorted(zip(model_experiments, model_labels), key=lambda x: len(x[0]), reverse=True)
        # Loop through each DataFrame and plot
        for df, label in sorted_zip:
            if 'number_of_characters' in df.columns and 'inference_time_ms' in df.columns:
                # Use the size mapping dictionary to determine marker-specific size
                base_size = SIZE_MAP.get(label.lower(), DEFAULT_SIZE)
                size = np.ones(len(df)) * base_size
                color = COLOR_MAP.get(label.lower(), DEFAULT_COLOR)
                ax.scatter(df['number_of_characters'], df['inference_time_ms'], 
                           marker=MARKER_MAP.get(label.lower(), 'o'),
                           color=color, alpha=0.6, label=label, s=size,
                           edgecolors='white', linewidths=0.2)

        ax.set_title(model_name, fontsize=25)
        
        # Only add X label to the bottom row subplots
        if is_bottom_row:
            ax.set_xlabel('Input Characters', fontsize=22)
        else:
            ax.set_xlabel('')

        # Only add Y label and keep ticks on the left column subplots
        if is_left_col:
            ax.set_ylabel('Inference Time (ms)', fontsize=22)
        else:
            ax.set_ylabel('')
            # Explicitly turn off tick labels for the right column
            ax.yaxis.set_tick_params(labelleft=False)

        # Set scale types and tick sizes inside the loop
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.grid(True, which="both", ls="-", alpha=0.5)

        if idx == 0:
            leg = ax.legend(fontsize=22, loc='lower right')
            for lh in leg.legend_handles: 
                lh.set_alpha(1.0)
                lh.set_sizes([150])

    # Remove last unused subplot if applicable BEFORE setting global limits
    if len(experiment_data) % 2 != 0:
        fig.delaxes(axes_flat[-1])

    # ---> CRITICAL FIX: Apply uniform limits globally to all remaining axes <---
    for ax in fig.axes:
        ax.set_xlim(300, 30000)
        ax.set_ylim(80, 20000)

    plt.tight_layout()
    plt.savefig(f'./plots/characters_vs_inference_time_{model}_model.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_input_character_distribution(
    datasets: List[pd.DataFrame], 
    dataset_names: List[str]
) -> None:
    """
    Plots the distribution of the number of input characters for each dataset in a single figure with 3 subplots.

    Args:
        datasets: List of DataFrames, each containing a 'number_of_characters' column for a specific dataset.
        dataset_names: The name of the datasets corresponding to each DataFrame, used for labeling the plots.
    """
    import matplotlib.ticker as ticker
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    COLOR_MAP = {
        'Spam/Ham': '#56B4E9',    # Sky Blue
        'AG News': '#009E73',     # Bluish Green
        'BoolQ': '#D55E00',       # Vermillion
        'lorem-ipsum': '#E69F00'  # Orange
    }

    for i, (dataset, name) in enumerate(zip(datasets, dataset_names)):
        ax = axes[i]  
        color = COLOR_MAP.get(name, 'gray')

        sns.histplot(
            data=dataset.dropna(),
            color=color,
            alpha=0.9,
            kde=True,
            ax=ax
        )

        ax.set_title(f"{name}", fontsize=25, pad=12)
        ax.set_xlabel('Number of Characters', fontsize=22, labelpad=10)
        
        # Only put the Y label on the first column plot
        if i == 0:
            ax.set_ylabel('Frequency', fontsize=22)
        else:
            ax.set_ylabel('')

        # Fix the cluttered x-ticks by forcing a maximum of 4 clean intervals
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Enable clean horizontal grid lines; keep vertical lines soft
        ax.grid(True, axis='y', ls='-', alpha=0.5)
        ax.grid(False, axis='x')

    # Add explicit horizontal padding between subplots to stop numbers from bumping into each other
    plt.tight_layout(pad=2.0, w_pad=0.3)
    plt.savefig(f'./plots/characters_distribution_per_dataset.pdf', format='pdf', dpi=300)
    plt.show()


def extract_basic_metrics(df: pd.DataFrame, name: str = "Server") -> Dict[str, float]:
    """
    Extracts basic queuing metrics from a raw experiment dataframe.
    Uses columns: 'job_start_ts', 'inference_end_ts', 'inference_time_ms', 'total_latency_ms'
    """
    # Work on a copy to avoid modifying the original dataframe
    df = df.copy()
    df = df.sort_values('job_start_ts')

    # Arrival Rate (Lambda)
    # Total time window of the experiment (observed from first arrival to last completion)
    start_time = df['job_start_ts'].min()
    end_time = df['inference_end_ts'].max()
    experiment_duration_sec = (end_time - start_time) / 1000.0

    num_requests = len(df)
    arrival_rate = num_requests / experiment_duration_sec if experiment_duration_sec > 0 else 0

    # Mean Service Demand (S_bar)
    # inference_time_ms is the pure processing time (service time, no queueing)
    service_times_sec = df['inference_time_ms'] / 1000.0
    mean_service_demand = service_times_sec.mean()

    # Empirical Response Time (R)
    # total_latency_ms = inference_end_ts - job_start_ts (includes queueing + service)
    response_times_sec = df['total_latency_ms'] / 1000.0

    mean_response_time = response_times_sec.mean()
    p50_response = response_times_sec.median()
    p95_response = response_times_sec.quantile(0.95)
    p99_response = response_times_sec.quantile(0.99)

    # Utilization (rho)
    # Utilization Law: rho = lambda * S
    utilization = arrival_rate * mean_service_demand

    print(f"--- Metrics for {name} ---")
    print(f"  Count:                   {num_requests}")
    print(f"  Duration:                {experiment_duration_sec:.2f} s")
    print(f"  Arrival Rate (λ):        {arrival_rate:.4f} req/s")
    print(f"  Mean Service Demand (S): {mean_service_demand:.4f} s")
    print(f"  Mean Response Time (R):  {mean_response_time:.4f} s")
    print(f"  Response Time P95:       {p95_response:.4f} s")
    print(f"  Utilization (ρ = λ*S):   {utilization:.2%}")
    print("-" * 30)

    return {
        'lambda': float(arrival_rate),
        'mean_service_time': float(mean_service_demand),
        'mean_response_time': float(mean_response_time),
        'p95_response_time': float(p95_response),
        'utilization': float(utilization)
    }


def estimate_linear_relationship(
    df: pd.DataFrame, 
    label: str = "Model", 
    plot: bool = True
) -> Tuple[float, float, float]:
    """
    Estimates the linear relationship between input characters and inference time.
    Returns (slope, intercept, std_dev_of_residuals).
    """
    x = df['number_of_characters']
    y = df['inference_time_ms'] / 1000.0 

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    predicted_y = slope * x + intercept
    residuals = y - predicted_y
    std_dev_residuals = np.std(residuals)

    print(f"--- {label} ---")
    print(f"Slope: {slope:.6f} s/char")
    print(f"Intercept: {intercept:.6f} s")
    print(f"R-squared: {r_value**2:.4f}")
    print(f"Std Dev of Residuals (Noise): {std_dev_residuals:.6f} s")

    if plot:
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, alpha=0.3, label='Data Points')
        plt.plot(x, slope * x + intercept, color='red', label=f'Fit: y={slope:.5f}x + {intercept:.3f}')
        plt.xlabel('Input Characters')
        plt.ylabel('Inference Time (s)')
        plt.title(f'Linear Fit: {label}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    return slope, intercept, std_dev_residuals


def plot_system_performance(systems_to_analyze: List[Tuple[str, pd.DataFrame]]) -> List[float]:
    """
    Plots a Kingman's Approximation analysis for given systems and calculates the mu values.
    """
    fig, axes = plt.subplots(1, len(systems_to_analyze), figsize=(18, 7), sharey=True)
    if len(systems_to_analyze) == 1:
        axes = [axes]
        
    fig.suptitle('G/G/1 Performance Analysis', fontsize=16)

    all_finite_latencies = []
    mus = []

    for i, (system_name, df_single_system) in enumerate(systems_to_analyze):
        ax = axes[i] 

        service_times_s = df_single_system['inference_time_ms'] / 1000.0
        mean_service_time = service_times_s.mean()
        std_dev_service_time = service_times_s.std()
        mu = 1.0 / mean_service_time
        mus.append(mu)
        cs = std_dev_service_time / mean_service_time

        print(f"--- Analyzing: {system_name} ---")
        print(f"Mean Service Time (E[S]): {mean_service_time:.4f} s")
        print(f"Service Rate (μ): {mu:.2f} req/s")
        print(f"Service Time CoV (cs): {cs:.2f}\n")

        lambda_range = np.linspace(0.01, mu * 0.999, 1000)

        scenarios = {
            "Deterministic Arrivals (ca=0)": 0.0,
            "Poisson Arrivals (ca=1)": 1.0
        }
        results = {}

        for scenario_name, ca in scenarios.items():
            response_times = []
            for lam in lambda_range:
                rho = lam * mean_service_time
                if rho >= 1:
                    mean_response_time = float('inf')
                else:
                    mean_wait_time = (rho / (1 - rho)) * ((ca**2 + cs**2) / 2) * mean_service_time
                    mean_response_time = mean_wait_time + mean_service_time
                response_times.append(mean_response_time)
            results[scenario_name] = response_times
            all_finite_latencies.extend([r for r in response_times if r < float('inf')])

        for scenario_name, latencies in results.items():
            ax.plot(lambda_range, latencies, label=scenario_name, linewidth=2)

        ax.axvline(x=mu, color='r', linestyle='--', label=f'Saturation Point (μ = {mu:.2f} req/s)')
        ax.set_xlabel('Arrival Rate λ (requests/second)')
        ax.set_title(f'Performance of {system_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Mean Response Time (s)')

    if all_finite_latencies:
        upper_limit = np.percentile(all_finite_latencies, 99) * 1.2
        plt.ylim(bottom=0, top=upper_limit)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()
    return mus


def simulate_routing_synthetic(
    thresholds: Union[List[int], range], 
    lambda_total: float, 
    num_jobs: int = 10000, 
    ca: float = 1.0,
    char_params: Tuple[float, float] = (500.0, 200.0), 
    dev_model: Tuple[float, float, float] = (0.001, 0.1, 0.0), 
    cloud_model: Tuple[float, float, float] = (0.0005, 0.05, 0.0),
    add_noise: bool = False
) -> pd.DataFrame:
    """
    Synthetic simulation that generates data on-the-fly using a linear model.
    Service Time = Slope * Chars + Intercept + Noise
    """
    mean_inter_arrival = 1.0 / lambda_total if lambda_total > 0 else 1.0
    results = []

    char_mu, char_sigma = char_params
    slope_d, int_d, std_res_d = dev_model
    slope_c, int_c, std_res_c = cloud_model

    for T in thresholds:
        t_now = 0.0
        server_free_dev = 0.0
        server_free_cloud = 0.0
        total_response_time = 0.0

        random.seed(42)

        for _ in range(num_jobs):
            # Generate Arrival
            if ca == 0.0:
                inter_arrival = mean_inter_arrival
            elif ca == 1.0:
                inter_arrival = random.expovariate(lambda_total)
            else:
                alpha = 1.0 / (ca**2)
                beta = mean_inter_arrival / alpha
                inter_arrival = random.gammavariate(alpha, beta)

            t_now += inter_arrival

            # Generate Synthetic Job (Correlated)
            chars = max(10, random.gauss(char_mu, char_sigma))

            # Calculate Service Times based on Linear Model + Random Noise
            noise = 0.0
            if add_noise:
                # Noise represents variability not explained by length
                noise = random.gauss(0, 0.05) # 50ms noise
            s_dev = max(0.01, (slope_d * chars + int_d) + noise)
            s_cloud = max(0.01, (slope_c * chars + int_c) + noise)

            # Routing & Queueing
            if chars <= T:
                start_service = max(t_now, server_free_dev)
                completion = start_service + s_dev
                server_free_dev = completion
                response = completion - t_now
            else:
                start_service = max(t_now, server_free_cloud)
                completion = start_service + s_cloud
                server_free_cloud = completion
                response = completion - t_now

            total_response_time += response

        avg_latency = total_response_time / num_jobs
        results.append({'threshold': T, 'sim_latency': avg_latency})

    return pd.DataFrame(results)


def plot_threshold_comparisons(
    test_lambdas: List[float], 
    thresholds: Union[List[int], range], 
    char_params: Tuple[float, float], 
    dev_model: Tuple[float, float, float], 
    cloud_model: Tuple[float, float, float]
) -> None:
    """
    Plots the impact of the threshold choice across different lambdas and arrival patterns.
    """
    scenarios = {
        r"Deterministic Arrivals ($c_{a}=0.0$)": 0.0,
        r"Poisson Arrivals ($c_{a}=1.0$)": 1.0
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_lambdas)))
    all_finite_latencies = []

    for i, (title, ca_val) in enumerate(scenarios.items()):
        ax = axes[i]
        print(f"--- Running simulations for {title} ---")

        for j, lam in enumerate(test_lambdas):
            print(f"  -> Simulating λ = {lam:.1f} req/s")
            
            sim_res = simulate_routing_synthetic(
                thresholds=thresholds, 
                lambda_total=lam, 
                num_jobs=50000,  
                ca=ca_val,
                char_params=char_params, 
                dev_model=dev_model, 
                cloud_model=cloud_model
            )
            
            ax.plot(sim_res['threshold'], sim_res['sim_latency'], 
                    label=f'λ = {lam:.1f}', color=colors[j], linewidth=2)
            
            finite_vals = sim_res[sim_res['sim_latency'] != float('inf')]['sim_latency']
            all_finite_latencies.extend(finite_vals.dropna().tolist())

        ax.set_xlabel('Threshold (Characters)')
        ax.set_title(title)
        ax.grid(True, which="both", linestyle='--', linewidth=0.5) 
        ax.set_yscale('log') 

    axes[0].set_ylabel('Mean Response Time (s) [Log Scale]')
    axes[0].legend(title="Arrival Rate (req/s)")

    if all_finite_latencies:
        positive_latencies = [l for l in all_finite_latencies if l > 0]
        if positive_latencies:
            upper_lim = np.percentile(positive_latencies, 99) * 1.5 
            lower_lim = min(positive_latencies) * 0.9
            plt.ylim(lower_lim, upper_lim)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_dir = Path('./plots')
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    filename = out_dir / f'threshold_comparison_{timestamp}.pdf'
    plt.savefig(filename, format='pdf', dpi=300)
    print(f"Saved plot: {filename}")

    plt.show()


def compare_policy_performance_with_table(
    test_lambdas: List[float], 
    policies_to_test: List[str], 
    char_params: Tuple[float, float], 
    dev_model: Tuple[float, float, float], 
    cloud_model: Tuple[float, float, float], 
    name_device: str, 
    name_cloud: str, 
    table_and_plot: bool = True, 
    num_sim_requests: int = 500000, 
    char_pool: Optional[Union[pd.Series, np.ndarray]] = None
) -> List[Dict[str, Any]]:
    """
    Compares the simulated performance of various routing policies and returns the statistics.
    """
    detailed_results = []
    print("--- Running simulations for different arrival rates ---")

    for lam in test_lambdas:
        print(f"\n>>> Simulating for λ = {lam:.2f} req/s...")
        
        request_stream = []
        current_time = 0.0
        
        if char_pool is not None and not (isinstance(char_pool, pd.Series) and char_pool.empty):
            print("  -> Using empirical character distribution from provided data pool.")
            input_sizes = np.random.choice(char_pool, size=num_sim_requests, replace=True)
        else:
            print("  -> WARNING: No character pool provided. Falling back to synthetic Normal distribution.")
            char_mean, char_std = char_params
            input_sizes = [int(max(1, np.random.normal(char_mean, char_std))) for _ in range(num_sim_requests)]

        for i in range(num_sim_requests):
            inter_arrival_time = random.expovariate(lam)
            current_time += inter_arrival_time
            request_stream.append({'id': i, 'arrival_time': current_time, 'size': input_sizes[i]})

        optimal_T_stateless = 0
        if 'Stateless Threshold' in policies_to_test:
            thresholds_scan = range(0, 10000, 1) # Scan thresholds to find the best one
            sim_results_for_T = simulate_routing_synthetic(
                thresholds_scan, lam, num_jobs=2000, ca=1.0, 
                char_params=char_params, 
                dev_model=dev_model, 
                cloud_model=cloud_model
            )
            optimal_T_stateless = sim_results_for_T.loc[sim_results_for_T['sim_latency'].idxmin()]['threshold']
            print(f"  Optimal stateless T for λ={lam:.2f} is {optimal_T_stateless} chars.")

        schedulers: Dict[str, Any] = {}
        for policy_name in policies_to_test:
            if 'Always Device' in policy_name:
                schedulers[policy_name] = HeroPolicyScheduler(
                    dev_model=dev_model, cloud_model=cloud_model, force_policy='device'
                )
            elif 'Always Cloud' in policy_name:
                schedulers[policy_name] = HeroPolicyScheduler(
                    dev_model=dev_model, cloud_model=cloud_model, force_policy='cloud'
                )
            elif policy_name == 'Round Robin 2x Cloud':
                schedulers[policy_name] = MultiServerScheduler(
                    model=cloud_model, num_servers=2
                )
            elif policy_name == 'Round Robin 2x Device':
                schedulers[policy_name] = MultiServerScheduler(
                    model=dev_model, num_servers=2
                )
            elif policy_name == 'Round Robin (Device/Cloud)':
                schedulers[policy_name] = HeroPolicyScheduler(
                    dev_model=dev_model, cloud_model=cloud_model, force_policy='round_robin'
                )
            else: 
                schedulers[policy_name] = HeroPolicyScheduler(
                    dev_model=dev_model, cloud_model=cloud_model
                )

        for policy_name in policies_to_test:
            stats = run_detailed_policy_simulation(
                policy_name, 
                schedulers[policy_name], 
                request_stream, 
                stateless_threshold=optimal_T_stateless
            )
            df_stats = pd.DataFrame(stats)
            
            df_stats['total_latency'] = df_stats['wait_time'] + df_stats['service_time']

            avg_service_time = df_stats['service_time'].mean()
            avg_total_latency = df_stats['total_latency'].mean()
            std_total_latency = df_stats['total_latency'].std()
            std_service_time = df_stats['service_time'].std()

            # Calculation of total_simulation_time
            df_stats['finish_time'] = df_stats['wait_time'] + df_stats['service_time'] + pd.Series([r['arrival_time'] for r in request_stream])
            total_simulation_time = df_stats['finish_time'].max()
            
            df_stats['decision_clean'] = df_stats['decision'].str.strip().str.lower()
            device_stats = df_stats[df_stats['decision_clean'] == 'device']
            cloud_stats = df_stats[df_stats['decision_clean'] == 'cloud']
            
            
            # Calculate effective arrival rate to each queue
            lambda_device = len(device_stats) / total_simulation_time if total_simulation_time > 0 else 0
            lambda_cloud = len(cloud_stats) / total_simulation_time if total_simulation_time > 0 else 0
            
            # Calculate average wait time in each queue
            avg_wait_device = device_stats['wait_time'].mean() if not device_stats.empty else 0
            avg_wait_cloud = cloud_stats['wait_time'].mean() if not cloud_stats.empty else 0
            
            # This is the application of Little's Law: Lq = λq * Wq
            # If avg_wait_device is non-zero, this will produce a non-zero queue length.
            avg_q_len_device = lambda_device * avg_wait_device
            avg_q_len_cloud = lambda_cloud * avg_wait_cloud
            
            detailed_results.append({
                'Lambda': lam,
                'Policy': policy_name,
                'Total Latency (s)': avg_total_latency,
                'Total Latency std (s)': std_total_latency,
                'Avg Inference Time (s)': avg_service_time,
                'Inference Time std (s)': std_service_time,
                'Avg Device Queue Length': avg_q_len_device,
                'Avg Cloud Queue Length': avg_q_len_cloud
            })
            print(f"  - {policy_name:<20}: Done.")

    if table_and_plot:
        results_df = pd.DataFrame(detailed_results)

        for policy in policies_to_test:
            print(f"\n\n--- Results for: {policy} ---")
            policy_df = results_df[results_df['Policy'] == policy].set_index('Lambda')
            display(policy_df[[
                'Total Latency (s)', 
                'Avg Inference Time (s)', 
                'Avg Device Queue Length', 
                'Avg Cloud Queue Length'
            ]].style.format({
                'Total Latency (s)': '{:.4f}',
                'Avg Inference Time (s)': '{:.4f}',
                'Avg Device Queue Length': '{:.2f}',
                'Avg Cloud Queue Length': '{:.2f}'
            }).background_gradient(
                cmap='viridis', 
                subset=['Total Latency (s)', 'Avg Device Queue Length', 'Avg Cloud Queue Length']
            ))

        sns.set_theme(style="whitegrid")

        g = sns.FacetGrid(results_df, col="Policy", col_wrap=2, height=5, aspect=1.2, sharey=False)
        g.map_dataframe(sns.lineplot, x="Lambda", y="Avg Device Queue Length", color='#1f77b4', marker='o', label="Device Queue")
        g.map_dataframe(sns.lineplot, x="Lambda", y="Avg Cloud Queue Length", color='#2ca02c', marker='x', label="Cloud Queue")

        g.set_axis_labels("Arrival Rate λ (req/s)", "Average Queue Length (Lq)")
        g.set_titles(col_template="{col_name} Policy")
        g.add_legend(title="Queue Type")
        sns.move_legend(g, "lower center", bbox_to_anchor=(0.475, 0.9))

        g.fig.suptitle('Average Queue Length vs. Arrival Rate for Each Policy', y=1.03, fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.9, 0.97])
        plt.show()

    return detailed_results


class HeroPolicyScheduler:
    """
    HERO: A stateful scheduler that uses the 'Join the Shortest Expected Queue' (JSEQ) policy.
    It keeps track of when each server (device and cloud) will be free and routes
    incoming requests to the server that is predicted to finish the job first.
    """
    def __init__(
        self, 
        dev_model: Tuple[float, float, float], 
        cloud_model: Tuple[float, float, float], 
        force_policy: Optional[str] = None
    ) -> None:
        self.dev_model = dev_model
        self.cloud_model = cloud_model
        self.device_free_at = 0.0
        self.cloud_free_at = 0.0
        self.rr_next_is_device = True 

        if force_policy and force_policy not in ['device', 'cloud', 'round_robin']:
            raise ValueError("force_policy must be 'device', 'cloud', 'round_robin', or None.")
        self.force_policy = force_policy

    def reset(self) -> None:
        """Resets the scheduler's state by clearing the queues."""
        self.device_free_at = 0.0
        self.cloud_free_at = 0.0
        self.rr_next_is_device = False

    def _predict_service_time(self, size: float, model: Tuple[float, float, float]) -> float:
        """
        Predicts the service time in seconds for a given job size.
        """
        slope, intercept, _ = model 
        predicted_s = slope * size + intercept
        return max(0.0, predicted_s)

    def decide_at_time(self, size: float, arrival_time: float) -> str:
        """
        Makes a routing decision for a new request based on the JSEQ policy.
        """
        if self.force_policy == 'device':
            return "Device"
        if self.force_policy == 'cloud':
            return "Cloud"
        if self.force_policy == 'round_robin':
            if self.rr_next_is_device:
                self.rr_next_is_device = False 
                return "Device"
            else:
                self.rr_next_is_device = True 
                return "Cloud"

        st_dev = self._predict_service_time(size, self.dev_model)
        st_cloud = self._predict_service_time(size, self.cloud_model)

        finish_dev = max(arrival_time, self.device_free_at) + st_dev
        finish_cloud = max(arrival_time, self.cloud_free_at) + st_cloud

        if finish_dev <= finish_cloud:
            return "Device"
        else:
            return "Cloud"
        
    def update_server_state(self, decision: str, actual_finish_time: float) -> None:
        """
        Updates the state of the chosen server with the actual finish time.
        """
        if decision == "Device":
            self.device_free_at = actual_finish_time
        elif decision == "Cloud":
            self.cloud_free_at = actual_finish_time
        # If round-robin was used, the decision will already be 'Device' or 'Cloud'.

    def route_to_device(self, size: float, arrival_time: float) -> Tuple[str, float, float]:
        """Routes a job to the device and returns its stats."""
        st_dev = self._predict_service_time(size, self.dev_model)
        start_dev = max(arrival_time, self.device_free_at)
        finish_dev = start_dev + st_dev
        self.device_free_at = finish_dev
        return "Device", start_dev, finish_dev

    def route_to_cloud(self, size: float, arrival_time: float) -> Tuple[str, float, float]:
        """Routes a job to the cloud and returns its stats."""
        st_cloud = self._predict_service_time(size, self.cloud_model)
        start_cloud = max(arrival_time, self.cloud_free_at)
        finish_cloud = start_cloud + st_cloud
        self.cloud_free_at = finish_cloud
        return "Cloud", start_cloud, finish_cloud
    

class MultiServerScheduler:
    """
    A scheduler for a pool of 'c' identical servers, routing via round-robin.
    This is used to simulate policies like '2x Cloud' or '2x Device'.
    """
    def __init__(self, model: Tuple[float, float, float], num_servers: int = 2) -> None:
        self.model = model
        self.num_servers = num_servers
        self.server_free_times = [0.0] * num_servers
        self.next_server_idx = 0
        self.last_assigned_server_idx = -1 

    def reset(self) -> None:
        """Resets the scheduler's state."""
        self.server_free_times = [0.0] * self.num_servers
        self.next_server_idx = 0
        self.last_assigned_server_idx = -1

    def _predict_service_time(self, size: float, model: Tuple[float, float, float]) -> float:
        """Predicts the service time in seconds for a given job size."""
        slope, intercept, _ = model 
        predicted_s = slope * size + intercept
        return max(0.0, predicted_s)

    def decide_at_time(self, size: float, arrival_time: float) -> str:
        """
        Makes a routing decision using round-robin. ONLY decides.
        """
        server_idx = self.next_server_idx
        self.last_assigned_server_idx = server_idx 
        self.next_server_idx = (self.next_server_idx + 1) % self.num_servers
        return "Multi-Server"

    def update_server_state(self, decision: str, actual_finish_time: float) -> None:
        """
        Updates the state of the chosen server with the actual finish time.
        """
        if decision == "Multi-Server" and self.last_assigned_server_idx != -1:
            self.server_free_times[self.last_assigned_server_idx] = actual_finish_time
    

def run_detailed_policy_simulation(
    policy_name: str, 
    scheduler: Union[HeroPolicyScheduler, MultiServerScheduler], 
    requests: List[Dict[str, float]], 
    stateless_threshold: Optional[int] = None
) -> List[Dict[str, Union[float, str]]]:
    """
    Runs a simulation and returns detailed statistics for each request.
    """
    request_stats = []
    scheduler.reset()

    stateless_state = {'device_free_at': 0.0, 'cloud_free_at': 0.0}

    dev_model = getattr(scheduler, 'dev_model', None)
    cloud_model = getattr(scheduler, 'cloud_model', None)
    multi_server_model = getattr(scheduler, 'model', None)

    for req in requests:
        arrival_time = req['arrival_time']
        size = req['size']
        
        is_stateless = 'Stateless Threshold' in policy_name
        if is_stateless and stateless_threshold is not None:
            decision = "Device" if size <= stateless_threshold else "Cloud"
        else:
            decision = scheduler.decide_at_time(size, arrival_time)

        actual_service_time = 0.0
        start_time = 0.0

        if decision == "Device" and dev_model:
            slope, intercept, std_dev = dev_model
            base_time = slope * size + intercept
            noise = random.gauss(0, std_dev) if std_dev > 0 else 0
            actual_service_time = max(0.01, base_time + noise)
            
            free_at_time = stateless_state['device_free_at'] if is_stateless else getattr(scheduler, 'device_free_at', 0.0)
            start_time = max(arrival_time, free_at_time)

        elif decision == "Cloud" and cloud_model:
            slope, intercept, std_dev = cloud_model
            base_time = slope * size + intercept
            noise = random.gauss(0, std_dev) if std_dev > 0 else 0
            actual_service_time = max(0.01, base_time + noise)
            
            free_at_time = stateless_state['cloud_free_at'] if is_stateless else getattr(scheduler, 'cloud_free_at', 0.0)
            start_time = max(arrival_time, free_at_time)

        elif decision == "Multi-Server" and multi_server_model:
            slope, intercept, std_dev = multi_server_model
            base_time = slope * size + intercept
            noise = random.gauss(0, std_dev) if std_dev > 0 else 0
            actual_service_time = max(0.01, base_time + noise)
            server_idx = getattr(scheduler, 'last_assigned_server_idx', 0)
            server_times = getattr(scheduler, 'server_free_times', [0.0])
            start_time = max(arrival_time, server_times[server_idx])

        finish_time = start_time + actual_service_time
        
        if is_stateless:
            if decision == "Device":
                stateless_state['device_free_at'] = finish_time
            else:
                stateless_state['cloud_free_at'] = finish_time
        else:
            scheduler.update_server_state(decision, finish_time)

        wait_time = start_time - arrival_time
        request_stats.append({'wait_time': wait_time, 'service_time': actual_service_time, 'decision': decision})

    return request_stats


def plot_policy_comparison(
    detailed_results: List[Dict[str, Any]], 
    policies_to_test: List[str], 
    dataset_name: str = "unspecified",
    add_errorbars: bool = False,
    show_x_axis: bool = True
) -> None:
    """
    Plots the final line graph mapping mean response times against the arrival rate lambdas.
    Updated for strict black-and-white print compliance and high contrast, with fixed color matching.
    """
    results_df = pd.DataFrame(detailed_results)
    results_df = results_df[results_df['Policy'].isin(policies_to_test)]

    is_experimental = results_df['Policy'].str.contains("(Experiment)", regex=False)
    standard_policies_df = results_df[~is_experimental]
    experimental_policies_df = results_df[is_experimental]
    
    sim_policies = standard_policies_df['Policy'].unique().tolist()
    experimental_policy_names = experimental_policies_df['Policy'].unique().tolist()

    # Okabe-Ito high-contrast, colorblind-safe palette
    okabe_ito = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#D55E00', '#0072B2', '#F0E442', '#000000']
    
    # Map each simulation policy to a specific color so we can reuse it for experimental points
    sim_color_dict = dict(zip(sim_policies, okabe_ito[:len(sim_policies)]))

    # Distinct markers for grayscale separation
    marker_shapes = ['o', 's', '^', 'D', 'v', 'p', '*'][:len(sim_policies)]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    # 1. Add the clear, non-bolded title requested by the reviewer with exact casing
    TITLE_MAP = {
        'boolq': 'BoolQ',
        'ag-news': 'AG News',
        'spam/ham': 'Spam/Ham',
    }
    
    if dataset_name != "unspecified":
        clean_name = dataset_name.lower().strip()
        # Look up exact casing, fallback to the generic .title() if not found
        title_text = TITLE_MAP.get(clean_name, dataset_name.replace('_', ' ').title())
    else:
        title_text = "Dataset Performance"

    ax.set_title(title_text, fontweight='normal', fontsize=15, pad=10)

    if not standard_policies_df.empty:
        sns.lineplot(
            data=standard_policies_df,
            legend='full',
            x='Lambda',
            y='Total Latency (s)',
            hue='Policy',
            hue_order=sim_policies,
            style='Policy',
            style_order=sim_policies,
            ax=ax,
            palette=sim_color_dict,
            markers=marker_shapes, # Forces distinct geometric shapes
            dashes=True,           # Forces distinct line styles (solid, dashed, dotted)
            markersize=6,          
            linewidth=2.5,         
            errorbar='sd'
        )
        
        if add_errorbars and 'Total Latency std (s)' in standard_policies_df.columns:
            for policy in sim_policies:
                if (policy == 'HERO') & False:
                    policy_data = standard_policies_df[standard_policies_df['Policy'] == policy]
                    if not policy_data.empty:
                        ax.errorbar(
                            x=policy_data['Lambda'],
                            y=policy_data['Total Latency (s)'],
                            yerr=policy_data['Total Latency std (s)'],
                            fmt='none', 
                            color=sim_color_dict.get(policy),
                            capsize=5,
                            alpha=0.6
                        )

    # 2. Plot all the EXPERIMENTAL data mapped to their parent colors
    if not experimental_policies_df.empty:
        for policy_name in experimental_policy_names:
            policy_data = experimental_policies_df[experimental_policies_df['Policy'] == policy_name]
            y_err_col = 'Total Latency std (s)' if 'Total Latency std (s)' in policy_data.columns else None
            
            # Extract the base policy name (e.g., "Always Device (Experiment)" -> "Always Device")
            base_name = policy_name.replace(' (Experiment)', '').strip()
            
            # -> CRITICAL FIX: Robust substring match to find the parent color <-
            matched_color = '#000000' # Default fallback
            for sim_pol, color in sim_color_dict.items():
                if base_name in sim_pol:
                    matched_color = color
                    break
            
            ax.errorbar(
                x=policy_data['Lambda'],
                y=policy_data['Total Latency (s)'],
                yerr=policy_data[y_err_col].values if y_err_col else None,
                label=policy_name,
                fmt='X',           
                color=matched_color,
                markersize=8,               
                markeredgecolor='black',    
                markeredgewidth=0.5,
                capsize=4,
                elinewidth=1.5,
                zorder=10          
            )

    ax.set_ylabel('Mean Response Time (s)', fontsize=15)
    ax.set_xlabel('Arrival Rate λ (req/s)', fontsize=15)
    ax.set_ylim(0, 2.5)
    ax.set_xlim(0, 16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.grid(axis='y', linestyle='-', alpha=0.5)
    ax.grid(axis='x', linestyle='-', alpha=0.5)

    if show_x_axis:
        ax.set_xlabel('Arrival Rate λ (req/s)', fontsize=15)
    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x', which='both', labelbottom=False)  # Hides the numbers
    
    # Force Matplotlib to generate the legend based on the plot content
    temp_leg = ax.legend()
    handles = temp_leg.legend_handles
    labels = [text.get_text() for text in temp_leg.get_texts()]
    temp_leg.remove()

    plt.tight_layout() 
    
    out_dir = Path('./plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_ds = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in str(dataset_name)).strip().replace(' ', '_')
    filename = out_dir / f'policy_comparison_{safe_ds}.pdf'
    plt.savefig(filename, format='pdf', dpi=300)
    print(f"Saved plot: {filename}")

    plot_legend_only(handles, labels, out_dir)
    plt.show()


def plot_legend_only(
    handles: List[Any], 
    labels: List[str], 
    out_dir: Path
) -> None:
    """
    Creates and saves a plot containing only the legend.
    """
    import matplotlib.lines as mlines

    fig_legend = plt.figure(figsize=(8, 4)) 
    
    # Manually rebuild the handles for the experimental data to guarantee the 'X' marker shows
    fixed_handles = []
    for h, label in zip(handles, labels):
        if "(Experiment)" in label:
            # extract the color from the original handle container
            if hasattr(h, 'lines'):  # It's an ErrorbarContainer
                color = h.lines[0].get_color()
            elif hasattr(h, 'get_color'):  # It's a standard Line2D
                color = h.get_color()
            else:
                color = 'black' # Fallback
                
            # Build a custom proxy handle with the 'X' marker, matching the exact styling
            proxy = mlines.Line2D(
                [], [], 
                color=color, 
                marker='X', 
                markersize=8, 
                markeredgecolor='black', 
                markeredgewidth=0.5, 
                linestyle='None' # Removes the trailing line, leaving just the marker
            )
            fixed_handles.append(proxy)
        else:
            fixed_handles.append(h)

    legend = fig_legend.legend(
        fixed_handles, 
        labels, 
        loc='center', 
        frameon=False, 
        title=r"Applied Policy", 
        title_fontproperties={'size': 14},
        ncol=1,            
        fontsize=12
    )
    plt.tight_layout()
    
    plt.axis('off')

    legend_filename = out_dir / f'policy_comparison_legend.pdf'
    
    fig_legend.savefig(
        legend_filename, 
        format='pdf', 
        dpi=300, 
        bbox_inches='tight',
        pad_inches=0.0, 
        bbox_extra_artists=(legend,)
    )
    print(f"Saved complete legend-only plot: {legend_filename}")

    plt.show()
    plt.close(fig_legend)


def load_and_plot_policy_results(
    file_path_str: Union[str, Path], 
    augment_with_jseq: bool = False, 
    augment_with_baseline: bool = False, 
    dataset_name_for_exp: Optional[str] = None, 
    dataset_name_title: str = "unspecified",
    show_x_axis: bool = True
) -> None:
    """
    Loads policy simulation results from a CSV, optionally augments them with
    experimental JSEQ data, and then plots the comparison.
    """
    file_path = Path(file_path_str)
    
    try:
        results_df = pd.read_csv(file_path)
        if 'Policy' not in results_df.columns:
            print(f"[ERROR]: The required 'Policy' column was not found in {file_path}.")
            return 
            
        detailed_results = results_df.to_dict('records')
        policies_to_test = results_df['Policy'].unique().tolist()

        policies_to_test = [
            p for p in policies_to_test 
            if not p.startswith('Round Robin 2x Device') and not p.startswith('Round Robin 2x Cloud')
        ]

        print(f"[SUCCESS]: Loaded data from {file_path}")
        print(f"           Found {len(detailed_results)} records and {len(policies_to_test)} policies to plot.")

    except FileNotFoundError:
        print(f"[ERROR]: The file was not found at {file_path}")
        return 
    except Exception as e:
        print(f"An unexpected error occurred while loading the base CSV: {e}")
        return 

    if augment_with_baseline and dataset_name_for_exp:
        always_cloud_exp_dir = Path('../results/baseline_experiments/desktop-cloud-exp')
        always_device_exp_dir = Path('../results/baseline_experiments/desktop-device-exp')
        always_cloud_policy_name = 'Always Cloud (Experiment)'
        always_device_policy_name = 'Always Device (Experiment)'

        if not always_device_exp_dir.exists():
            print(f"  [WARNING]: Always Device experiment directory not found at {always_device_exp_dir}.")
        else:
            print(f"\nLoading experimental Always Device results for '{dataset_name_for_exp}'...")
            # Look for RAW files instead of stats files
            always_device_raw_files = sorted(always_device_exp_dir.glob(f'{dataset_name_for_exp}*raw*.csv'))
            
            if not always_device_raw_files:
                print(f"  [WARNING]: No Always Device raw files found for dataset '{dataset_name_for_exp}'.")
            else:
                always_device_data_added = False
                print("  --- Sanity Check: Always Device Experimental Values ---")
                for raw_file in always_device_raw_files:
                    try:
                        parts = raw_file.stem.split('_')
                        lambda_str = parts[1].replace('lambda', '')
                        lambda_val = float(lambda_str)
                        
                        raw_df = pd.read_csv(raw_file)
                        raw_df.columns = raw_df.columns.str.strip()
                        
                        latencies_ms = raw_df['total_latency_ms']
                        avg_latency_ms = latencies_ms.mean()
                        std_latency_ms = latencies_ms.std()
                        
                        avg_latency_s = avg_latency_ms / 1000.0
                        std_latency_s = std_latency_ms / 1000.0
                        
                        print(f"    - Lambda: {lambda_val:<4} -> Avg: {avg_latency_s:.4f}s, Std: {std_latency_s:.4f}s")

                        detailed_results.append({
                            'Policy': always_device_policy_name,
                            'Lambda': lambda_val,
                            'Total Latency (s)': avg_latency_s,
                            'Total Latency std (s)': std_latency_s
                        })
                        always_device_data_added = True
                    except (ValueError, IndexError, KeyError) as e:
                        print(f"  [ERROR]: Could not parse or process {raw_file.name}: {e}")
                print("  --------------------------------------------")

                if always_device_data_added and always_device_policy_name not in policies_to_test:
                    policies_to_test.append(always_device_policy_name)
                    print(f"\nAppended '{always_device_policy_name}' to policies_to_test.")

        if not always_cloud_exp_dir.exists():
            print(f"  [WARNING]: Always Cloud experiment directory not found at {always_cloud_exp_dir}.")
        else:
            print(f"\nLoading experimental Always Cloud results for '{dataset_name_for_exp}'...")
            # Look for RAW files instead of stats files
            always_cloud_raw_files = sorted(always_cloud_exp_dir.glob(f'{dataset_name_for_exp}*raw*.csv'))
            
            if not always_cloud_raw_files:
                print(f"  [WARNING]: No Always Cloud raw files found for dataset '{dataset_name_for_exp}'.")
            else:
                always_cloud_data_added = False
                print("  --- Sanity Check: Always Cloud Experimental Values ---")
                for raw_file in always_cloud_raw_files:
                    try:
                        parts = raw_file.stem.split('_')
                        lambda_str = parts[1].replace('lambda', '')
                        lambda_val = float(lambda_str)
                        
                        raw_df = pd.read_csv(raw_file)
                        raw_df.columns = raw_df.columns.str.strip()
                        
                        latencies_ms = raw_df['total_latency_ms']
                        avg_latency_ms = latencies_ms.mean()
                        std_latency_ms = latencies_ms.std()
                        
                        avg_latency_s = avg_latency_ms / 1000.0
                        std_latency_s = std_latency_ms / 1000.0
                        
                        print(f"    - Lambda: {lambda_val:<4} -> Avg: {avg_latency_s:.4f}s, Std: {std_latency_s:.4f}s")

                        detailed_results.append({
                            'Policy': always_cloud_policy_name,
                            'Lambda': lambda_val,
                            'Total Latency (s)': avg_latency_s,
                            'Total Latency std (s)': std_latency_s
                        })
                        always_cloud_data_added = True
                    except (ValueError, IndexError, KeyError) as e:
                        print(f"  [ERROR]: Could not parse or process {raw_file.name}: {e}")
                print("  --------------------------------------------")

                if always_cloud_data_added and always_cloud_policy_name not in policies_to_test:
                    policies_to_test.append(always_cloud_policy_name)
                    print(f"\nAppended '{always_cloud_policy_name}' to policies_to_test.")

    if augment_with_jseq and dataset_name_for_exp:
        jseq_exp_dir = Path('../results/policy_experiments')
        jseq_policy_name = 'HERO (Experiment)'

        if not jseq_exp_dir.exists():
            print(f"  [WARNING]: JSEQ experiment directory not found at {jseq_exp_dir}.")
        else:
            print(f"\nLoading experimental JSEQ results for '{dataset_name_for_exp}'...")
            # Look for RAW files instead of stats files
            jseq_raw_files = sorted(jseq_exp_dir.glob(f'{dataset_name_for_exp}*raw*.csv'))
            
            if not jseq_raw_files:
                print(f"  [WARNING]: No JSEQ raw files found for dataset '{dataset_name_for_exp}'.")
            else:
                jseq_data_added = False
                print("  --- Sanity Check: JSEQ Experimental Values ---")
                for raw_file in jseq_raw_files:
                    try:
                        parts = raw_file.stem.split('_')
                        lambda_str = parts[1].replace('lambda', '')
                        lambda_val = float(lambda_str)
                        
                        raw_df = pd.read_csv(raw_file)
                        raw_df.columns = raw_df.columns.str.strip()
                        
                        latencies_ms = raw_df['total_latency_ms']
                        avg_latency_ms = latencies_ms.mean()
                        std_latency_ms = latencies_ms.std()
                        
                        avg_latency_s = avg_latency_ms / 1000.0
                        std_latency_s = std_latency_ms / 1000.0
                        
                        print(f"    - Lambda: {lambda_val:<4} -> Avg: {avg_latency_s:.4f}s, Std: {std_latency_s:.4f}s")

                        detailed_results.append({
                            'Policy': jseq_policy_name,
                            'Lambda': lambda_val,
                            'Total Latency (s)': avg_latency_s,
                            'Total Latency std (s)': std_latency_s
                        })
                        jseq_data_added = True
                    except (ValueError, IndexError, KeyError) as e:
                        print(f"  [ERROR]: Could not parse or process {raw_file.name}: {e}")
                print("  --------------------------------------------")

                if jseq_data_added and jseq_policy_name not in policies_to_test:
                    policies_to_test.append(jseq_policy_name)
                    print(f"\nAppended '{jseq_policy_name}' to policies_to_test.")
    
    try:
        print("\nAttempting to plot results...")
        plot_policy_comparison(detailed_results, policies_to_test, dataset_name_title, show_x_axis=show_x_axis)
        print("Plot generated successfully.")
    except Exception as e:
        print(f"[ERROR]: An unexpected error occurred during plotting: {e}")
        print("   Please check the structure of 'detailed_results' and 'policies_to_test'.")


def run_multi_run_analysis(
    file_list: List[Union[str, Path]], 
    show_scatter: bool = True, 
    show_evolution: bool = True
) -> None:
    """
    Analyzes and plots total error deviations across multple experiment runs.
    """
    if not file_list:
        print("Error: The file list is empty.")
        return

    all_data = []
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}, skipping.")
            continue
        
        df = pd.read_csv(file_path)
        df = df.sort_values('dataset_item_id')

        df['pred_inf'] = df.apply(lambda x: x['jseq_predicted_cloud_inference_time'] if x['route'] == 'cloud' 
                                  else x['jseq_predicted_device_inference_time'], axis=1)
        df['pred_total'] = df.apply(lambda x: x['jseq_predicted_cloud_total_time'] if x['route'] == 'cloud' 
                                    else x['jseq_predicted_device_total_time'], axis=1)
        
        df['abs_inf_err'] = (df['inference_time_ms'] - df['pred_inf']).abs()
        df['abs_total_err'] = (df['total_latency_ms'] - df['pred_total']).abs()
        
        all_data.append(df[['dataset_item_id', 'abs_inf_err', 'abs_total_err', 'total_latency_ms', 'pred_total', 'route']].copy())

    if not all_data:
        print("Error: No valid files were processed.")
        return      

    combined_df = pd.concat(all_data)
    
    agg_df = combined_df.groupby('dataset_item_id')[['abs_inf_err', 'abs_total_err']].agg(['mean', 'std']).reset_index()
    agg_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_df.columns.values]
    
    print(f"\nAnalyzing {len(file_list)} runs...")

    if show_scatter:
        first_run_df = all_data[0]
        plt.figure(figsize=(10, 6))
        for route in ['cloud', 'device']:
            subset = first_run_df[first_run_df['route'] == route]
            if not subset.empty:
                plt.scatter(subset['pred_total'], subset['total_latency_ms'], label=f'{route} (Total)', alpha=0.5)
        
        limit = max(first_run_df['total_latency_ms'].max(), first_run_df['pred_total'].max())
        plt.plot([0, limit], [0, limit], 'r--', label='Ideal')
        plt.title('Total Latency: Predicted vs Actual (Example from first run)')
        plt.xlabel('Predicted Total (ms)')
        plt.ylabel('Actual Total (ms)')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.15)
        plt.show()

    if show_evolution:
        plt.figure(figsize=(8, 3.8))
        
        plt.plot(agg_df['dataset_item_id'], agg_df['abs_total_err_mean'], color='darkslategray', lw=1, label='Mean Inference + Queueing Time Prediction Error')
        plt.fill_between(
            agg_df['dataset_item_id'],
            agg_df['abs_total_err_mean'] - agg_df['abs_total_err_std'],
            agg_df['abs_total_err_mean'] + agg_df['abs_total_err_std'],
            color='darkslategray', alpha=0.05, label='Std. Dev. Inference + Queueing Time Prediction Error'
        )
        
        plt.plot(agg_df['dataset_item_id'], agg_df['abs_inf_err_mean'], color='darkviolet', lw=1, label='Mean Inference Time Prediction Error')
        plt.fill_between(
            agg_df['dataset_item_id'],
            agg_df['abs_inf_err_mean'] - agg_df['abs_inf_err_std'],
            agg_df['abs_inf_err_mean'] + agg_df['abs_inf_err_std'],
            color='darkviolet', alpha=0.05, label='Std. Dev. Inference Time Prediction Error'
        )
        
        plt.yscale('log') 
        plt.xlabel('Dataset Item ID')
        plt.ylabel('Absolute Error (ms) [Log Scale]')
        plt.legend()
        
        out_dir = Path('./plots')
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        filename = out_dir / f'time_prediction_error_{timestamp}.pdf'

        plt.savefig(filename, format='pdf', dpi=300)
        print(f"Saved plot: {filename}")

        plt.show()

    final_mae_total = combined_df['abs_total_err'].mean()
    final_mae_inf = combined_df['abs_inf_err'].mean()
    print(f"Overall MAE Inference (across all runs): {final_mae_inf:.2f} ms")
    print(f"Overall MAE Total (across all runs):     {final_mae_total:.2f} ms")