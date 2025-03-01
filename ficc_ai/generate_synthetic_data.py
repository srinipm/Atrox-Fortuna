"""
Synthetic Data Generator for FICC Spread Analysis

This script generates synthetic spread data for corporate bonds and benchmarks
for testing and demonstration purposes.
"""

import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime


def generate_synthetic_data(start_date='2020-01-01', 
                           end_date='2023-01-01', 
                           freq='B',
                           corp_base=150,
                           bench_base=100,
                           trend_strength=50,
                           seasonality=20,
                           corp_volatility=15,
                           bench_volatility=8,
                           anomaly_count=20,
                           anomaly_strength=50,
                           output_dir=None):
    """
    Generate synthetic corporate and benchmark spread data.
    
    Parameters:
    -----------
    start_date : str
        Start date for the data (YYYY-MM-DD)
    end_date : str
        End date for the data (YYYY-MM-DD)
    freq : str
        Frequency of data points ('B' for business days, 'D' for daily)
    corp_base : float
        Base level for corporate spreads (in bps)
    bench_base : float
        Base level for benchmark spreads (in bps)
    trend_strength : float
        Strength of the upward trend in spreads
    seasonality : float
        Amplitude of seasonal patterns
    corp_volatility : float
        Volatility (standard deviation) of corporate spread noise
    bench_volatility : float
        Volatility (standard deviation) of benchmark spread noise
    anomaly_count : int
        Number of anomalies to introduce
    anomaly_strength : float
        Strength (standard deviation) of anomalies
    output_dir : str
        Directory to save the output files (if None, current directory is used)
    
    Returns:
    --------
    tuple
        Paths to the generated corporate and benchmark CSV files
    """
    # Set up output directory
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    corporate_file = os.path.join(output_dir, "corporate_spreads.csv")
    benchmark_file = os.path.join(output_dir, "benchmark_spreads.csv")
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    num_points = len(date_range)
    
    # Create trend, seasonality, and noise components
    trend = np.linspace(0, trend_strength, num_points)
    
    # Create multiple seasonal patterns of different frequencies
    seasonality_annual = seasonality * np.sin(np.linspace(0, 2*np.pi, 252))  # Annual cycle (252 business days)
    seasonality_pattern = np.tile(seasonality_annual, int(np.ceil(num_points/252)))[:num_points]
    
    # Add quarterly pattern
    seasonality_quarterly = seasonality/2 * np.sin(np.linspace(0, 8*np.pi, 252))  # 4x faster
    seasonality_pattern += np.tile(seasonality_quarterly, int(np.ceil(num_points/252)))[:num_points]
    
    # Corporate spread components
    np.random.seed(42)  # For reproducibility
    corp_noise = np.random.normal(0, corp_volatility, num_points)
    
    # Add some anomalies
    corp_anomalies = np.zeros(num_points)
    anomaly_idx = np.random.choice(num_points, anomaly_count, replace=False)
    corp_anomalies[anomaly_idx] = np.random.normal(0, anomaly_strength, len(anomaly_idx))
    
    # Combine components for corporate spread
    corp_spread = corp_base + trend + seasonality_pattern + corp_noise + corp_anomalies
    corp_spread = np.maximum(corp_spread, 10)  # Ensure no negative spreads
    
    # Benchmark spread components (correlated with corporate but less volatile)
    bench_noise = np.random.normal(0, bench_volatility, num_points)
    bench_trend = 0.7 * trend  # Similar but smaller trend
    bench_seasonality = 0.8 * seasonality_pattern  # Similar but smaller seasonal pattern
    
    # Combine components for benchmark spread
    bench_spread = bench_base + bench_trend + bench_seasonality + bench_noise
    bench_spread = np.maximum(bench_spread, 5)  # Ensure no negative spreads
    
    # Create DataFrames
    corp_df = pd.DataFrame({
        'date': date_range,
        'spread_corp': corp_spread
    })
    
    bench_df = pd.DataFrame({
        'date': date_range,
        'spread_benchmark': bench_spread
    })
    
    # Add some realistic market regime changes
    # Find a random segment to introduce a regime change
    if num_points > 100:  # Only if we have enough data points
        regime_start = np.random.randint(50, num_points - 50)
        regime_length = np.random.randint(30, 100)
        regime_end = min(regime_start + regime_length, num_points)
        
        # During this regime, increase correlation and volatility
        regime_shift = np.random.normal(30, 10, regime_end - regime_start)  # Random shift
        corp_df.loc[regime_start:regime_end, 'spread_corp'] += regime_shift
        bench_df.loc[regime_start:regime_end, 'spread_benchmark'] += regime_shift * 0.9  # Highly correlated
    
    # Save to CSV
    corp_df.to_csv(corporate_file, index=False)
    bench_df.to_csv(benchmark_file, index=False)
    
    print(f"Generated synthetic data:")
    print(f"- Corporate spreads: {corporate_file}")
    print(f"- Benchmark spreads: {benchmark_file}")
    
    return corporate_file, benchmark_file


def create_spread_visualization(corp_file, bench_file, output_dir=None):
    """
    Create a quick visualization of the generated spread data.
    
    Parameters:
    -----------
    corp_file : str
        Path to corporate spread file
    bench_file : str
        Path to benchmark spread file
    output_dir : str, optional
        Directory to save the visualization
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set visualization style
        sns.set(style='darkgrid')
        
        # Load data
        corp_data = pd.read_csv(corp_file, parse_dates=['date'])
        bench_data = pd.read_csv(bench_file, parse_dates=['date'])
        
        # Merge datasets
        merged_data = pd.merge(corp_data, bench_data, on='date')
        merged_data['spread_diff'] = merged_data['spread_corp'] - merged_data['spread_benchmark']
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot spreads
        plt.subplot(2, 1, 1)
        plt.plot(merged_data['date'], merged_data['spread_corp'], 'b-', label='Corporate Spread')
        plt.plot(merged_data['date'], merged_data['spread_benchmark'], 'r-', label='Benchmark Spread')
        plt.fill_between(merged_data['date'], merged_data['spread_corp'], 
                        merged_data['spread_benchmark'], alpha=0.3, color='green')
        plt.title('Synthetic Spread Data')
        plt.legend()
        
        # Plot spread difference
        plt.subplot(2, 1, 2)
        plt.plot(merged_data['date'], merged_data['spread_diff'], 'g-')
        plt.title('Spread Difference')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        plt.tight_layout()
        
        # Save if output_dir provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'synthetic_spread_data.png'), dpi=300)
            print(f"Visualization saved to {os.path.join(output_dir, 'synthetic_spread_data.png')}")
        
        # Display plot
        plt.show()
        
    except ImportError:
        print("Matplotlib or seaborn not found. Visualization skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic FICC spread data for testing')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--frequency', type=str, default='B', help='Data frequency: B for business days, D for calendar days')
    parser.add_argument('--corp-base', type=float, default=150, help='Base level for corporate spreads (bps)')
    parser.add_argument('--bench-base', type=float, default=100, help='Base level for benchmark spreads (bps)')
    parser.add_argument('--trend', type=float, default=50, help='Strength of trend component')
    parser.add_argument('--seasonality', type=float, default=20, help='Amplitude of seasonal patterns')
    parser.add_argument('--corp-vol', type=float, default=15, help='Corporate spread volatility')
    parser.add_argument('--bench-vol', type=float, default=8, help='Benchmark spread volatility')
    parser.add_argument('--anomalies', type=int, default=20, help='Number of anomalies to introduce')
    parser.add_argument('--anomaly-strength', type=float, default=50, help='Strength of anomalies')
    parser.add_argument('--output-dir', type=str, default='./data', help='Output directory for CSV files')
    parser.add_argument('--visualize', action='store_true', help='Create visualization of generated data')
    
    args = parser.parse_args()
    
    print("Generating synthetic spread data with the following parameters:")
    print(f"- Date range: {args.start_date} to {args.end_date} ({args.frequency} frequency)")
    print(f"- Corporate base: {args.corp_base} bps, Benchmark base: {args.bench_base} bps")
    print(f"- Trend strength: {args.trend}, Seasonality: {args.seasonality}")
    print(f"- Volatility: Corp {args.corp_vol}, Bench {args.bench_vol}")
    print(f"- Anomalies: {args.anomalies} with strength {args.anomaly_strength}")
    print(f"- Output directory: {args.output_dir}")
    
    # Generate data
    corp_file, bench_file = generate_synthetic_data(
        start_date=args.start_date,
        end_date=args.end_date,
        freq=args.frequency,
        corp_base=args.corp_base,
        bench_base=args.bench_base,
        trend_strength=args.trend,
        seasonality=args.seasonality,
        corp_volatility=args.corp_vol,
        bench_volatility=args.bench_vol,
        anomaly_count=args.anomalies,
        anomaly_strength=args.anomaly_strength,
        output_dir=args.output_dir
    )
    
    # Create visualization if requested
    if args.visualize:
        print("Creating visualization...")
        create_spread_visualization(corp_file, bench_file, args.output_dir)