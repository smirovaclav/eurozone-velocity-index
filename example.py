"""
Example usage of the Eurozone Velocity Index nowcasting system.

This script demonstrates how to:
1. Load and process TIPS transaction data
2. Calculate Divisia monetary aggregates
3. Generate real-time nowcasts of consumer demand
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from eurozone_velocity import NowcastingEngine, TIPSProcessor, DivisiaAggregates
from eurozone_velocity.visualization.plots import (
    plot_nowcast_results, plot_component_analysis, create_dashboard
)


def generate_sample_tips_data(n_periods: int = 1000) -> pd.DataFrame:
    """
    Generate sample TIPS transaction data for demonstration.
    
    Parameters:
    -----------
    n_periods : int
        Number of time periods to generate
        
    Returns:
    --------
    pd.DataFrame
        Sample TIPS transaction data
    """
    # Generate timestamps (hourly data)
    start_date = datetime.now() - timedelta(hours=n_periods)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_periods)]
    
    # Generate transaction data with realistic patterns
    np.random.seed(42)
    base_volume = 10000
    
    # Add daily seasonality
    hours = np.array([t.hour for t in timestamps])
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * hours / 24)
    
    # Add weekly pattern
    weekdays = np.array([t.weekday() for t in timestamps])
    weekly_factor = np.where(weekdays < 5, 1.0, 0.7)  # Lower on weekends
    
    # Generate volumes
    trend = np.linspace(1.0, 1.2, n_periods)  # Upward trend
    noise = np.random.normal(1, 0.1, n_periods)
    volumes = base_volume * trend * seasonal_factor * weekly_factor * noise
    
    # Generate transaction amounts
    amounts = volumes * np.random.uniform(50, 200, n_periods)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'volume': volumes.astype(int),
        'amount': amounts
    })
    
    return data


def generate_sample_divisia_data(n_periods: int = 100) -> pd.DataFrame:
    """
    Generate sample Divisia monetary components data.
    
    Parameters:
    -----------
    n_periods : int
        Number of time periods (months)
        
    Returns:
    --------
    pd.DataFrame
        Sample monetary components data
    """
    start_date = datetime.now() - timedelta(days=30*n_periods)
    dates = [start_date + timedelta(days=30*i) for i in range(n_periods)]
    
    np.random.seed(42)
    
    components = ['M0', 'M1', 'M2', 'M3']
    base_quantities = [1000, 5000, 10000, 15000]
    base_rates = [0.0, 0.01, 0.015, 0.02]
    
    data = []
    for date in dates:
        for comp, base_qty, base_rate in zip(components, base_quantities, base_rates):
            # Add growth trend
            growth = np.random.normal(1.002, 0.01)
            quantity = base_qty * growth ** len(data)
            
            # Rate varies around base
            rate = max(0, base_rate + np.random.normal(0, 0.003))
            
            data.append({
                'date': date,
                'component_name': comp,
                'quantity': quantity,
                'interest_rate': rate
            })
    
    return pd.DataFrame(data)


def main():
    """
    Main example demonstrating the nowcasting system.
    """
    print("=" * 70)
    print("Eurozone Consumer Demand Nowcasting System - Example")
    print("=" * 70)
    print()
    
    # Step 1: Generate sample data
    print("Step 1: Generating sample data...")
    tips_data = generate_sample_tips_data(n_periods=1000)
    divisia_data = generate_sample_divisia_data(n_periods=100)
    print(f"  Generated {len(tips_data)} TIPS transactions")
    print(f"  Generated {len(divisia_data)} Divisia data points")
    print()
    
    # Step 2: Process TIPS data
    print("Step 2: Processing TIPS transaction data...")
    tips_processor = TIPSProcessor(aggregation_window='1h')
    tips_processor.load_transaction_data(tips_data)
    tips_indicators = tips_processor.extract_consumer_indicators()
    tips_features = tips_processor.get_nowcast_features()
    print(f"  Extracted {len(tips_indicators.columns)} consumer indicators")
    print(f"  Generated {len(tips_features.columns)} features")
    print()
    
    # Step 3: Calculate Divisia aggregates
    print("Step 3: Calculating Divisia monetary aggregates...")
    divisia = DivisiaAggregates()
    divisia.load_monetary_data(divisia_data)
    divisia_index = divisia.build_divisia_index()
    divisia_msi = divisia.get_monetary_services_index()
    print(f"  Built Divisia index with {len(divisia_index)} periods")
    print()
    
    # Step 4: Initialize nowcasting engine
    print("Step 4: Initializing nowcasting engine...")
    engine = NowcastingEngine(state_dim=3)
    
    # Prepare features - align time series
    # For this example, we'll use recent TIPS data
    recent_tips = tips_features.tail(100)
    
    # Create matching divisia features (repeated to match TIPS frequency)
    divisia_features = pd.DataFrame({
        'divisia_level': divisia_msi['divisia_index'].iloc[-1],
        'divisia_growth': divisia_msi['msi_growth'].iloc[-1]
    }, index=recent_tips.index)
    
    # Fit the model
    print("  Fitting nowcasting model...")
    engine.fit(recent_tips, divisia_features)
    print(f"  Processed {len(engine.nowcast_history)} observations")
    print()
    
    # Step 5: Generate nowcast
    print("Step 5: Generating real-time nowcast...")
    latest_tips = tips_features.tail(10)
    latest_divisia = divisia_features.tail(10)
    
    predictions = engine.predict(latest_tips, latest_divisia, steps_ahead=24)
    print(f"  Generated {len(predictions)} forecast steps")
    print()
    
    # Step 6: Get nowcast summary
    print("Step 6: Nowcast Summary")
    print("-" * 70)
    summary = engine.get_nowcast_summary()
    print(f"  Current Estimate: {summary['current_estimate']:.4f}")
    print(f"  Standard Deviation: {summary['current_std']:.4f}")
    print(f"  95% Confidence Interval: "
          f"({summary['confidence_interval'][0]:.4f}, "
          f"{summary['confidence_interval'][1]:.4f})")
    print(f"  Trend Component: {summary['trend_component']:.4f}")
    print(f"  Volatility Component: {summary['volatility_component']:.4f}")
    print(f"  Recent Trend: {summary['recent_trend']:.4f}")
    print()
    
    # Step 7: Visualize results
    print("Step 7: Creating visualizations...")
    
    # Convert nowcast history to DataFrame
    if len(engine.nowcast_history) > 0:
        nowcast_df = pd.DataFrame(engine.nowcast_history[:len(recent_tips)])
        nowcast_df.index = recent_tips.index[:len(nowcast_df)]
    
    # Create plots
    try:
        import matplotlib.pyplot as plt
        
        # Plot nowcast results
        fig1 = plot_nowcast_results(nowcast_df)
        plt.savefig('nowcast_results.png', dpi=300, bbox_inches='tight')
        print("  Saved: nowcast_results.png")
        
        # Plot component analysis
        fig2 = plot_component_analysis(tips_indicators.tail(200), 
                                       divisia_msi)
        plt.savefig('component_analysis.png', dpi=300, bbox_inches='tight')
        print("  Saved: component_analysis.png")
        
        # Create dashboard
        fig3 = create_dashboard(summary)
        plt.savefig('nowcast_dashboard.png', dpi=300, bbox_inches='tight')
        print("  Saved: nowcast_dashboard.png")
        
        print()
        print("Visualization files created successfully!")
        
    except Exception as e:
        print(f"  Note: Could not create visualizations: {e}")
    
    print()
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
