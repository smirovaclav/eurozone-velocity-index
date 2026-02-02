# Eurozone Velocity Index

Real-time nowcasting of Eurozone consumer demand using TIPS transaction data and Divisia monetary aggregates.

## Overview

This system provides real-time nowcasting capabilities for Eurozone consumer demand by combining:
- **TIPS Transaction Data**: High-frequency payment data from the TARGET Instant Payment Settlement system
- **Divisia Monetary Aggregates**: Sophisticated weighted measures of money supply that account for varying liquidity
- **Kalman Filtering**: State-space models for real-time estimation with uncertainty quantification

## Features

- **Real-time Processing**: Process TIPS transaction data at high frequency (hourly/daily)
- **Divisia Aggregates**: Calculate sophisticated monetary aggregates that weight components by liquidity
- **Nowcasting Engine**: State-space model with Kalman filtering for real-time consumer demand estimation
- **Uncertainty Quantification**: Confidence intervals and volatility measures for all estimates
- **Visualization Tools**: Comprehensive plotting and dashboard capabilities

## Installation

```bash
# Clone the repository
git clone https://github.com/smirovaclav/eurozone-velocity-index.git
cd eurozone-velocity-index

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
from eurozone_velocity import NowcastingEngine, TIPSProcessor, DivisiaAggregates

# 1. Process TIPS transaction data
tips_processor = TIPSProcessor(aggregation_window='1H')
tips_processor.load_transaction_data('tips_data.csv')
tips_features = tips_processor.get_nowcast_features()

# 2. Calculate Divisia aggregates
divisia = DivisiaAggregates()
divisia.load_monetary_data('monetary_data.csv')
divisia_index = divisia.build_divisia_index()
divisia_features = divisia.get_monetary_services_index()

# 3. Generate nowcast
engine = NowcastingEngine(state_dim=3)
engine.fit(tips_features, divisia_features)
predictions = engine.predict(tips_features, divisia_features, steps_ahead=24)

# 4. Get summary
summary = engine.get_nowcast_summary()
print(f"Current Demand Estimate: {summary['current_estimate']:.4f}")
```

## Components

### TIPS Transaction Processor

Processes real-time payment data from the TIPS system to extract consumer spending indicators:

- Transaction volume and value metrics
- Spending intensity indices
- Transaction velocity and momentum
- Time-series features for modeling

### Divisia Monetary Aggregates

Implements Divisia monetary aggregates that provide superior measurement of monetary services:

- User cost calculation for each monetary component
- Divisia index construction using weighted growth rates
- Monetary services index (MSI) with momentum indicators
- Velocity of money calculations

### Nowcasting Engine

State-space model with Kalman filtering for real-time estimation:

- **State Vector**: Latent consumer demand components (level, trend, volatility)
- **Kalman Filter**: Optimal real-time state estimation
- **Prediction**: Multi-step ahead forecasts with confidence intervals
- **Adaptivity**: Model updates with each new observation

## Data Requirements

### TIPS Transaction Data

CSV file or DataFrame with columns:
- `timestamp`: Transaction timestamp (datetime)
- `amount`: Transaction value (float)
- Additional fields: customer_type, merchant_category, etc.

### Monetary Components Data

CSV file or DataFrame with columns:
- `date`: Observation date (datetime)
- `component_name`: Name of monetary component (e.g., M0, M1, M2, M3)
- `quantity`: Amount of the component (float)
- `interest_rate`: Interest rate of the component (float)

## Methodology

### Nowcasting Approach

The system uses a state-space framework:

1. **State Equation**: x_t = F * x_{t-1} + w_t
   - Captures latent consumer demand dynamics
   
2. **Observation Equation**: z_t = H * x_t + v_t
   - Links observations (TIPS + Divisia) to latent state

3. **Kalman Filter**: Provides optimal real-time estimates
   - Prediction step: Forward propagation
   - Update step: Incorporation of new observations

### Divisia Aggregates

User cost formula: π_i = (R - r_i) / (1 + R)
- R: Benchmark rate (maximum among components)
- r_i: Component interest rate

Divisia growth: D_t = Σ s_{i,t} * Δln(M_{i,t})
- s_{i,t}: Expenditure share of component i
- M_{i,t}: Quantity of component i

## Example

Run the included example:

```bash
python example.py
```

This demonstrates:
- Sample data generation
- TIPS processing pipeline
- Divisia calculation
- Nowcast generation
- Visualization creation

## Visualization

The system includes comprehensive visualization tools:

```python
from eurozone_velocity.visualization.plots import (
    plot_nowcast_results,
    plot_component_analysis,
    create_dashboard
)

# Plot nowcast with confidence intervals
fig = plot_nowcast_results(nowcast_df, actual=actual_values)

# Analyze TIPS and Divisia components
fig = plot_component_analysis(tips_indicators, divisia_index)

# Create comprehensive dashboard
fig = create_dashboard(engine.get_nowcast_summary())
```

## Advanced Usage

### Custom State-Space Model

```python
# Initialize with custom dimensions
engine = NowcastingEngine(state_dim=5, obs_dim=20)

# Fit with target data for parameter optimization
engine.fit(tips_features, divisia_features, target=consumer_demand_series)
```

### Feature Engineering

```python
# Extract custom features from TIPS data
tips_processor = TIPSProcessor(aggregation_window='30min')
indicators = tips_processor.extract_consumer_indicators()

# Access raw transaction metrics
metrics = tips_processor.calculate_transaction_metrics()
```

### Velocity Analysis

```python
# Calculate velocity of money
velocity = divisia.calculate_velocity(gdp_data)

# Visualize velocity trends
from eurozone_velocity.visualization.plots import plot_velocity_analysis
fig = plot_velocity_analysis(velocity)
```

## Technical Details

- **Language**: Python 3.8+
- **Key Dependencies**: NumPy, Pandas, SciPy, statsmodels, matplotlib
- **Model**: Linear Gaussian state-space model with Kalman filter
- **Frequency**: Supports hourly to monthly data
- **Real-time**: Designed for streaming data processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Citation

If you use this system in your research, please cite:

```
Eurozone Velocity Index: Real-time Nowcasting of Consumer Demand
using TIPS Transaction Data and Divisia Monetary Aggregates (2026)
```

## References

- Barnett, W. A. (1980). "Economic Monetary Aggregates: An Application of Index Number and Aggregation Theory"
- Durbin, J., & Koopman, S. J. (2012). "Time Series Analysis by State Space Methods"
- ECB TARGET Services: https://www.ecb.europa.eu/paym/target/tips/html/index.en.html

## Contact

For questions or issues, please open an issue on GitHub.