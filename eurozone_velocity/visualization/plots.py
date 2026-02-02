"""
Visualization tools for nowcasting results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


def plot_nowcast_results(nowcast_df: pd.DataFrame,
                        actual: Optional[pd.Series] = None,
                        title: str = "Consumer Demand Nowcast",
                        figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot nowcast estimates with confidence intervals.
    
    Parameters:
    -----------
    nowcast_df : pd.DataFrame
        DataFrame with columns: estimate, lower, upper
    actual : pd.Series, optional
        Actual values for comparison
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot nowcast estimate
    ax.plot(nowcast_df.index, nowcast_df['estimate'], 
            label='Nowcast', linewidth=2, color='blue')
    
    # Plot confidence interval
    if 'lower' in nowcast_df.columns and 'upper' in nowcast_df.columns:
        ax.fill_between(nowcast_df.index,
                        nowcast_df['lower'],
                        nowcast_df['upper'],
                        alpha=0.3, color='blue', label='95% CI')
    
    # Plot actual values if available
    if actual is not None:
        common_idx = nowcast_df.index.intersection(actual.index)
        ax.plot(common_idx, actual[common_idx],
                label='Actual', linewidth=2, color='red', 
                linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Consumer Demand Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_component_analysis(tips_indicators: pd.DataFrame,
                           divisia_index: pd.DataFrame,
                           figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Plot analysis of TIPS and Divisia components.
    
    Parameters:
    -----------
    tips_indicators : pd.DataFrame
        TIPS indicators
    divisia_index : pd.DataFrame
        Divisia monetary index
    figsize : tuple
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # TIPS volume indicator
    if 'spending_intensity' in tips_indicators.columns:
        axes[0, 0].plot(tips_indicators.index, 
                       tips_indicators['spending_intensity'],
                       color='green', linewidth=1.5)
        axes[0, 0].set_title('TIPS Spending Intensity', fontweight='bold')
        axes[0, 0].set_ylabel('Normalized Volume')
        axes[0, 0].grid(True, alpha=0.3)
    
    # TIPS demand index
    if 'demand_index' in tips_indicators.columns:
        axes[0, 1].plot(tips_indicators.index,
                       tips_indicators['demand_index'],
                       color='orange', linewidth=1.5)
        axes[0, 1].set_title('TIPS Demand Index', fontweight='bold')
        axes[0, 1].set_ylabel('Normalized Value')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Divisia index
    if 'divisia_index' in divisia_index.columns:
        axes[1, 0].plot(divisia_index.index,
                       divisia_index['divisia_index'],
                       color='purple', linewidth=1.5)
        axes[1, 0].set_title('Divisia Monetary Index', fontweight='bold')
        axes[1, 0].set_ylabel('Index Value')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Divisia growth
    if 'msi_growth' in divisia_index.columns:
        axes[1, 1].plot(divisia_index.index,
                       divisia_index['msi_growth'],
                       color='red', linewidth=1.5)
        axes[1, 1].set_title('Divisia Growth Rate', fontweight='bold')
        axes[1, 1].set_ylabel('Growth Rate')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_forecast_performance(predictions: pd.DataFrame,
                             actual: pd.Series,
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot forecast performance metrics.
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        Predictions with columns: estimate, lower, upper
    actual : pd.Series
        Actual values
    figsize : tuple
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Align data
    common_idx = predictions.index.intersection(actual.index)
    pred_values = predictions.loc[common_idx, 'estimate']
    actual_values = actual[common_idx]
    
    # Plot predictions vs actual
    axes[0].plot(common_idx, pred_values, label='Predicted', 
                linewidth=2, color='blue')
    axes[0].plot(common_idx, actual_values, label='Actual',
                linewidth=2, color='red', linestyle='--')
    axes[0].set_ylabel('Value', fontsize=11)
    axes[0].set_title('Nowcast vs Actual', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot forecast errors
    errors = actual_values - pred_values
    axes[1].bar(common_idx, errors, alpha=0.6, color='gray')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1].set_xlabel('Time', fontsize=11)
    axes[1].set_ylabel('Error', fontsize=11)
    axes[1].set_title('Forecast Errors', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Add error statistics
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    axes[1].text(0.02, 0.98, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}',
                transform=axes[1].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_velocity_analysis(velocity: pd.DataFrame,
                          figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot velocity of money analysis.
    
    Parameters:
    -----------
    velocity : pd.DataFrame
        Velocity data
    figsize : tuple
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'velocity' in velocity.columns:
        ax.plot(velocity.index, velocity['velocity'],
               linewidth=2, color='darkblue')
        
        # Add moving average
        ma = velocity['velocity'].rolling(window=12).mean()
        ax.plot(velocity.index, ma,
               linewidth=2, color='red', linestyle='--',
               label='12-period MA', alpha=0.7)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Velocity of Money', fontsize=12)
        ax.set_title('Eurozone Money Velocity (Divisia)', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_dashboard(nowcast_summary: dict,
                    figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Create a comprehensive dashboard of nowcasting results.
    
    Parameters:
    -----------
    nowcast_summary : dict
        Summary dictionary from NowcastingEngine
    figsize : tuple
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Current estimate gauge
    ax1 = fig.add_subplot(gs[0, :])
    ax1.text(0.5, 0.5, f"{nowcast_summary['current_estimate']:.2f}",
            ha='center', va='center', fontsize=48, fontweight='bold')
    ax1.text(0.5, 0.2, "Current Consumer Demand Estimate",
            ha='center', va='center', fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Confidence interval
    ax2 = fig.add_subplot(gs[1, 0])
    ci = nowcast_summary['confidence_interval']
    ax2.barh(['Estimate'], [nowcast_summary['current_estimate']], 
            color='blue', alpha=0.6)
    ax2.errorbar([nowcast_summary['current_estimate']], ['Estimate'],
                xerr=[[nowcast_summary['current_estimate'] - ci[0]],
                      [ci[1] - nowcast_summary['current_estimate']]],
                fmt='none', color='black', capsize=10)
    ax2.set_xlabel('Value')
    ax2.set_title('95% Confidence Interval', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Components
    ax3 = fig.add_subplot(gs[1, 1])
    components = ['Trend', 'Volatility']
    values = [nowcast_summary['trend_component'],
             nowcast_summary['volatility_component']]
    colors = ['green' if v > 0 else 'red' for v in values]
    ax3.barh(components, values, color=colors, alpha=0.6)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_xlabel('Component Value')
    ax3.set_title('State Components', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Statistics
    ax4 = fig.add_subplot(gs[2, :])
    stats_text = f"""
    Recent Trend: {nowcast_summary['recent_trend']:.4f}
    Standard Deviation: {nowcast_summary['current_std']:.4f}
    Number of Observations: {nowcast_summary['n_observations']}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
            family='monospace')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    fig.suptitle('Eurozone Consumer Demand Nowcasting Dashboard',
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig
