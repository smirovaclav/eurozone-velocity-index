"""
Eurozone Velocity Index - Real-time Nowcasting System

This package provides tools for real-time nowcasting of Eurozone consumer demand
using TIPS transaction data and Divisia monetary aggregates.
"""

__version__ = "0.1.0"

from .models.nowcasting import NowcastingEngine
from .data.tips_processor import TIPSProcessor
from .data.divisia_aggregates import DivisiaAggregates

__all__ = [
    "NowcastingEngine",
    "TIPSProcessor", 
    "DivisiaAggregates",
]
