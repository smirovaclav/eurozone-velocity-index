"""
Real-time Nowcasting Engine

This module implements the nowcasting model for Eurozone consumer demand using
Kalman filtering and state-space models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import linalg
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.tools import (
    is_invertible, constrain_stationary_univariate
)


class NowcastingEngine:
    """
    Real-time nowcasting engine for Eurozone consumer demand.
    
    Uses a state-space model with Kalman filtering to combine:
    - TIPS transaction data (high-frequency)
    - Divisia monetary aggregates (lower-frequency)
    - Historical consumer demand patterns
    """
    
    def __init__(self, 
                 state_dim: int = 3,
                 obs_dim: Optional[int] = None):
        """
        Initialize the nowcasting engine.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of the hidden state (default: 3)
        obs_dim : int, optional
            Dimension of observations (inferred from data if None)
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.model = None
        self.state_estimate = None
        self.state_covariance = None
        self.nowcast_history = []
        
    def initialize_model(self, observation_data: pd.DataFrame) -> None:
        """
        Initialize the state-space model.
        
        Parameters:
        -----------
        observation_data : pd.DataFrame
            Initial observation data for model initialization
        """
        if self.obs_dim is None:
            self.obs_dim = observation_data.shape[1]
        
        # Initialize state vector (latent consumer demand component)
        self.state_estimate = np.zeros(self.state_dim)
        
        # Initialize state covariance
        self.state_covariance = np.eye(self.state_dim)
        
        # State transition matrix (AR(1) model for latent demand)
        self.F = np.array([
            [0.9, 0.0, 0.0],   # Persistence in demand level
            [0.0, 0.7, 0.0],   # Persistence in demand trend
            [0.0, 0.0, 0.5]    # Persistence in demand volatility
        ])
        
        # Observation matrix (how observations relate to state)
        self.H = np.random.randn(self.obs_dim, self.state_dim) * 0.1
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * 0.01
        
        # Measurement noise covariance
        self.R = np.eye(self.obs_dim) * 0.1
        
    def kalman_predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman filter prediction step.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Predicted state and covariance
        """
        # Predict state: x_pred = F * x
        state_pred = self.F @ self.state_estimate
        
        # Predict covariance: P_pred = F * P * F' + Q
        cov_pred = self.F @ self.state_covariance @ self.F.T + self.Q
        
        return state_pred, cov_pred
    
    def kalman_update(self, 
                      observation: np.ndarray,
                      state_pred: np.ndarray,
                      cov_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman filter update step.
        
        Parameters:
        -----------
        observation : np.ndarray
            New observation vector
        state_pred : np.ndarray
            Predicted state from prediction step
        cov_pred : np.ndarray
            Predicted covariance from prediction step
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Updated state and covariance
        """
        # Innovation: y = z - H * x_pred
        innovation = observation - self.H @ state_pred
        
        # Innovation covariance: S = H * P_pred * H' + R
        S = self.H @ cov_pred @ self.H.T + self.R
        
        # Kalman gain: K = P_pred * H' * S^(-1)
        K = cov_pred @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x_pred + K * innovation
        state_updated = state_pred + K @ innovation
        
        # Update covariance: P = (I - K * H) * P_pred
        I = np.eye(self.state_dim)
        cov_updated = (I - K @ self.H) @ cov_pred
        
        return state_updated, cov_updated
    
    def process_observation(self, observation: np.ndarray) -> Dict:
        """
        Process a new observation and update the nowcast.
        
        Parameters:
        -----------
        observation : np.ndarray
            New observation vector
            
        Returns:
        --------
        Dict
            Dictionary with nowcast estimate and confidence
        """
        # Prediction step
        state_pred, cov_pred = self.kalman_predict()
        
        # Update step
        self.state_estimate, self.state_covariance = self.kalman_update(
            observation, state_pred, cov_pred
        )
        
        # Extract consumer demand estimate (first state component)
        demand_estimate = self.state_estimate[0]
        demand_variance = self.state_covariance[0, 0]
        demand_std = np.sqrt(demand_variance)
        
        # Calculate confidence interval (95%)
        confidence_lower = demand_estimate - 1.96 * demand_std
        confidence_upper = demand_estimate + 1.96 * demand_std
        
        nowcast = {
            'estimate': demand_estimate,
            'std': demand_std,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'trend': self.state_estimate[1],
            'volatility': self.state_estimate[2]
        }
        
        self.nowcast_history.append(nowcast)
        
        return nowcast
    
    def fit(self, 
            tips_features: pd.DataFrame,
            divisia_features: pd.DataFrame,
            target: Optional[pd.Series] = None) -> None:
        """
        Fit the nowcasting model to historical data.
        
        Parameters:
        -----------
        tips_features : pd.DataFrame
            Features from TIPS transaction data
        divisia_features : pd.DataFrame
            Features from Divisia aggregates
        target : pd.Series, optional
            Target consumer demand series (if available)
        """
        # Combine features
        features = pd.concat([tips_features, divisia_features], axis=1)
        features = features.fillna(method='ffill').fillna(0)
        
        # Initialize model
        self.initialize_model(features)
        
        # If target is available, optimize model parameters
        if target is not None:
            self._optimize_parameters(features, target)
        
        # Process all historical observations
        for idx in range(len(features)):
            obs = features.iloc[idx].values
            self.process_observation(obs)
    
    def _optimize_parameters(self, 
                            features: pd.DataFrame,
                            target: pd.Series) -> None:
        """
        Optimize model parameters using historical target data.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
        target : pd.Series
            Target consumer demand series
        """
        # Simple parameter optimization using least squares
        # Optimize observation matrix H to minimize prediction error
        
        # Align features and target
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx].values
        y = target.loc[common_idx].values
        
        if len(y) > self.state_dim:
            # Use first state_dim features for simple optimization
            X_reduced = X[:, :min(self.state_dim, X.shape[1])]
            
            # Pad if needed
            if X_reduced.shape[1] < self.state_dim:
                padding = np.zeros((X_reduced.shape[0], 
                                   self.state_dim - X_reduced.shape[1]))
                X_reduced = np.hstack([X_reduced, padding])
            
            # Estimate observation matrix
            try:
                H_opt = np.linalg.lstsq(X_reduced, y, rcond=None)[0]
                self.H[0, :] = H_opt  # Update first row for demand component
            except:
                pass  # Keep initialized values if optimization fails
    
    def predict(self, 
                tips_features: pd.DataFrame,
                divisia_features: pd.DataFrame,
                steps_ahead: int = 1) -> pd.DataFrame:
        """
        Generate nowcast predictions.
        
        Parameters:
        -----------
        tips_features : pd.DataFrame
            Current TIPS features
        divisia_features : pd.DataFrame
            Current Divisia features
        steps_ahead : int
            Number of steps to predict ahead
            
        Returns:
        --------
        pd.DataFrame
            Predictions with confidence intervals
        """
        # Combine current features
        current_features = pd.concat([tips_features, divisia_features], axis=1)
        current_features = current_features.fillna(method='ffill').fillna(0)
        
        predictions = []
        
        # Process latest observation
        if len(current_features) > 0:
            latest_obs = current_features.iloc[-1].values
            nowcast = self.process_observation(latest_obs)
            
            predictions.append({
                'step': 0,
                'estimate': nowcast['estimate'],
                'lower': nowcast['confidence_lower'],
                'upper': nowcast['confidence_upper']
            })
        
        # Predict multiple steps ahead
        for step in range(1, steps_ahead):
            state_pred, cov_pred = self.kalman_predict()
            
            # Store prediction
            demand_pred = state_pred[0]
            demand_var = cov_pred[0, 0]
            demand_std = np.sqrt(demand_var)
            
            predictions.append({
                'step': step,
                'estimate': demand_pred,
                'lower': demand_pred - 1.96 * demand_std,
                'upper': demand_pred + 1.96 * demand_std
            })
            
            # Update for next iteration
            self.state_estimate = state_pred
            self.state_covariance = cov_pred
        
        return pd.DataFrame(predictions)
    
    def get_nowcast_summary(self) -> Dict:
        """
        Get summary of current nowcast state.
        
        Returns:
        --------
        Dict
            Summary statistics and current estimates
        """
        if not self.nowcast_history:
            return {'status': 'No nowcasts generated yet'}
        
        latest = self.nowcast_history[-1]
        
        # Calculate recent trend
        if len(self.nowcast_history) >= 3:
            recent_estimates = [n['estimate'] for n in self.nowcast_history[-3:]]
            recent_trend = np.diff(recent_estimates).mean()
        else:
            recent_trend = 0.0
        
        return {
            'current_estimate': latest['estimate'],
            'current_std': latest['std'],
            'confidence_interval': (latest['confidence_lower'], 
                                   latest['confidence_upper']),
            'trend_component': latest['trend'],
            'volatility_component': latest['volatility'],
            'recent_trend': recent_trend,
            'n_observations': len(self.nowcast_history)
        }
