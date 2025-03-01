"""
FICC Spread Analysis AI Solution

This solution provides automated analysis of corporate bond spreads against benchmark spreads,
including visualization, anomaly detection, and forecasting capabilities.

Main components:
1. Data processing
2. Visualization
3. Anomaly detection
4. Notification system
5. Forecasting
6. Main application
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
import logging
from threading import Timer
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("spread_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration parameters
CONFIG = {
    "anomaly_threshold": 2.0,  # Standard deviations from mean for anomaly detection
    "notification_email": "user@example.com",
    "smtp_server": "smtp.example.com",
    "smtp_port": 587,
    "smtp_username": "notification@example.com",
    "smtp_password": "your_password",
    "real_time_interval": 300,  # 5 minutes in seconds
    "historical_data_path": "./data/",
    "model_save_path": "./models/",
    "visualization_path": "./visualizations/",
}

# Create directories if they don't exist
for path in [CONFIG["historical_data_path"], CONFIG["model_save_path"], CONFIG["visualization_path"]]:
    os.makedirs(path, exist_ok=True)


###########################################
# 1. Data Processing Module
###########################################

class DataProcessor:
    """
    Handles data loading, preprocessing, and feature engineering for spread analysis.
    """
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.data = None
        
    def load_data(self, corporate_file, benchmark_file):
        """
        Load corporate and benchmark spread data from CSV files.
        
        Parameters:
        -----------
        corporate_file : str
            Path to corporate bond spread data file
        benchmark_file : str
            Path to benchmark spread data file
            
        Returns:
        --------
        pandas.DataFrame
            Combined dataset with both spreads
        """
        try:
            # Load data
            corp_data = pd.read_csv(corporate_file, parse_dates=['date'])
            benchmark_data = pd.read_csv(benchmark_file, parse_dates=['date'])
            
            # Merge datasets on date
            merged_data = pd.merge(
                corp_data, 
                benchmark_data, 
                on='date', 
                how='inner',
                suffixes=('_corp', '_benchmark')
            )
            
            # Calculate spread difference
            merged_data['spread_diff'] = merged_data['spread_corp'] - merged_data['spread_benchmark']
            
            # Sort by date
            merged_data.sort_values('date', inplace=True)
            
            self.data = merged_data
            logger.info(f"Data loaded successfully. Shape: {merged_data.shape}")
            
            return merged_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, data=None):
        """
        Clean the dataset by handling missing values and outliers.
        
        Parameters:
        -----------
        data : pandas.DataFrame, optional
            Dataset to clean. If None, uses self.data
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned dataset
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data available. Load data first.")
        
        try:
            # Make a copy to avoid modifying the original
            cleaned_data = data.copy()
            
            # Check for missing values
            missing_count = cleaned_data.isnull().sum()
            if missing_count.sum() > 0:
                logger.info(f"Missing values found: {missing_count}")
                
                # Interpolate missing values
                cleaned_data = cleaned_data.interpolate(method='time')
                
                # Fill remaining missing values (if any at the beginning/end)
                cleaned_data = cleaned_data.fillna(method='bfill').fillna(method='ffill')
            
            # Handle extreme outliers
            for col in ['spread_corp', 'spread_benchmark', 'spread_diff']:
                if col in cleaned_data.columns:
                    # Calculate IQR
                    Q1 = cleaned_data[col].quantile(0.25)
                    Q3 = cleaned_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Define outlier bounds (using a more conservative factor than the typical 1.5)
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    # Log outliers but don't remove them (just cap them)
                    outliers = cleaned_data[(cleaned_data[col] < lower_bound) | 
                                           (cleaned_data[col] > upper_bound)]
                    if not outliers.empty:
                        logger.info(f"Found {len(outliers)} outliers in {col}")
                        
                        # Cap outliers instead of removing them
                        cleaned_data.loc[cleaned_data[col] < lower_bound, col] = lower_bound
                        cleaned_data.loc[cleaned_data[col] > upper_bound, col] = upper_bound
            
            self.data = cleaned_data
            logger.info("Data cleaning completed")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
    
    def add_features(self, data=None):
        """
        Add additional features for analysis.
        
        Parameters:
        -----------
        data : pandas.DataFrame, optional
            Dataset to enhance. If None, uses self.data
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with additional features
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data available. Load data first.")
        
        try:
            # Make a copy to avoid modifying the original
            enhanced_data = data.copy()
            
            # Calculate rolling statistics
            for window in [5, 10, 20]:  # 1-week, 2-week, 1-month (assuming business days)
                # Rolling mean
                enhanced_data[f'corp_rolling_mean_{window}d'] = enhanced_data['spread_corp'].rolling(window=window).mean()
                enhanced_data[f'benchmark_rolling_mean_{window}d'] = enhanced_data['spread_benchmark'].rolling(window=window).mean()
                enhanced_data[f'diff_rolling_mean_{window}d'] = enhanced_data['spread_diff'].rolling(window=window).mean()
                
                # Rolling standard deviation
                enhanced_data[f'corp_rolling_std_{window}d'] = enhanced_data['spread_corp'].rolling(window=window).std()
                enhanced_data[f'benchmark_rolling_std_{window}d'] = enhanced_data['spread_benchmark'].rolling(window=window).std()
                enhanced_data[f'diff_rolling_std_{window}d'] = enhanced_data['spread_diff'].rolling(window=window).std()
                
                # Z-score using rolling statistics
                enhanced_data[f'corp_zscore_{window}d'] = (enhanced_data['spread_corp'] - 
                                                           enhanced_data[f'corp_rolling_mean_{window}d']) / \
                                                           enhanced_data[f'corp_rolling_std_{window}d']
                enhanced_data[f'benchmark_zscore_{window}d'] = (enhanced_data['spread_benchmark'] - 
                                                                enhanced_data[f'benchmark_rolling_mean_{window}d']) / \
                                                                enhanced_data[f'benchmark_rolling_std_{window}d']
                enhanced_data[f'diff_zscore_{window}d'] = (enhanced_data['spread_diff'] - 
                                                           enhanced_data[f'diff_rolling_mean_{window}d']) / \
                                                           enhanced_data[f'diff_rolling_std_{window}d']
            
            # Calculate relative movement
            enhanced_data['relative_movement'] = enhanced_data['spread_corp'].pct_change() - \
                                                enhanced_data['spread_benchmark'].pct_change()
            
            # Fill NA values resulting from calculations
            enhanced_data = enhanced_data.fillna(method='bfill').fillna(method='ffill')
            
            self.data = enhanced_data
            logger.info("Feature engineering completed")
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error adding features: {str(e)}")
            raise
            
    def get_latest_data(self, days=30):
        """
        Get the most recent data for real-time analysis.
        
        Parameters:
        -----------
        days : int
            Number of recent days to retrieve
            
        Returns:
        --------
        pandas.DataFrame
            Recent data subset
        """
        if self.data is None:
            raise ValueError("No data available. Load data first.")
        
        try:
            # Get most recent date in the dataset
            latest_date = self.data['date'].max()
            
            # Calculate start date for the recent period
            start_date = latest_date - timedelta(days=days)
            
            # Filter dataset
            recent_data = self.data[self.data['date'] >= start_date]
            
            logger.info(f"Retrieved {len(recent_data)} records from the last {days} days")
            
            return recent_data
            
        except Exception as e:
            logger.error(f"Error retrieving recent data: {str(e)}")
            raise


###########################################
# 2. Visualization Module
###########################################

class VisualizationTool:
    """
    Creates visualizations for spread analysis.
    """
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        # Set up plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def create_spread_trend_chart(self, data, save_path=None):
        """
        Create a trend chart showing corporate and benchmark spreads over time.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset containing spread data
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        try:
            plt.figure(figsize=(14, 8))
            
            # Plot both spreads
            plt.plot(data['date'], data['spread_corp'], 'b-', linewidth=2, label='Corporate Spread')
            plt.plot(data['date'], data['spread_benchmark'], 'r-', linewidth=2, label='Benchmark Spread')
            
            # Add spread difference as a filled area
            plt.fill_between(data['date'], 
                             data['spread_corp'], 
                             data['spread_benchmark'], 
                             alpha=0.3, 
                             color='green' if (data['spread_corp'] > data['spread_benchmark']).all() else 'red',
                             label='Spread Difference')
            
            # Add labels and title
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Spread (bps)', fontsize=12)
            plt.title('Corporate vs Benchmark Spread Trend', fontsize=16)
            plt.legend(loc='best', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved spread trend chart to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error creating spread trend chart: {str(e)}")
            raise
    
    def create_anomaly_chart(self, data, anomalies, save_path=None):
        """
        Create a chart highlighting detected anomalies.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset containing spread data
        anomalies : pandas.DataFrame
            Dataset containing anomaly flags
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        try:
            plt.figure(figsize=(14, 8))
            
            # Plot spread difference
            plt.plot(data['date'], data['spread_diff'], 'b-', linewidth=2, label='Spread Difference')
            
            # Highlight anomalies
            anomaly_dates = anomalies[anomalies['is_anomaly']]['date']
            anomaly_values = anomalies[anomalies['is_anomaly']]['spread_diff']
            plt.scatter(anomaly_dates, anomaly_values, c='red', s=100, label='Anomalies')
            
            # Add threshold lines if available
            if 'upper_threshold' in anomalies.columns and 'lower_threshold' in anomalies.columns:
                plt.plot(anomalies['date'], anomalies['upper_threshold'], 'r--', alpha=0.5, label='Upper Threshold')
                plt.plot(anomalies['date'], anomalies['lower_threshold'], 'r--', alpha=0.5, label='Lower Threshold')
            
            # Add labels and title
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Spread Difference (bps)', fontsize=12)
            plt.title('Spread Difference with Detected Anomalies', fontsize=16)
            plt.legend(loc='best', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved anomaly chart to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error creating anomaly chart: {str(e)}")
            raise
    
    def create_forecast_chart(self, historical_data, forecast_data, save_path=None):
        """
        Create a chart showing historical data and forecasts.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Dataset containing historical spread data
        forecast_data : pandas.DataFrame
            Dataset containing forecast spread data
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        try:
            plt.figure(figsize=(14, 8))
            
            # Plot historical data
            plt.plot(historical_data['date'], historical_data['spread_diff'], 
                    'b-', linewidth=2, label='Historical Spread Difference')
            
            # Plot forecast data
            plt.plot(forecast_data['forecast_date'], forecast_data['forecast_value'], 
                    'r-', linewidth=2, label='Forecasted Spread Difference')
            
            # Add confidence intervals if available
            if 'lower_ci' in forecast_data.columns and 'upper_ci' in forecast_data.columns:
                plt.fill_between(forecast_data['forecast_date'], 
                                forecast_data['lower_ci'], 
                                forecast_data['upper_ci'], 
                                color='r', alpha=0.2, label='95% Confidence Interval')
            
            # Add vertical line separating historical and forecast data
            latest_date = historical_data['date'].max()
            plt.axvline(x=latest_date, color='k', linestyle='--', alpha=0.5, 
                        label='Forecast Start')
            
            # Add labels and title
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Spread Difference (bps)', fontsize=12)
            plt.title('Historical and Forecasted Spread Difference', fontsize=16)
            plt.legend(loc='best', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved forecast chart to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error creating forecast chart: {str(e)}")
            raise
    
    def create_dashboard(self, data, anomalies, forecast_data, save_path=None):
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset containing spread data
        anomalies : pandas.DataFrame
            Dataset containing anomaly flags
        forecast_data : pandas.DataFrame
            Dataset containing forecast spread data
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        try:
            fig = plt.figure(figsize=(20, 15))
            
            # Create 3 subplots
            ax1 = plt.subplot2grid((3, 1), (0, 0))
            ax2 = plt.subplot2grid((3, 1), (1, 0))
            ax3 = plt.subplot2grid((3, 1), (2, 0))
            
            # Plot 1: Spread trends
            ax1.plot(data['date'], data['spread_corp'], 'b-', linewidth=2, label='Corporate Spread')
            ax1.plot(data['date'], data['spread_benchmark'], 'r-', linewidth=2, label='Benchmark Spread')
            ax1.fill_between(data['date'], data['spread_corp'], data['spread_benchmark'], 
                             alpha=0.3, color='green', label='Spread Difference')
            ax1.set_title('Corporate vs Benchmark Spread Trend', fontsize=14)
            ax1.set_ylabel('Spread (bps)', fontsize=12)
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Anomalies
            ax2.plot(data['date'], data['spread_diff'], 'b-', linewidth=2, label='Spread Difference')
            
            # Highlight anomalies
            anomaly_dates = anomalies[anomalies['is_anomaly']]['date']
            anomaly_values = anomalies[anomalies['is_anomaly']]['spread_diff']
            ax2.scatter(anomaly_dates, anomaly_values, c='red', s=100, label='Anomalies')
            
            # Add threshold lines if available
            if 'upper_threshold' in anomalies.columns and 'lower_threshold' in anomalies.columns:
                ax2.plot(anomalies['date'], anomalies['upper_threshold'], 'r--', alpha=0.5, label='Thresholds')
                ax2.plot(anomalies['date'], anomalies['lower_threshold'], 'r--', alpha=0.5)
            
            ax2.set_title('Spread Difference with Detected Anomalies', fontsize=14)
            ax2.set_ylabel('Spread Diff (bps)', fontsize=12)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Forecast
            historical_end = data['date'].max()
            historical_data = data[data['date'] <= historical_end]
            
            ax3.plot(historical_data['date'], historical_data['spread_diff'], 
                    'b-', linewidth=2, label='Historical')
            ax3.plot(forecast_data['forecast_date'], forecast_data['forecast_value'], 
                    'r-', linewidth=2, label='Forecast')
            
            # Add confidence intervals if available
            if 'lower_ci' in forecast_data.columns and 'upper_ci' in forecast_data.columns:
                ax3.fill_between(forecast_data['forecast_date'], 
                                forecast_data['lower_ci'], 
                                forecast_data['upper_ci'], 
                                color='r', alpha=0.2, label='95% CI')
            
            # Add vertical line separating historical and forecast
            ax3.axvline(x=historical_end, color='k', linestyle='--', alpha=0.5, 
                        label='Forecast Start')
            
            ax3.set_title('Historical and Forecasted Spread Difference', fontsize=14)
            ax3.set_xlabel('Date', fontsize=12)
            ax3.set_ylabel('Spread Diff (bps)', fontsize=12)
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
            
            # Format dates and adjust layout
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
            
            plt.tight_layout()
            
            # Add dashboard title
            plt.suptitle('FICC Spread Analysis Dashboard', fontsize=20, y=1.02)
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved dashboard to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise


###########################################
# 3. Anomaly Detection Module
###########################################

class AnomalyDetector:
    """
    Detects anomalies in spread data using various methods.
    """
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.model = None
        self.threshold = self.config.get("anomaly_threshold", 2.0)
        
    def detect_statistical_anomalies(self, data, column='spread_diff', window=20):
        """
        Detect anomalies using statistical methods (z-score approach).
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset containing spread data
        column : str
            Column name to analyze for anomalies
        window : int
            Rolling window size for calculating statistics
            
        Returns:
        --------
        pandas.DataFrame
            Original data with anomaly flags and thresholds
        """
        try:
            # Make a copy to avoid modifying the original
            result = data.copy()
            
            # Calculate rolling mean and standard deviation
            rolling_mean = result[column].rolling(window=window).mean()
            rolling_std = result[column].rolling(window=window).std()
            
            # Define thresholds
            upper_threshold = rolling_mean + (self.threshold * rolling_std)
            lower_threshold = rolling_mean - (self.threshold * rolling_std)
            
            # Identify anomalies
            result['is_anomaly'] = ((result[column] > upper_threshold) | 
                                    (result[column] < lower_threshold))
            
            # Add thresholds for visualization
            result['upper_threshold'] = upper_threshold
            result['lower_threshold'] = lower_threshold
            result['rolling_mean'] = rolling_mean
            
            # Count anomalies
            anomaly_count = result['is_anomaly'].sum()
            logger.info(f"Detected {anomaly_count} statistical anomalies using z-score method")
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting statistical anomalies: {str(e)}")
            raise
    
    def train_isolation_forest(self, data, features=None):
        """
        Train an Isolation Forest model for anomaly detection.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset for training
        features : list, optional
            List of features to use for training. If None, uses spread_diff and its derivatives
            
        Returns:
        --------
        sklearn.ensemble.IsolationForest
            Trained model
        """
        try:
            # Define features if not provided
            if features is None:
                # Use spread_diff and relevant features if available
                potential_features = [
                    'spread_diff', 
                    'diff_zscore_5d', 'diff_zscore_10d', 'diff_zscore_20d',
                    'relative_movement'
                ]
                features = [f for f in potential_features if f in data.columns]
            
            # Prepare training data
            X = data[features].copy()
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill')
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Isolation Forest
            model = IsolationForest(
                contamination=0.05,  # Expected percentage of anomalies
                random_state=42,
                n_estimators=100
            )
            model.fit(X_scaled)
            
            # Save model and scaler
            self.model = model
            self.scaler = scaler
            self.features = features
            
            # Save to disk
            os.makedirs(self.config["model_save_path"], exist_ok=True)
            joblib.dump(model, os.path.join(self.config["model_save_path"], "isolation_forest_model.pkl"))
            joblib.dump(scaler, os.path.join(self.config["model_save_path"], "scaler.pkl"))
            joblib.dump(features, os.path.join(self.config["model_save_path"], "features.pkl"))
            
            logger.info(f"Trained Isolation Forest model with features: {features}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training Isolation Forest model: {str(e)}")
            raise
    
    def load_model(self, model_path=None):
        """
        Load a previously trained model.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to the model file. If None, uses default path
            
        Returns:
        --------
        sklearn.ensemble.IsolationForest
            Loaded model
        """
        try:
            if model_path is None:
                model_path = os.path.join(self.config["model_save_path"], "isolation_forest_model.pkl")
                scaler_path = os.path.join(self.config["model_save_path"], "scaler.pkl")
                features_path = os.path.join(self.config["model_save_path"], "features.pkl")
            
            # Load model and scaler
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.features = joblib.load(features_path)
            
            logger.info(f"Loaded model from {model_path}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def detect_anomalies_with_model(self, data):
        """
        Detect anomalies using the trained model.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset to analyze
            
        Returns:
        --------
        pandas.DataFrame
            Original data with anomaly flags
        """
        try:
            if self.model is None:
                try:
                    self.load_model()
                except:
                    logger.warning("No model found. Training a new one.")
                    self.train_isolation_forest(data)
            
            # Make a copy to avoid modifying the original
            result = data.copy()
            
            # Prepare features
            X = result[self.features].copy()
            X = X.fillna(method='ffill').fillna(method='bfill')
            X_scaled = self.scaler.transform(X)
            
            # Predict anomalies
            # Isolation Forest returns -1 for anomalies and 1 for normal points
            predictions = self.model.predict(X_scaled)
            anomaly_score = self.model.decision_function(X_scaled)
            
            # Add predictions to result
            result['is_anomaly'] = (predictions == -1)
            result['anomaly_score'] = anomaly_score
            
            # Count anomalies
            anomaly_count = result['is_anomaly'].sum()
            logger.info(f"Detected {anomaly_count} anomalies using Isolation Forest model")
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting anomalies with model: {str(e)}")
            raise
    
    def detect_divergence(self, data, corp_col='spread_corp', benchmark_col='spread_benchmark', window=10):
        """
        Detect when spreads are moving away from each other (diverging).
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset containing spread data
        corp_col : str
            Column name for corporate spread
        benchmark_col : str
            Column name for benchmark spread
        window : int
            Window size for calculating correlation
            
        Returns:
        --------
        pandas.DataFrame
            Original data with divergence flags
        """
        try:
            # Make a copy to avoid modifying the original
            result = data.copy()
            
            # Calculate rolling correlation
            result['rolling_correlation'] = result[corp_col].rolling(window=window).corr(result[benchmark_col])
            
            # Calculate the rate of change in spread difference
            result['spread_diff_change'] = result['spread_diff'].diff()
            
            # Define divergence as correlation below threshold and increasing spread difference
            # Adjust these thresholds based on your specific data patterns
            correlation_threshold = 0.3
            result['is_diverging'] = ((result['rolling_correlation'] < correlation_threshold) & 
                                     (result['spread_diff_change'].abs() > result['spread_diff_change'].abs().rolling(window=window).mean()))
            
            # Add a divergence score (higher means stronger divergence)
            result['divergence_score'] = ((1 - result['rolling_correlation']) * 
                                         result['spread_diff_change'].abs())
            
            # Count divergence instances
            divergence_count = result['is_diverging'].sum()
            logger.info(f"Detected {divergence_count} instances of spread divergence")
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting divergence: {str(e)}")
            raise


###########################################
# 4. Notification Module
###########################################

class NotificationSystem:
    """
    Sends notifications when anomalies are detected.
    """
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        
    def send_email(self, subject, message, recipient=None, html_content=None):
        """
        Send an email notification.
        
        Parameters:
        -----------
        subject : str
            Email subject
        message : str
            Email plain text message
        recipient : str, optional
            Email recipient. If None, uses config
        html_content : str, optional
            HTML version of the email
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Set up email parameters
            sender = self.config["smtp_username"]
            recipient = recipient or self.config["notification_email"]
            
            # Create message container
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = sender
            msg['To'] = recipient
            
            # Create the plain-text message
            text_part = MIMEText(message, 'plain')
            msg.attach(text_part)
            
            # Add HTML version if provided
            if html_content:
                html_part = MIMEText(html_content, 'html')
                msg.attach(html_part)
            
            # Connect to server and send
            server = smtplib.SMTP(self.config["smtp_server"], self.config["smtp_port"])
            server.starttls()
            server.login(self.config["smtp_username"], self.config["smtp_password"])
            server.sendmail(sender, recipient, msg.as_string())
            server.quit()
            
            logger.info(f"Sent email notification to {recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return False
    
    def notify_anomaly(self, anomaly_data, date=None):
        """
        Send notification about detected anomalies.
        
        Parameters:
        -----------
        anomaly_data : pandas.DataFrame
            Data containing the anomalies
        date : datetime, optional
            Specific date for the anomaly. If None, uses latest
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Filter anomalies
            anomalies = anomaly_data[anomaly_data['is_anomaly']]
            
            if anomalies.empty:
                logger.info("No anomalies to notify about")
                return False
            
            # If date specified, filter for that date
            if date:
                anomalies = anomalies[anomalies['date'] == date]
                if anomalies.empty:
                    logger.info(f"No anomalies found for date {date}")
                    return False
            
            # Get the most recent anomaly if date not specified
            if not date:
                latest_anomaly = anomalies.iloc[-1]
                anomaly_date = latest_anomaly['date']
            else:
                anomaly_date = date
            
            # Create notification message
            subject = f"ALERT: Spread Anomaly Detected on {anomaly_date.strftime('%Y-%m-%d')}"
            
            # Plain text message
            message = f"""
FICC Spread Analysis Alert
--------------------------

Anomaly detected in spread data on {anomaly_date.strftime('%Y-%m-%d')}.

Details:
- Corporate Spread: {anomalies.iloc[-1]['spread_corp']:.2f} bps
- Benchmark Spread: {anomalies.iloc[-1]['spread_benchmark']:.2f} bps
- Spread Difference: {anomalies.iloc[-1]['spread_diff']:.2f} bps

This is outside the expected range based on recent historical patterns.

Please review the data and take appropriate action.

---
This is an automated notification from the FICC Spread Analysis System.
            """
            
            # HTML version of the message
            html_content = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .header {{ background-color: #f44336; color: white; padding: 10px; }}
        .content {{ padding: 15px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>FICC Spread Analysis Alert</h2>
    </div>
    <div class="content">
        <p>Anomaly detected in spread data on <strong>{anomaly_date.strftime('%Y-%m-%d')}</strong>.</p>
        
        <h3>Details:</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Corporate Spread</td><td>{anomalies.iloc[-1]['spread_corp']:.2f} bps</td></tr>
            <tr><td>Benchmark Spread</td><td>{anomalies.iloc[-1]['spread_benchmark']:.2f} bps</td></tr>
            <tr><td>Spread Difference</td><td>{anomalies.iloc[-1]['spread_diff']:.2f} bps</td></tr>
        </table>
        
        <p>This is outside the expected range based on recent historical patterns.</p>
        
        <p>Please review the data and take appropriate action.</p>
        
        <hr>
        <p><em>This is an automated notification from the FICC Spread Analysis System.</em></p>
    </div>
</body>
</html>
            """
            
            # Send the notification
            return self.send_email(subject, message, html_content=html_content)
            
        except Exception as e:
            logger.error(f"Error preparing anomaly notification: {str(e)}")
            return False


###########################################
# 5. Forecasting Module
###########################################

class SpreadForecaster:
    """
    Forecasts future spread values using time series methods.
    """
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.model = None
        
    def train_arima_model(self, data, column='spread_diff', order=(5,1,0)):
        """
        Train an ARIMA model for forecasting.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset for training
        column : str
            Column to forecast
        order : tuple
            ARIMA model order (p,d,q)
            
        Returns:
        --------
        statsmodels.tsa.arima.model.ARIMAResults
            Trained model
        """
        try:
            # Prepare training data
            train_data = data[column].copy()
            
            # Train ARIMA model
            model = ARIMA(train_data, order=order)
            results = model.fit()
            
            # Save model
            self.model = results
            self.model_type = 'ARIMA'
            self.target_column = column
            
            # Save model parameters for future reference
            self.model_params = {
                'order': order,
                'target_column': column
            }
            
            # Save model to disk
            os.makedirs(self.config["model_save_path"], exist_ok=True)
            results.save(os.path.join(self.config["model_save_path"], "arima_model.pkl"))
            
            with open(os.path.join(self.config["model_save_path"], "arima_params.json"), 'w') as f:
                import json
                json.dump(self.model_params, f)
            
            logger.info(f"Trained ARIMA{order} model for {column}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}")
            raise
    
    def train_sarimax_model(self, data, target_column='spread_diff', exog_columns=None, 
                          order=(5,1,0), seasonal_order=(0,0,0,0)):
        """
        Train a SARIMAX model for forecasting with exogenous variables.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset for training
        target_column : str
            Column to forecast
        exog_columns : list, optional
            List of exogenous variables columns
        order : tuple
            ARIMA model order (p,d,q)
        seasonal_order : tuple
            Seasonal order (P,D,Q,s)
            
        Returns:
        --------
        statsmodels.tsa.statespace.sarimax.SARIMAXResults
            Trained model
        """
        try:
            # Prepare training data
            train_data = data[target_column].copy()
            
            # Prepare exogenous variables if provided
            exog = None
            if exog_columns:
                exog = data[exog_columns].copy()
            
            # Train SARIMAX model
            model = SARIMAX(train_data, exog=exog, 
                          order=order, seasonal_order=seasonal_order)
            results = model.fit(disp=False)
            
            # Save model
            self.model = results
            self.model_type = 'SARIMAX'
            self.target_column = target_column
            self.exog_columns = exog_columns
            
            # Save model parameters
            self.model_params = {
                'order': order,
                'seasonal_order': seasonal_order,
                'target_column': target_column,
                'exog_columns': exog_columns
            }
            
            # Save model to disk
            os.makedirs(self.config["model_save_path"], exist_ok=True)
            results.save(os.path.join(self.config["model_save_path"], "sarimax_model.pkl"))
            
            with open(os.path.join(self.config["model_save_path"], "sarimax_params.json"), 'w') as f:
                import json
                json.dump({
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'target_column': target_column,
                    'exog_columns': exog_columns
                }, f)
            
            logger.info(f"Trained SARIMAX model for {target_column}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training SARIMAX model: {str(e)}")
            raise
    
    def load_model(self, model_type='ARIMA', model_path=None):
        """
        Load a previously trained model.
        
        Parameters:
        -----------
        model_type : str
            Type of model to load ('ARIMA' or 'SARIMAX')
        model_path : str, optional
            Path to the model file. If None, uses default path
            
        Returns:
        --------
        statsmodels model
            Loaded model
        """
        try:
            if model_path is None:
                if model_type == 'ARIMA':
                    model_path = os.path.join(self.config["model_save_path"], "arima_model.pkl")
                    params_path = os.path.join(self.config["model_save_path"], "arima_params.json")
                else:
                    model_path = os.path.join(self.config["model_save_path"], "sarimax_model.pkl")
                    params_path = os.path.join(self.config["model_save_path"], "sarimax_params.json")
            
            # Load model
            if model_type == 'ARIMA':
                self.model = ARIMA.load(model_path)
            else:
                self.model = SARIMAX.load(model_path)
            
            self.model_type = model_type
            
            # Load parameters
            with open(params_path, 'r') as f:
                import json
                self.model_params = json.load(f)
            
            self.target_column = self.model_params['target_column']
            if model_type == 'SARIMAX':
                self.exog_columns = self.model_params['exog_columns']
            
            logger.info(f"Loaded {model_type} model from {model_path}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def forecast(self, steps=30, exog_future=None, return_conf_int=True, alpha=0.05):
        """
        Generate forecasts using the trained model.
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast
        exog_future : pandas.DataFrame, optional
            Future values of exogenous variables for SARIMAX
        return_conf_int : bool
            Whether to return confidence intervals
        alpha : float
            Significance level for confidence intervals
            
        Returns:
        --------
        pandas.DataFrame
            Forecasted values with dates and confidence intervals
        """
        try:
            if self.model is None:
                raise ValueError("No model available. Train or load a model first.")
            
            # Generate forecasts
            if self.model_type == 'ARIMA':
                forecast_result = self.model.forecast(steps=steps)
                if return_conf_int:
                    forecast_result_conf = self.model.forecast(steps=steps, alpha=alpha)
                    lower_ci = forecast_result_conf.conf_int()[:, 0]
                    upper_ci = forecast_result_conf.conf_int()[:, 1]
            else:  # SARIMAX
                if exog_future is None and self.exog_columns:
                    raise ValueError("Exogenous variables required for SARIMAX forecast")
                
                forecast_result = self.model.forecast(steps=steps, exog=exog_future)
                if return_conf_int:
                    forecast_result_conf = self.model.get_forecast(steps=steps, exog=exog_future)
                    lower_ci = forecast_result_conf.conf_int(alpha=alpha)[:, 0]
                    upper_ci = forecast_result_conf.conf_int(alpha=alpha)[:, 1]
            
            # Create DataFrame with forecasts
            # We need to start from the end date of training data + 1
            if hasattr(self.model, 'data'):
                last_date = self.model.data.dates[-1]
            else:
                last_date = datetime.now()
            
            # Generate future dates
            future_dates = pd.date_range(start=last_date, periods=steps+1)[1:]
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'forecast_date': future_dates,
                'forecast_value': forecast_result
            })
            
            # Add confidence intervals if requested
            if return_conf_int:
                forecast_df['lower_ci'] = lower_ci
                forecast_df['upper_ci'] = upper_ci
            
            logger.info(f"Generated {steps} steps forecast with {self.model_type}")
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def evaluate_forecast_accuracy(self, test_data, forecast_data):
        """
        Evaluate forecast accuracy using various metrics.
        
        Parameters:
        -----------
        test_data : pandas.DataFrame
            Actual data for testing
        forecast_data : pandas.DataFrame
            Forecasted data
            
        Returns:
        --------
        dict
            Dictionary of accuracy metrics
        """
        try:
            # Ensure the dates align
            merged_data = pd.merge(
                test_data[['date', self.target_column]], 
                forecast_data[['forecast_date', 'forecast_value']],
                left_on='date',
                right_on='forecast_date',
                how='inner'
            )
            
            if merged_data.empty:
                raise ValueError("No overlapping dates between test and forecast data")
            
            # Calculate metrics
            actual = merged_data[self.target_column]
            predicted = merged_data['forecast_value']
            
            # Mean Absolute Error
            mae = np.mean(np.abs(actual - predicted))
            
            # Mean Absolute Percentage Error
            # Avoid division by zero
            non_zero_actual = actual != 0
            if non_zero_actual.any():
                mape = np.mean(np.abs((actual[non_zero_actual] - predicted[non_zero_actual]) / actual[non_zero_actual])) * 100
            else:
                mape = np.nan
            
            # Root Mean Squared Error
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            
            # R-squared
            ss_total = np.sum((actual - np.mean(actual)) ** 2)
            ss_residual = np.sum((actual - predicted) ** 2)
            r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else np.nan
            
            # Direction Accuracy (percentage of times the direction is predicted correctly)
            actual_direction = np.sign(actual.diff().dropna())
            predicted_direction = np.sign(predicted.diff().dropna())
            # Align the arrays
            actual_direction = actual_direction[1:]
            predicted_direction = predicted_direction[1:]
            direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            metrics = {
                'MAE': mae,
                'MAPE': mape,
                'RMSE': rmse,
                'R2': r2,
                'Direction_Accuracy': direction_accuracy
            }
            
            logger.info(f"Forecast evaluation metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating forecast accuracy: {str(e)}")
            raise


###########################################
# 6. Main Application
###########################################

class SpreadAnalysisApp:
    """
    Main application class that orchestrates all components.
    """
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.data_processor = DataProcessor(config)
        self.visualizer = VisualizationTool(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.notifier = NotificationSystem(config)
        self.forecaster = SpreadForecaster(config)
        
        # Initialize state variables
        self.data = None
        self.anomalies = None
        self.forecast = None
        self.real_time_running = False
        self.real_time_timer = None
    
    def initialize(self, corporate_file, benchmark_file):
        """
        Initialize the application by loading and processing data.
        
        Parameters:
        -----------
        corporate_file : str
            Path to corporate spread data file
        benchmark_file : str
            Path to benchmark spread data file
            
        Returns:
        --------
        bool
            True if initialization successful
        """
        try:
            logger.info("Initializing Spread Analysis Application")
            
            # Load data
            self.data = self.data_processor.load_data(corporate_file, benchmark_file)
            
            # Clean the data
            self.data = self.data_processor.clean_data()
            
            # Add features
            self.data = self.data_processor.add_features()
            
            # Detect anomalies
            self.anomalies = self.anomaly_detector.detect_statistical_anomalies(self.data)
            
            # Train anomaly detection model
            self.anomaly_detector.train_isolation_forest(self.data)
            
            # Train forecasting model
            self.forecaster.train_arima_model(self.data)
            
            # Generate forecast
            self.forecast = self.forecaster.forecast(steps=30)
            
            logger.info("Application initialization completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing application: {str(e)}")
            return False
    
    def generate_visualizations(self):
        """
        Generate all visualizations.
        
        Returns:
        --------
        bool
            True if successful
        """
        try:
            if self.data is None:
                raise ValueError("No data available. Initialize the application first.")
            
            # Generate trend chart
            trend_path = os.path.join(self.config["visualization_path"], "spread_trend.png")
            self.visualizer.create_spread_trend_chart(self.data, save_path=trend_path)
            
            # Generate anomaly chart
            anomaly_path = os.path.join(self.config["visualization_path"], "anomalies.png")
            self.visualizer.create_anomaly_chart(self.data, self.anomalies, save_path=anomaly_path)
            
            # Generate forecast chart
            forecast_path = os.path.join(self.config["visualization_path"], "forecast.png")
            self.visualizer.create_forecast_chart(self.data, self.forecast, save_path=forecast_path)
            
            # Generate comprehensive dashboard
            dashboard_path = os.path.join(self.config["visualization_path"], "dashboard.png")
            self.visualizer.create_dashboard(self.data, self.anomalies, self.forecast, save_path=dashboard_path)
            
            logger.info("All visualizations generated successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return False
    
    def notify_recent_anomalies(self, days=1):
        """
        Send notifications for recent anomalies.
        
        Parameters:
        -----------
        days : int
            Number of recent days to check for anomalies
            
        Returns:
        --------
        bool
            True if notifications sent
        """
        try:
            if self.anomalies is None:
                raise ValueError("No anomaly data available")
            
            # Get recent data
            latest_date = self.anomalies['date'].max()
            start_date = latest_date - timedelta(days=days)
            
            # Filter recent anomalies
            recent_anomalies = self.anomalies[(self.anomalies['date'] >= start_date) & 
                                             self.anomalies['is_anomaly']]
            
            if recent_anomalies.empty:
                logger.info(f"No anomalies found in the last {days} days")
                return False
            
            # Send notifications for each anomaly date
            dates_notified = []
            for date in recent_anomalies['date'].unique():
                date_anomalies = recent_anomalies[recent_anomalies['date'] == date]
                success = self.notifier.notify_anomaly(date_anomalies, date=date)
                if success:
                    dates_notified.append(date)
            
            if dates_notified:
                logger.info(f"Sent notifications for anomalies on: {dates_notified}")
                return True
            else:
                logger.info("No notifications were sent")
                return False
            
        except Exception as e:
            logger.error(f"Error sending notifications: {str(e)}")
            return False
    
    def start_real_time_monitoring(self, interval=None):
        """
        Start real-time monitoring process.
        
        Parameters:
        -----------
        interval : int, optional
            Time interval in seconds between checks. If None, uses config
            
        Returns:
        --------
        bool
            True if started successfully
        """
        if self.real_time_running:
            logger.warning("Real-time monitoring is already running")
            return False
        
        try:
            interval = interval or self.config["real_time_interval"]
            
            def monitor_task():
                try:
                    logger.info("Running real-time monitoring check")
                    
                    # In a real implementation, this would fetch the latest data
                    # For this example, we'll simulate by using the most recent data
                    latest_data = self.data_processor.get_latest_data(days=5)
                    
                    # Detect anomalies
                    latest_anomalies = self.anomaly_detector.detect_anomalies_with_model(latest_data)
                    
                    # Check if the latest data point is an anomaly
                    latest_date = latest_data['date'].max()
                    latest_point = latest_anomalies[latest_anomalies['date'] == latest_date]
                    
                    if not latest_point.empty and latest_point['is_anomaly'].any():
                        logger.info(f"Anomaly detected in real-time data for {latest_date}")
                        self.notifier.notify_anomaly(latest_point)
                    
                    # Update forecast
                    new_forecast = self.forecaster.forecast(steps=15)
                    
                    # Schedule the next check if still running
                    if self.real_time_running:
                        self.real_time_timer = Timer(interval, monitor_task)
                        self.real_time_timer.daemon = True
                        self.real_time_timer.start()
                
                except Exception as e:
                    logger.error(f"Error in real-time monitoring task: {str(e)}")
                    # Still reschedule to continue monitoring
                    if self.real_time_running:
                        self.real_time_timer = Timer(interval, monitor_task)
                        self.real_time_timer.daemon = True
                        self.real_time_timer.start()
            
            # Start monitoring
            self.real_time_running = True
            self.real_time_timer = Timer(interval, monitor_task)
            self.real_time_timer.daemon = True
            self.real_time_timer.start()
            
            logger.info(f"Real-time monitoring started with {interval} second interval")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting real-time monitoring: {str(e)}")
            self.real_time_running = False
            return False
    
    def stop_real_time_monitoring(self):
        """
        Stop the real-time monitoring process.
        
        Returns:
        --------
        bool
            True if stopped successfully
        """
        try:
            if not self.real_time_running:
                logger.warning("Real-time monitoring is not running")
                return False
            
            # Stop monitoring
            self.real_time_running = False
            
            # Cancel timer if it exists
            if self.real_time_timer:
                self.real_time_timer.cancel()
                self.real_time_timer = None
            
            logger.info("Real-time monitoring stopped")
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping real-time monitoring: {str(e)}")
            return False
    
    def generate_report(self, output_path=None):
        """
        Generate a comprehensive report with all analyses.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the report
            
        Returns:
        --------
        str
            Path to the report
        """
        try:
            if self.data is None:
                raise ValueError("No data available. Initialize the application first.")
            
            # Default output path
            if output_path is None:
                output_path = os.path.join(self.config["visualization_path"], "spread_analysis_report.html")
            
            # Create report
            report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FICC Spread Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
        .stats {{ display: flex; flex-wrap: wrap; }}
        .stat-box {{ background-color: #f8f9fa; padding: 15px; margin: 10px; border-radius: 5px; flex: 1; min-width: 200px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .anomaly {{ background-color: #ffcccc; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>FICC Spread Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Dataset Summary</h2>
            <div class="stats">
                <div class="stat-box">
                    <h3>Date Range</h3>
                    <p>From: {self.data['date'].min().strftime('%Y-%m-%d')}</p>
                    <p>To: {self.data['date'].max().strftime('%Y-%m-%d')}</p>
                    <p>Total Days: {len(self.data)}</p>
                </div>
                <div class="stat-box">
                    <h3>Corporate Spread</h3>
                    <p>Average: {self.data['spread_corp'].mean():.2f} bps</p>
                    <p>Min: {self.data['spread_corp'].min():.2f} bps</p>
                    <p>Max: {self.data['spread_corp'].max():.2f} bps</p>
                </div>
                <div class="stat-box">
                    <h3>Benchmark Spread</h3>
                    <p>Average: {self.data['spread_benchmark'].mean():.2f} bps</p>
                    <p>Min: {self.data['spread_benchmark'].min():.2f} bps</p>
                    <p>Max: {self.data['spread_benchmark'].max():.2f} bps</p>
                </div>
                <div class="stat-box">
                    <h3>Spread Difference</h3>
                    <p>Average: {self.data['spread_diff'].mean():.2f} bps</p>
                    <p>Min: {self.data['spread_diff'].min():.2f} bps</p>
                    <p>Max: {self.data['spread_diff'].max():.2f} bps</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
            <h3>Spread Trend</h3>
            <img src="spread_trend.png" alt="Spread Trend Chart">
            
            <h3>Anomalies</h3>
            <img src="anomalies.png" alt="Anomaly Chart">
            
            <h3>Forecast</h3>
            <img src="forecast.png" alt="Forecast Chart">
            
            <h3>Dashboard</h3>
            <img src="dashboard.png" alt="Dashboard">
        </div>
        
        <div class="section">
            <h2>Anomaly Detection</h2>
            <p>Total anomalies detected: {self.anomalies['is_anomaly'].sum()}</p>
            <p>Anomaly rate: {(self.anomalies['is_anomaly'].sum() / len(self.anomalies) * 100):.2f}%</p>
            
            <h3>Recent Anomalies</h3>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Corporate Spread</th>
                    <th>Benchmark Spread</th>
                    <th>Spread Difference</th>
                    <th>Anomaly Score</th>
                </tr>
            """
            
            # Add recent anomalies to the table
            recent_days = 30
            recent_data = self.anomalies[self.anomalies['date'] >= 
                                        (self.anomalies['date'].max() - timedelta(days=recent_days))]
            recent_anomalies = recent_data[recent_data['is_anomaly']]
            
            for _, row in recent_anomalies.iterrows():
                report_html += f"""
                <tr class="anomaly">
                    <td>{row['date'].strftime('%Y-%m-%d')}</td>
                    <td>{row['spread_corp']:.2f}</td>
                    <td>{row['spread_benchmark']:.2f}</td>
                    <td>{row['spread_diff']:.2f}</td>
                    <td>{row.get('anomaly_score', 'N/A')}</td>
                </tr>
                """
            
            # Add forecast section
            report_html += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>Forecast</h2>
            <p>Forecast period: {self.forecast['forecast_date'].min().strftime('%Y-%m-%d')} to {self.forecast['forecast_date'].max().strftime('%Y-%m-%d')}</p>
            
            <h3>Forecast Values</h3>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Forecasted Spread Difference</th>
                    <th>Lower CI</th>
                    <th>Upper CI</th>
                </tr>
            """
            
            # Add forecast values to the table
            for _, row in self.forecast.iterrows():
                report_html += f"""
                <tr>
                    <td>{row['forecast_date'].strftime('%Y-%m-%d')}</td>
                    <td>{row['forecast_value']:.2f}</td>
                    <td>{row.get('lower_ci', 'N/A')}</td>
                    <td>{row.get('upper_ci', 'N/A')}</td>
                </tr>
                """
            
            # Finalize report
            report_html += """
            </table>
        </div>
    </div>
</body>
</html>
            """
            
            # Write report to file
            with open(output_path, 'w') as f:
                f.write(report_html)
            
            logger.info(f"Generated report saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise


###########################################
# Example Usage
###########################################

def main():
    """
    Example usage of the FICC Spread Analysis application.
    """
    try:
        # Create application instance
        app = SpreadAnalysisApp()
        
        # Initialize with data
        # In a real scenario, you would provide actual file paths
        corporate_file = "corporate_spreads.csv"
        benchmark_file = "benchmark_spreads.csv"
        
        # For demo purposes, let's create some synthetic data if files don't exist
        if not os.path.exists(corporate_file) or not os.path.exists(benchmark_file):
            logger.info("Creating synthetic data for demonstration")
            create_synthetic_data(corporate_file, benchmark_file)
        
        # Initialize application
        app.initialize(corporate_file, benchmark_file)
        
        # Generate visualizations
        app.generate_visualizations()
        
        # Generate report
        app.generate_report()
        
        # Start real-time monitoring (for demonstration)
        app.start_real_time_monitoring(interval=60)  # Every 60 seconds
        
        # In a real application, we would keep the program running
        # For this example, let it run for a while then stop
        logger.info("Application running...")
        time.sleep(180)  # Run for 3 minutes
        
        # Stop monitoring
        app.stop_real_time_monitoring()
        
        logger.info("Spread Analysis application completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")


def create_synthetic_data(corporate_file, benchmark_file):
    """
    Create synthetic data for demonstration purposes.
    """
    try:
        # Date range
        date_range = pd.date_range(start='2020-01-01', end='2025-01-01', freq='B')
        
        # Create corporate spread data
        np.random.seed(42)
        corp_base = 150  # Base spread level in bps
        corp_trend = np.linspace(0, 50, len(date_range))  # Upward trend
        corp_seasonal = 20 * np.sin(np.linspace(0, 12*np.pi, len(date_range)))  # Seasonal pattern
        corp_noise = np.random.normal(0, 15, len(date_range))  # Random noise
        
        # Add some anomalies
        anomaly_idx = np.random.choice(len(date_range), 20, replace=False)
        corp_anomalies = np.zeros(len(date_range))
        corp_anomalies[anomaly_idx] = np.random.normal(0, 50, len(anomaly_idx))
        
        # Combine components
        corp_spread = corp_base + corp_trend + corp_seasonal + corp_noise + corp_anomalies
        corp_spread = np.maximum(corp_spread, 10)  # Ensure no negative spreads
        
        # Create benchmark spread data (correlated with corporate but less volatile)
        bench_base = 100  # Base spread level in bps
        bench_trend = 0.7 * corp_trend  # Similar but smaller trend
        bench_seasonal = 0.8 * corp_seasonal  # Similar but smaller seasonal pattern
        bench_noise = np.random.normal(0, 8, len(date_range))  # Less random noise
        
        # Combine components
        bench_spread = bench_base + bench_trend + bench_seasonal + bench_noise
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
        
        # Save to CSV
        corp_df.to_csv(corporate_file, index=False)
        bench_df.to_csv(benchmark_file, index=False)
        
        logger.info(f"Synthetic data created: {corporate_file}, {benchmark_file}")
        
    except Exception as e:
        logger.error(f"Error creating synthetic data: {str(e)}")
        raise


if __name__ == "__main__":
    main()
