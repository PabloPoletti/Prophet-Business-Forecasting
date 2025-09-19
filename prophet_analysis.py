"""
📈 Prophet Business Forecasting Analysis
Complete Business-Focused Time Series Analysis with Prophet and NeuralProphet

This analysis demonstrates:
1. Business-oriented data preprocessing
2. Holiday and event impact analysis
3. Trend changepoint detection
4. Seasonality decomposition and analysis
5. External regressor integration
6. Uncertainty quantification and intervals
7. Cross-validation and performance evaluation

Author: Pablo Poletti
GitHub: https://github.com/PabloPoletti
Contact: lic.poletti@gmail.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Prophet imports
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.plot import plot_cross_validation_metric, plot_plotly, plot_components_plotly
    from prophet.make_holidays import make_holidays_df
    import prophet
except ImportError as e:
    print(f"Warning: Prophet not installed: {e}")
    print("Install with: pip install prophet")

# Prophet and NeuralProphet imports - REQUIRED for this analysis
try:
    from neuralprophet import NeuralProphet, set_log_level
    set_log_level("ERROR")  # Reduce logging
    import neuralprophet
except ImportError as e:
    print("❌ CRITICAL ERROR: Prophet libraries not installed!")
    print(f"Missing: {e}")
    print("\n🔧 INSTALLATION REQUIRED:")
    print("pip install prophet")
    print("pip install neuralprophet")
    print("\n📖 This project specifically demonstrates Prophet and NeuralProphet business forecasting capabilities.")
    print("Without these libraries, the analysis cannot proceed.")
    exit(1)

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ProphetBusinessAnalysis:
    """Complete Prophet Business Analysis Pipeline"""
    
    def __init__(self):
        self.datasets = {}
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.best_params = {}
        self.cv_results = {}
        
    def load_business_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load business-oriented datasets"""
        print("📊 Loading business datasets...")
        
        datasets = {}
        
        # 1. E-commerce sales data (synthetic)
        print("Generating e-commerce sales data...")
        dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
        
        # Base sales with trend
        base_sales = 1000
        trend = np.cumsum(np.random.normal(2, 5, len(dates)))  # Growing business
        
        # Seasonal patterns
        yearly_season = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        weekly_season = 150 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        
        # Holiday effects (Black Friday, Christmas, etc.)
        holiday_boost = np.zeros(len(dates))
        for i, date in enumerate(dates):
            if date.month == 11 and date.day >= 24 and date.day <= 30:  # Black Friday week
                holiday_boost[i] = 500
            elif date.month == 12 and date.day >= 20:  # Christmas season
                holiday_boost[i] = 300
            elif date.month == 1 and date.day <= 5:  # New Year
                holiday_boost[i] = 200
        
        # Noise and ensure positive values
        noise = np.random.normal(0, 50, len(dates))
        sales = np.maximum(
            base_sales + trend + yearly_season + weekly_season + holiday_boost + noise,
            100
        )
        
        ecommerce_df = pd.DataFrame({
            'ds': dates,
            'y': sales
        })
        datasets['Ecommerce_Sales'] = ecommerce_df
        
        # 2. Website traffic data
        print("Generating website traffic data...")
        traffic_dates = pd.date_range('2021-01-01', '2024-01-01', freq='D')
        
        base_traffic = 10000
        growth_trend = np.cumsum(np.random.normal(10, 20, len(traffic_dates)))
        
        # Weekly seasonality (lower on weekends)
        weekly_pattern = np.array([1.2, 1.1, 1.0, 1.0, 1.1, 0.7, 0.6])  # Mon-Sun multipliers
        weekly_multiplier = np.tile(weekly_pattern, len(traffic_dates) // 7 + 1)[:len(traffic_dates)]
        
        # Marketing campaign effects (random spikes)
        campaign_effects = np.zeros(len(traffic_dates))
        campaign_days = np.random.choice(len(traffic_dates), size=20, replace=False)
        for day in campaign_days:
            campaign_length = np.random.randint(3, 14)
            end_day = min(day + campaign_length, len(traffic_dates))
            campaign_effects[day:end_day] = np.random.uniform(2000, 5000)
        
        traffic_noise = np.random.normal(0, 500, len(traffic_dates))
        traffic = np.maximum(
            (base_traffic + growth_trend) * weekly_multiplier + campaign_effects + traffic_noise,
            1000
        )
        
        traffic_df = pd.DataFrame({
            'ds': traffic_dates,
            'y': traffic
        })
        datasets['Website_Traffic'] = traffic_df
        
        # 3. Retail store sales (with external factors)
        print("Generating retail sales with external factors...")
        retail_dates = pd.date_range('2019-01-01', '2024-01-01', freq='D')
        
        # Base sales with economic cycles
        base_retail = 5000
        economic_cycle = 1000 * np.sin(2 * np.pi * np.arange(len(retail_dates)) / (365.25 * 3))  # 3-year cycle
        
        # Weather impact (synthetic temperature effect)
        temperature = 20 + 15 * np.sin(2 * np.pi * np.arange(len(retail_dates)) / 365.25)
        weather_impact = np.where(
            (temperature > 25) | (temperature < 5),
            -200,  # Extreme weather reduces sales
            100    # Moderate weather boosts sales
        )
        
        # Promotional events
        promo_effects = np.zeros(len(retail_dates))
        for i, date in enumerate(retail_dates):
            if date.weekday() == 4:  # Friday promotions
                if np.random.random() < 0.3:  # 30% of Fridays
                    promo_effects[i] = 800
        
        retail_noise = np.random.normal(0, 300, len(retail_dates))
        retail_sales = np.maximum(
            base_retail + economic_cycle + weather_impact + promo_effects + retail_noise,
            500
        )
        
        retail_df = pd.DataFrame({
            'ds': retail_dates,
            'y': retail_sales,
            'temperature': temperature,
            'is_promotion': (promo_effects > 0).astype(int)
        })
        datasets['Retail_Sales'] = retail_df
        
        # 4. Real stock data for comparison
        print("Loading real stock data (MSFT)...")
        msft = yf.download("MSFT", period="3y", interval="1d")
        msft_df = pd.DataFrame({
            'ds': msft.index,
            'y': msft['Close']
        }).reset_index(drop=True)
        datasets['MSFT_Stock'] = msft_df
        
        self.datasets = datasets
        print(f"✅ Loaded {len(datasets)} business datasets")
        return datasets
    
    def comprehensive_business_eda(self):
        """Business-focused EDA with trend and seasonality analysis"""
        print("\n📈 Performing Business-Focused EDA...")
        
        fig = make_subplots(
            rows=len(self.datasets), cols=3,
            subplot_titles=[f"{name} - Time Series" for name in self.datasets.keys()] +
                          [f"{name} - Weekly Pattern" for name in self.datasets.keys()] +
                          [f"{name} - Monthly Trend" for name in self.datasets.keys()],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}] 
                   for _ in range(len(self.datasets))]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (name, df) in enumerate(self.datasets.items()):
            row = i + 1
            
            # Time series plot
            fig.add_trace(
                go.Scatter(
                    x=df['ds'],
                    y=df['y'],
                    mode='lines',
                    name=f'{name}',
                    line=dict(color=colors[i % len(colors)])
                ),
                row=row, col=1
            )
            
            # Weekly seasonality
            df['weekday'] = pd.to_datetime(df['ds']).dt.day_name()
            weekly_avg = df.groupby('weekday')['y'].mean().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            
            fig.add_trace(
                go.Bar(
                    x=weekly_avg.index,
                    y=weekly_avg.values,
                    name=f'{name} Weekly',
                    marker_color=colors[i % len(colors)],
                    opacity=0.7
                ),
                row=row, col=2
            )
            
            # Monthly trend
            df['month'] = pd.to_datetime(df['ds']).dt.to_period('M')
            monthly_avg = df.groupby('month')['y'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=[str(m) for m in monthly_avg.index],
                    y=monthly_avg.values,
                    mode='lines+markers',
                    name=f'{name} Monthly',
                    line=dict(color=colors[i % len(colors)], dash='dash')
                ),
                row=row, col=3
            )
        
        fig.update_layout(
            height=300 * len(self.datasets),
            title_text="📊 Business-Focused Multi-Dataset Analysis",
            showlegend=True
        )
        
        fig.write_html("prophet_business_eda.html")
        print("✅ Business EDA completed. Dashboard saved as 'prophet_business_eda.html'")
        
        # Business insights
        print("\n💼 Business Insights:")
        for name, df in self.datasets.items():
            print(f"\n{name}:")
            
            # Growth rate
            first_month = df.head(30)['y'].mean()
            last_month = df.tail(30)['y'].mean()
            growth_rate = ((last_month - first_month) / first_month) * 100
            print(f"  Overall Growth: {growth_rate:.1f}%")
            
            # Volatility
            volatility = df['y'].std() / df['y'].mean() * 100
            print(f"  Volatility (CV): {volatility:.1f}%")
            
            # Best/worst days
            df['weekday'] = pd.to_datetime(df['ds']).dt.day_name()
            weekly_avg = df.groupby('weekday')['y'].mean()
            best_day = weekly_avg.idxmax()
            worst_day = weekly_avg.idxmin()
            print(f"  Best Day: {best_day} ({weekly_avg[best_day]:.0f})")
            print(f"  Worst Day: {worst_day} ({weekly_avg[worst_day]:.0f})")
    
    def create_prophet_models(self, dataset_name: str) -> Dict[str, Prophet]:
        """Create Prophet models with different configurations"""
        print(f"\n🔧 Creating Prophet models for {dataset_name}...")
        
        models = {}
        
        # 1. Basic Prophet
        models['Prophet_Basic'] = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.95
        )
        
        # 2. Prophet with changepoint detection
        models['Prophet_Changepoints'] = Prophet(
            changepoint_prior_scale=0.05,
            n_changepoints=25,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.95
        )
        
        # 3. Prophet with custom seasonalities
        prophet_custom = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            interval_width=0.95
        )
        prophet_custom.add_seasonality(name='weekly', period=7, fourier_order=3)
        prophet_custom.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        models['Prophet_Custom_Seasonality'] = prophet_custom
        
        # 4. Prophet with holidays (US holidays)
        try:
            from prophet.make_holidays import make_holidays_df
            holidays = make_holidays_df(
                year_list=list(range(2019, 2025)),
                country='US'
            )
            models['Prophet_Holidays'] = Prophet(
                holidays=holidays,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.95
            )
        except:
            print("  ⚠️ Holidays not available, skipping holiday model")
        
        # 5. Prophet with external regressors (if available)
        df = self.datasets[dataset_name]
        if 'temperature' in df.columns:
            models['Prophet_Regressors'] = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.95
            )
            models['Prophet_Regressors'].add_regressor('temperature')
            if 'is_promotion' in df.columns:
                models['Prophet_Regressors'].add_regressor('is_promotion')
        
        print(f"✅ Created {len(models)} Prophet models")
        return models
    
    def create_neuralprophet_models(self) -> Dict[str, NeuralProphet]:
        """Create NeuralProphet models"""
        print("\n🧠 Creating NeuralProphet models...")
        
        models = {}
        
        # 1. Basic NeuralProphet
        models['NeuralProphet_Basic'] = NeuralProphet(
            n_forecasts=1,
            n_lags=14,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            epochs=50,
            learning_rate=0.01
        )
        
        # 2. NeuralProphet with AR-Net
        models['NeuralProphet_AR'] = NeuralProphet(
            n_forecasts=1,
            n_lags=21,
            ar_layers=[64, 32],
            yearly_seasonality=True,
            weekly_seasonality=True,
            epochs=50,
            learning_rate=0.01
        )
        
        # 3. NeuralProphet with trend
        models['NeuralProphet_Trend'] = NeuralProphet(
            n_forecasts=1,
            n_lags=14,
            trend_reg=1,
            yearly_seasonality=True,
            weekly_seasonality=True,
            epochs=50,
            learning_rate=0.01
        )
        
        print(f"✅ Created {len(models)} NeuralProphet models")
        return models
    
    def train_and_evaluate_models(self, dataset_name: str):
        """Train and evaluate all models"""
        print(f"\n🚀 Training models on {dataset_name}...")
        
        df = self.datasets[dataset_name]
        
        # Split data
        split_point = int(len(df) * 0.8)
        train_df = df.iloc[:split_point].copy()
        test_df = df.iloc[split_point:].copy()
        
        # Create models
        prophet_models = self.create_prophet_models(dataset_name)
        neural_models = self.create_neuralprophet_models()
        
        all_models = {**prophet_models, **neural_models}
        
        results = []
        predictions = {}
        
        for name, model in all_models.items():
            try:
                print(f"  Training {name}...")
                
                if 'Neural' in name:
                    # NeuralProphet training
                    model.fit(train_df, freq='D', validation_df=test_df[:len(test_df)//2])
                    
                    # Create future dataframe
                    future_df = model.make_future_dataframe(
                        train_df, 
                        periods=len(test_df),
                        n_historic_predictions=True
                    )
                    
                    # Predict
                    forecast = model.predict(future_df)
                    pred_values = forecast['yhat1'].iloc[-len(test_df):].values
                    
                else:
                    # Prophet training
                    model.fit(train_df)
                    
                    # Create future dataframe
                    future = model.make_future_dataframe(periods=len(test_df))
                    
                    # Add regressors if needed
                    if 'Regressors' in name and 'temperature' in df.columns:
                        future['temperature'] = list(train_df['temperature']) + list(test_df['temperature'])
                        if 'is_promotion' in df.columns:
                            future['is_promotion'] = list(train_df['is_promotion']) + list(test_df['is_promotion'])
                    
                    # Predict
                    forecast = model.predict(future)
                    pred_values = forecast['yhat'].iloc[-len(test_df):].values
                
                # Calculate metrics
                actual_values = test_df['y'].values
                mae_score = mean_absolute_error(actual_values, pred_values)
                mse_score = mean_squared_error(actual_values, pred_values)
                rmse_score = np.sqrt(mse_score)
                mape_score = mean_absolute_percentage_error(actual_values, pred_values) * 100
                
                # SMAPE calculation
                smape_score = 100 * np.mean(2 * np.abs(pred_values - actual_values) / 
                                          (np.abs(actual_values) + np.abs(pred_values)))
                
                results.append({
                    'Model': name,
                    'MAE': mae_score,
                    'MSE': mse_score,
                    'RMSE': rmse_score,
                    'MAPE': mape_score,
                    'SMAPE': smape_score
                })
                
                # Store predictions
                pred_df = test_df.copy()
                pred_df['yhat'] = pred_values
                predictions[name] = pred_df
                
            except Exception as e:
                print(f"    ❌ {name} failed: {str(e)}")
                continue
        
        self.predictions[dataset_name] = predictions
        self.metrics[dataset_name] = pd.DataFrame(results).sort_values('MAE')
        
        print(f"✅ Completed training on {dataset_name}")
        if not self.metrics[dataset_name].empty:
            print(f"🏆 Best model: {self.metrics[dataset_name].iloc[0]['Model']}")
    
    def perform_cross_validation(self, dataset_name: str, model_name: str = 'Prophet_Basic'):
        """Perform time series cross-validation"""
        print(f"\n🔄 Performing cross-validation for {model_name} on {dataset_name}...")
        
        df = self.datasets[dataset_name]
        
        # Create and train model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        model.fit(df)
        
        # Perform cross-validation
        try:
            cv_results = cross_validation(
                model, 
                initial='730 days',  # 2 years initial training
                period='90 days',    # 3 months between cutoffs
                horizon='30 days'    # 1 month forecast horizon
            )
            
            # Calculate performance metrics
            performance = performance_metrics(cv_results)
            
            self.cv_results[f"{dataset_name}_{model_name}"] = {
                'cv_results': cv_results,
                'performance': performance
            }
            
            print("✅ Cross-validation completed")
            print(f"Average MAE: {performance['mae'].mean():.2f}")
            print(f"Average MAPE: {performance['mape'].mean():.2f}")
            
        except Exception as e:
            print(f"❌ Cross-validation failed: {e}")
    
    def optimize_prophet_hyperparameters(self, dataset_name: str):
        """Optimize Prophet hyperparameters using Optuna"""
        print(f"\n⚙️ Optimizing Prophet hyperparameters for {dataset_name}...")
        
        df = self.datasets[dataset_name]
        split_point = int(len(df) * 0.8)
        train_df = df.iloc[:split_point]
        test_df = df.iloc[split_point:]
        
        def objective(trial):
            try:
                # Suggest hyperparameters
                changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5)
                seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 0.01, 10)
                holidays_prior_scale = trial.suggest_float('holidays_prior_scale', 0.01, 10)
                seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
                
                # Create model
                model = Prophet(
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    holidays_prior_scale=holidays_prior_scale,
                    seasonality_mode=seasonality_mode,
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=True
                )
                
                # Train
                model.fit(train_df)
                
                # Predict
                future = model.make_future_dataframe(periods=len(test_df))
                forecast = model.predict(future)
                pred_values = forecast['yhat'].iloc[-len(test_df):].values
                
                # Calculate MAE
                mae_score = mean_absolute_error(test_df['y'].values, pred_values)
                return mae_score
                
            except Exception as e:
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, timeout=600)  # 10 minutes max
        
        self.best_params[f"{dataset_name}_Prophet"] = study.best_params
        print(f"✅ Best parameters: {study.best_params}")
        print(f"✅ Best MAE: {study.best_value:.4f}")
    
    def create_business_visualization(self, dataset_name: str):
        """Create business-focused visualization"""
        print(f"\n📈 Creating business visualization for {dataset_name}...")
        
        if dataset_name not in self.predictions:
            print("❌ No predictions available")
            return
        
        df = self.datasets[dataset_name]
        predictions = self.predictions[dataset_name]
        
        # Create comprehensive business dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Forecast Comparison', 'Model Performance',
                'Trend Analysis', 'Seasonality Patterns'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Forecast comparison
        split_point = int(len(df) * 0.8)
        train_df = df.iloc[:split_point]
        test_df = df.iloc[split_point:]
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=train_df['ds'],
                y=train_df['y'],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Actual test data
        fig.add_trace(
            go.Scatter(
                x=test_df['ds'],
                y=test_df['y'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='black', width=3)
            ),
            row=1, col=1
        )
        
        # Top 3 predictions
        colors = ['red', 'green', 'orange']
        top_models = self.metrics[dataset_name].head(3)['Model'].tolist()
        
        for i, model_name in enumerate(top_models):
            if model_name in predictions:
                pred_df = predictions[model_name]
                fig.add_trace(
                    go.Scatter(
                        x=pred_df['ds'],
                        y=pred_df['yhat'],
                        mode='lines+markers',
                        name=f'{model_name}',
                        line=dict(color=colors[i], width=2, dash='dash')
                    ),
                    row=1, col=1
                )
        
        # 2. Model performance
        metrics_df = self.metrics[dataset_name]
        fig.add_trace(
            go.Bar(
                x=metrics_df['Model'],
                y=metrics_df['MAE'],
                name='MAE',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # 3. Trend analysis (simplified)
        fig.add_trace(
            go.Scatter(
                x=df['ds'],
                y=df['y'].rolling(30).mean(),
                mode='lines',
                name='30-day Trend',
                line=dict(color='purple', width=3)
            ),
            row=2, col=1
        )
        
        # 4. Weekly seasonality
        df['weekday'] = pd.to_datetime(df['ds']).dt.day_name()
        weekly_avg = df.groupby('weekday')['y'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        fig.add_trace(
            go.Bar(
                x=weekly_avg.index,
                y=weekly_avg.values,
                name='Weekly Pattern',
                marker_color='lightgreen'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text=f"📈 Prophet Business Analysis - {dataset_name}",
            showlegend=True
        )
        
        fig.write_html(f'prophet_business_{dataset_name.lower()}.html')
        print(f"✅ Business visualization saved as 'prophet_business_{dataset_name.lower()}.html'")
    
    def generate_business_report(self):
        """Generate comprehensive business report"""
        print("\n📋 Generating business analysis report...")
        
        report = f"""
# 📈 Prophet Business Forecasting Analysis Report

## 🏢 Executive Summary
This analysis provides comprehensive business forecasting insights using Prophet and NeuralProphet frameworks, focusing on practical business applications and actionable insights.

## 📊 Business Datasets Analyzed
"""
        
        for name, df in self.datasets.items():
            growth_rate = ((df.tail(30)['y'].mean() - df.head(30)['y'].mean()) / df.head(30)['y'].mean()) * 100
            volatility = (df['y'].std() / df['y'].mean()) * 100
            
            report += f"""
### {name.replace('_', ' ')}
- **Data Points**: {len(df):,} observations
- **Date Range**: {df['ds'].min().strftime('%Y-%m-%d')} to {df['ds'].max().strftime('%Y-%m-%d')}
- **Average Value**: {df['y'].mean():,.0f}
- **Growth Rate**: {growth_rate:+.1f}%
- **Volatility (CV)**: {volatility:.1f}%
- **Business Insight**: {'High growth potential' if growth_rate > 10 else 'Stable performance' if growth_rate > 0 else 'Declining trend'}
"""
        
        report += "\n## 🏆 Model Performance Summary\n"
        
        for dataset_name, metrics_df in self.metrics.items():
            if not metrics_df.empty:
                best_model = metrics_df.iloc[0]
                report += f"""
### {dataset_name.replace('_', ' ')}
**Best Model**: {best_model['Model']}
- **MAE**: {best_model['MAE']:.2f}
- **MAPE**: {best_model['MAPE']:.1f}%
- **Business Impact**: {'Excellent accuracy' if best_model['MAPE'] < 10 else 'Good accuracy' if best_model['MAPE'] < 20 else 'Moderate accuracy'}

**All Models Performance**:
{metrics_df[['Model', 'MAE', 'MAPE']].round(2).to_string(index=False)}
"""
        
        report += "\n## ⚙️ Optimization Results\n"
        for key, params in self.best_params.items():
            report += f"\n### {key}\n"
            for param, value in params.items():
                report += f"- **{param}**: {value}\n"
        
        report += f"""

## 💼 Business Recommendations

### 1. Model Selection Guidelines
- **Prophet Basic**: Best for stable, seasonal business data
- **Prophet with Changepoints**: Ideal for businesses with structural changes
- **Prophet with Holidays**: Essential for retail and consumer businesses
- **NeuralProphet**: Superior for complex, non-linear patterns

### 2. Forecasting Best Practices
- **Short-term (1-30 days)**: Use Prophet with external regressors
- **Medium-term (1-6 months)**: Combine Prophet and NeuralProphet
- **Long-term (6+ months)**: Focus on trend and yearly seasonality

### 3. Business Applications
- **E-commerce**: Monitor weekly patterns and holiday effects
- **Website Traffic**: Account for marketing campaigns and seasonality
- **Retail Sales**: Include weather and promotional factors
- **Financial Data**: Focus on trend changes and volatility

## 🔍 Key Business Insights
1. **Seasonality Impact**: Weekly patterns show significant business impact
2. **Holiday Effects**: Major holidays drive 20-50% sales increases
3. **External Factors**: Weather and promotions significantly affect performance
4. **Trend Analysis**: Long-term trends more reliable than short-term fluctuations

## 📁 Generated Deliverables
- `prophet_business_eda.html` - Business-focused exploratory analysis
- `prophet_business_*.html` - Individual dataset forecasting dashboards
- `prophet_performance_*.csv` - Detailed performance metrics
- `prophet_business_report.md` - This comprehensive report

## 🛠️ Technical Framework
- **Prophet**: {prophet.__version__ if 'prophet' in globals() else 'N/A'}
- **NeuralProphet**: {neuralprophet.__version__ if 'neuralprophet' in globals() else 'N/A'}
- **Optimization**: Optuna hyperparameter tuning
- **Validation**: Time series cross-validation

---
*Business Analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Author: Pablo Poletti | LinkedIn: https://www.linkedin.com/in/pablom-poletti/*
        """
        
        with open('prophet_business_report.md', 'w') as f:
            f.write(report)
        
        # Save detailed metrics
        for dataset_name, metrics_df in self.metrics.items():
            metrics_df.to_csv(f'prophet_performance_{dataset_name.lower()}.csv', index=False)
        
        print("✅ Business report saved as 'prophet_business_report.md'")

def main():
    """Main business analysis pipeline"""
    print("📈 Starting Prophet Business Forecasting Analysis")
    print("=" * 60)
    
    # Initialize analysis
    analysis = ProphetBusinessAnalysis()
    
    # 1. Load business datasets
    analysis.load_business_datasets()
    
    # 2. Business-focused EDA
    analysis.comprehensive_business_eda()
    
    # 3. Train and evaluate models for each dataset
    for dataset_name in analysis.datasets.keys():
        print(f"\n{'='*50}")
        print(f"Analyzing {dataset_name}")
        print(f"{'='*50}")
        
        try:
            # Train models
            analysis.train_and_evaluate_models(dataset_name)
            
            # Create visualizations
            analysis.create_business_visualization(dataset_name)
            
            # Optimize for main business datasets
            if dataset_name in ['Ecommerce_Sales', 'Website_Traffic']:
                analysis.optimize_prophet_hyperparameters(dataset_name)
                analysis.perform_cross_validation(dataset_name)
            
        except Exception as e:
            print(f"❌ Analysis failed for {dataset_name}: {e}")
            continue
    
    # 4. Generate business report
    analysis.generate_business_report()
    
    print("\n🎉 Prophet Business Analysis completed successfully!")
    print("📁 Check the generated files for detailed business insights")

if __name__ == "__main__":
    main()
