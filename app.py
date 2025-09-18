"""
📈 Prophet Business Forecasting Dashboard
Professional Business Time Series Forecasting with Meta's Prophet

Author: Pablo Poletti
GitHub: https://github.com/PabloPoletti
Contact: lic.poletti@gmail.com
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Prophet imports
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.plot import plot_cross_validation_metric
    import holidays
except ImportError as e:
    st.error(f"Error importing Prophet: {e}")
    st.stop()

# ML and optimization
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna

# Suppress warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Prophet Business Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .business-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .forecast-insight {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
COLORS = {
    'primary': '#FF6B6B',
    'secondary': '#4ECDC4', 
    'accent': '#45B7D1',
    'success': '#96CEB4',
    'warning': '#FECA57',
    'error': '#FF9FF3'
}

@st.cache_data
def generate_business_data() -> Dict[str, pd.DataFrame]:
    """Generate realistic business time series data"""
    
    datasets = {}
    
    # 1. E-commerce sales with seasonality and promotions
    dates = pd.date_range('2020-01-01', '2024-12-01', freq='D')
    np.random.seed(42)
    
    # Base trend
    trend = np.linspace(10000, 25000, len(dates))
    
    # Seasonal effects
    yearly_season = 3000 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    weekly_season = 2000 * np.sin(2 * np.pi * dates.weekday / 7)
    
    # Holiday effects (Black Friday, Christmas, etc.)
    holiday_boost = np.zeros(len(dates))
    for year in range(2020, 2025):
        # Black Friday (4th Thursday of November)
        bf_date = f"{year}-11-{22 + (3 - datetime(year, 11, 1).weekday()) % 7}"
        if bf_date in dates.strftime('%Y-%m-%d'):
            idx = np.where(dates.strftime('%Y-%m-%d') == bf_date)[0][0]
            holiday_boost[max(0, idx-2):min(len(dates), idx+3)] += 5000
        
        # Christmas season
        christmas_dates = pd.date_range(f'{year}-12-15', f'{year}-12-31', freq='D')
        for christmas_date in christmas_dates:
            if christmas_date in dates:
                idx = dates.get_loc(christmas_date)
                holiday_boost[idx] += 2000
    
    noise = np.random.normal(0, 500, len(dates))
    sales = trend + yearly_season + weekly_season + holiday_boost + noise
    sales = np.maximum(sales, 1000)
    
    datasets['E-commerce_Sales'] = pd.DataFrame({
        'ds': dates,
        'y': sales
    })
    
    # 2. SaaS revenue with growth and churn
    dates_monthly = pd.date_range('2020-01-01', '2024-12-01', freq='MS')
    np.random.seed(123)
    
    # Exponential growth with saturation
    months = np.arange(len(dates_monthly))
    base_revenue = 50000 * (1 - np.exp(-months / 12)) + 10000
    seasonal_effect = 5000 * np.sin(2 * np.pi * months / 12)
    noise_monthly = np.random.normal(0, 2000, len(dates_monthly))
    
    revenue = base_revenue + seasonal_effect + noise_monthly
    
    datasets['SaaS_Revenue'] = pd.DataFrame({
        'ds': dates_monthly,
        'y': revenue
    })
    
    # 3. Retail foot traffic
    dates_hourly = pd.date_range('2023-01-01', '2024-06-01', freq='D')
    np.random.seed(456)
    
    # Weekly patterns (higher on weekends)
    base_traffic = 1000
    weekly_pattern = 300 * np.sin(2 * np.pi * dates_hourly.weekday / 7)
    seasonal_pattern = 200 * np.sin(2 * np.pi * dates_hourly.dayofyear / 365.25)
    weather_effect = np.random.normal(0, 100, len(dates_hourly))
    
    traffic = base_traffic + weekly_pattern + seasonal_pattern + weather_effect
    traffic = np.maximum(traffic, 100)
    
    datasets['Retail_Traffic'] = pd.DataFrame({
        'ds': dates_hourly,
        'y': traffic
    })
    
    return datasets

def optimize_prophet_params(data: pd.DataFrame) -> Dict:
    """Optimize Prophet hyperparameters using Optuna"""
    
    def objective(trial):
        # Suggest hyperparameters
        changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True)
        seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 0.01, 10.0, log=True)
        holidays_prior_scale = trial.suggest_float('holidays_prior_scale', 0.01, 10.0, log=True)
        seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
        
        try:
            # Create model
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                seasonality_mode=seasonality_mode,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            
            # Split data for validation
            train_size = int(len(data) * 0.8)
            train_data = data.iloc[:train_size]
            val_data = data.iloc[train_size:]
            
            # Fit and predict
            model.fit(train_data)
            future = model.make_future_dataframe(periods=len(val_data), freq='D')
            forecast = model.predict(future)
            
            # Calculate error
            predictions = forecast['yhat'].iloc[-len(val_data):].values
            actual = val_data['y'].values
            error = mean_absolute_error(actual, predictions)
            
            return error
            
        except Exception:
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    
    return study.best_params

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Prophet Business Forecasting</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="forecast-insight">
    📈 <strong>Professional Business Forecasting with Meta's Prophet</strong><br>
    Intuitive time series forecasting designed for business applications with automatic seasonality and holiday detection
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Business Configuration")
        
        # Dataset selection
        datasets = generate_business_data()
        dataset_name = st.selectbox("📊 Select Business Dataset:", list(datasets.keys()))
        data = datasets[dataset_name]
        
        # Prophet parameters
        st.markdown("### 🔮 Prophet Setup")
        forecast_periods = st.slider("📅 Forecast Periods:", 30, 365, 90)
        
        # Seasonality settings
        st.markdown("### 📈 Seasonality")
        yearly_seasonality = st.checkbox("📅 Yearly Seasonality", value=True)
        weekly_seasonality = st.checkbox("📊 Weekly Seasonality", value=True)
        daily_seasonality = st.checkbox("🕐 Daily Seasonality", value=False)
        
        seasonality_mode = st.selectbox("🔄 Seasonality Mode:", 
                                      ['additive', 'multiplicative'])
        
        # Advanced options
        st.markdown("### ⚙️ Advanced Options")
        enable_holidays = st.checkbox("🎄 Holiday Effects", value=True)
        optimize_hyperparams = st.checkbox("🔧 Optimize Parameters", value=False)
        cross_validate = st.checkbox("✅ Cross Validation", value=False)
        
        # Confidence intervals
        interval_width = st.slider("📊 Confidence Interval:", 0.8, 0.99, 0.95)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Data visualization
        st.subheader("📊 Business Data Overview")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['ds'],
            y=data['y'],
            mode='lines',
            name='Historical Data',
            line=dict(color=COLORS['primary'], width=2)
        ))
        
        fig.update_layout(
            title=f"Business Dataset: {dataset_name}",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data insights
        st.markdown("### 📈 Business Insights")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("📊 Data Points", len(data))
        with col_b:
            growth_rate = ((data['y'].iloc[-30:].mean() / data['y'].iloc[:30].mean()) - 1) * 100
            st.metric("📈 Growth Rate", f"{growth_rate:.1f}%")
        with col_c:
            volatility = data['y'].std() / data['y'].mean() * 100
            st.metric("📊 Volatility", f"{volatility:.1f}%")
        with col_d:
            avg_value = data['y'].mean()
            st.metric("💰 Average Value", f"{avg_value:,.0f}")
    
    with col2:
        # Business summary
        st.subheader("💼 Business Summary")
        
        # Trend analysis
        recent_trend = data['y'].tail(30).mean() - data['y'].head(30).mean()
        if recent_trend > 0:
            st.success(f"📈 Upward trend: +{recent_trend:,.0f}")
        else:
            st.error(f"📉 Downward trend: {recent_trend:,.0f}")
        
        # Seasonality strength
        if len(data) > 365:
            seasonal_strength = data['y'].rolling(30).std().mean()
            st.info(f"🔄 Seasonal variation: {seasonal_strength:,.0f}")
        
        # Business cycle
        max_val = data['y'].max()
        min_val = data['y'].min()
        range_pct = ((max_val - min_val) / data['y'].mean()) * 100
        st.warning(f"📊 Value range: {range_pct:.1f}%")
    
    # Forecasting section
    if st.button("🚀 Generate Business Forecast", type="primary"):
        st.markdown("---")
        st.subheader("🔮 Business Forecasting Results")
        
        with st.spinner("Training Prophet model and generating forecasts..."):
            
            # Prepare model parameters
            model_params = {
                'yearly_seasonality': yearly_seasonality,
                'weekly_seasonality': weekly_seasonality,
                'daily_seasonality': daily_seasonality,
                'seasonality_mode': seasonality_mode,
                'interval_width': interval_width
            }
            
            # Add holidays if enabled
            if enable_holidays:
                country_holidays = holidays.US()  # Can be customized
                model_params['holidays'] = pd.DataFrame([
                    {'holiday': name, 'ds': date} 
                    for date, name in country_holidays.items()
                    if date >= data['ds'].min() and date <= data['ds'].max() + timedelta(days=forecast_periods)
                ])
            
            # Optimize parameters if requested
            if optimize_hyperparams:
                with st.spinner("Optimizing hyperparameters..."):
                    best_params = optimize_prophet_params(data)
                    model_params.update(best_params)
                    st.success(f"✅ Optimization completed! Best params: {best_params}")
            
            # Create and train model
            model = Prophet(**model_params)
            model.fit(data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_periods, freq='D')
            forecast = model.predict(future)
            
            # Plot forecast
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=data['ds'],
                y=data['y'],
                mode='lines',
                name='Historical',
                line=dict(color=COLORS['primary'], width=2)
            ))
            
            # Forecast
            future_data = forecast[forecast['ds'] > data['ds'].max()]
            fig.add_trace(go.Scatter(
                x=future_data['ds'],
                y=future_data['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color=COLORS['accent'], width=3, dash='dash')
            ))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=future_data['ds'],
                y=future_data['yhat_upper'],
                mode='lines',
                name='Upper Confidence',
                line=dict(color='rgba(255,0,0,0.3)', width=1),
                fill=None
            ))
            
            fig.add_trace(go.Scatter(
                x=future_data['ds'],
                y=future_data['yhat_lower'],
                mode='lines',
                name='Lower Confidence',
                line=dict(color='rgba(255,0,0,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)'
            ))
            
            fig.update_layout(
                title="📈 Business Forecast with Prophet",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Business metrics
            st.markdown("### 💼 Business Forecast Metrics")
            
            current_value = data['y'].iloc[-1]
            forecast_end = future_data['yhat'].iloc[-1]
            total_growth = ((forecast_end - current_value) / current_value) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Current Value", f"{current_value:,.0f}")
            with col2:
                st.metric("🔮 Forecast End", f"{forecast_end:,.0f}", 
                         f"{total_growth:+.1f}%")
            with col3:
                confidence_range = future_data['yhat_upper'].iloc[-1] - future_data['yhat_lower'].iloc[-1]
                st.metric("📊 Confidence Range", f"±{confidence_range/2:,.0f}")
            with col4:
                trend_strength = forecast['trend'].iloc[-1] - forecast['trend'].iloc[0]
                st.metric("📈 Trend Strength", f"{trend_strength:,.0f}")
            
            # Forecast table
            st.markdown("### 📋 Detailed Forecast")
            forecast_display = future_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
            forecast_display = forecast_display.round(0)
            
            st.dataframe(forecast_display, hide_index=True, height=300)
            
            # Download forecast
            csv = forecast_display.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast",
                data=csv,
                file_name=f"prophet_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Cross validation results
            if cross_validate:
                st.markdown("### ✅ Cross Validation Results")
                with st.spinner("Performing cross validation..."):
                    try:
                        cv_results = cross_validation(
                            model, 
                            initial='730 days', 
                            period='180 days', 
                            horizon='90 days'
                        )
                        performance = performance_metrics(cv_results)
                        
                        st.dataframe(performance.round(3))
                        
                    except Exception as e:
                        st.warning(f"Cross validation failed: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    📈 <strong>Prophet Business Forecasting</strong> | 
    Built with Meta's Prophet | 
    <a href="https://github.com/PabloPoletti" target="_blank">GitHub</a> | 
    <a href="mailto:lic.poletti@gmail.com">Contact</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
