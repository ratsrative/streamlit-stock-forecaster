import yfinance as yf
import streamlit as st
import datetime
import pandas as pd
import numpy as np
import random
from PIL import Image
st.write("""
### User manual
* you can select any of the companies that are components of the **:red[NIFTY 500]** index
* you can select the  Forecasting Models of Your Interest
""")
image1 = Image.open('./Pages/Stock Market Analysis Header.png')
st.image(image1)
@st.cache_data
def get_nifty500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_500", match='Nifty 500 List')[0]
    df.columns = df.iloc[0]
    df["Symbol"] = df["Symbol"] + '.NS'
    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(zip(df["Symbol"], df["Company Name"]))
    return tickers, tickers_companies_dict

available_tickers, tickers_companies_dict = get_nifty500_components()

st.sidebar.header("Forecasting Models")

tickers = st.sidebar.selectbox(
    "Ticker",
    available_tickers,
    format_func=tickers_companies_dict.get
)
start_date = st.sidebar.date_input(
    "Start date",
    datetime.date(2019, 1, 1)
)
end_date = st.sidebar.date_input(
    "End date",
    datetime.date.today()
)

if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")

# Download data
@st.cache_data
def load_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start_date, end_date)
        return data
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return None

forecast_models = st.sidebar.selectbox("Select Forecast Models", options=["Prophet", "AutoML"])

# Get the Data from Yahoo finance
df = load_data(tickers, start_date, end_date)


if df is not None and not df.empty:
    df = df.reset_index()
    df = df[['Date', 'Adj Close']]
else:
    st.warning("No data available for the selected stock and date range. Please adjust your selection.")

####################################PROPHET FORECAST MODEL##############################################################

from prophet import Prophet
import itertools
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.plot import plot_plotly, plot_components_plotly

if forecast_models == 'Prophet':

    if df is not None:
        df.rename(columns={'Date':'ds','Adj Close':'y'},inplace=True)

    initial_days = st.sidebar.number_input("Initial days", min_value=7, max_value=756, value=200, step=10)
    period_days = st.sidebar.number_input("Period days", min_value=7, max_value=252, value=10, step=10)
    horizon_days = st.sidebar.number_input("Horizon days", min_value=7, max_value=252, value=7, step=1)
    future_period_days = st.sidebar.number_input("Future Forecast Period (days)", min_value=1, max_value=252, value=7,
                                           step=1)
    # Set up parameter grid
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.05, 0.08, 0.5],
        'seasonality_prior_scale': [0.01, 1, 5, 10, 12],
        'holidays_prior_scale': [0.01, 0.1, 1, 10],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    # Reduce the number of combinations
    selected_params = random.sample(all_params, min(10, len(all_params)))

    from dask.distributed import Client

    #Specify the scheduler explicitly
    client = Client(processes=False)
    mapes = []
    with st.spinner('Tuning parameters and fitting the model...'):
    # Use cross validation to evaluate all parameters
        for params in selected_params:
            # Fit a model using one parameter combination
            m = Prophet(**params).fit(df)
            # Cross-validation
            df_cv = cross_validation(m, initial=f'{initial_days} days', period=f'{period_days} days', horizon=f'{horizon_days} days')
            # Model performance
            df_p = performance_metrics(df_cv, rolling_window=1)
            # Save model performance metrics
            mapes.append(df_p['mape'].values[0])

    # Tuning results
    tuning_results = pd.DataFrame(selected_params)
    tuning_results['mape'] = mapes

    # Find the best parameters
    best_params = selected_params[np.argmin(mapes)]
    # Fit the model using the best parameters
    auto_model = Prophet(changepoint_range=0.9,
                         changepoint_prior_scale=best_params['changepoint_prior_scale'],
                         holidays_prior_scale=best_params['holidays_prior_scale'],
                         seasonality_prior_scale=best_params['seasonality_prior_scale'],
                         seasonality_mode=best_params['seasonality_mode']).add_country_holidays(country_name="IND")

    # Fit the model on the training dataset
    auto_model.fit(df)

    # Cross validation
    auto_model_cv = cross_validation(auto_model,initial=f'{initial_days} days', period=f'{period_days} days',
                                     horizon=f'{horizon_days} days',
                                     parallel="dask")
    # Model performance metrics
    auto_model_p = performance_metrics(auto_model_cv, rolling_window=1)
    Mean_Percentage_error = round(auto_model_p['mape'][0],3)

    # Extend the dataframe with future dates
    future = auto_model.make_future_dataframe(periods=future_period_days)

    # Make predictions for the extended dataframe
    forecast = auto_model.predict(future)
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    st.markdown(f"**Mean Percentage Error: :green[{Mean_Percentage_error}]**")

    # plot the forecast
    st.subheader("Prophet Forecast Model with future dates")
    fig1 = plot_plotly(auto_model, forecast)
    st.plotly_chart(fig1)

    # # plot components only if 'trend' is present in the forecast
    components_available = 'trend' in auto_model.component_modes['additive']
    if components_available:
        fig2= plot_components_plotly(auto_model, forecast)
        st.plotly_chart(fig2)
    else:
         st.warning("The 'trend' component is not available for plotting.")

    st.subheader("Prophet Forecast Model on Cross Validation Dataset")
    fig3 = plot_plotly(auto_model, auto_model_cv)
    st.plotly_chart(fig3)
    # plot components only if 'trend' is present in the forecast
    components_available = 'trend' in auto_model.component_modes['additive']
    if components_available:
        fig4 = plot_components_plotly(auto_model, auto_model_cv)
        st.plotly_chart(fig4)
    else:
        st.warning("The 'trend' component is not available for plotting.")

    # Close the Dask client
    client.close()

##################################################AutoML################################################################
elif forecast_models == 'AutoML':
    if df is not None:
        df['Date'] = df['Date'].astype(str)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.groupby('Date').sum()
        df = df.asfreq(freq='D').ffill()
        # import the  framework
        from pycaret.time_series import TSForecastingExperiment
        import xgboost
        from packaging import version
        xgboost_version = version.parse(xgboost.__version__)
        import io
        with st.spinner('Tuning parameters and fitting the model...'):
            if xgboost_version < version.parse("1.1.0"):
            # Handle the case where XGBoost version is less than 1.1.0
            # This might involve updating XGBoost or using a workaround
                xgboost_version = "2.0.3"
                from xgboost import XGBRegressor
                xgb_model = XGBRegressor()
            else:
                exp = TSForecastingExperiment()
                exp.setup(df,fh=5,fold=5,session_id=42)
                best_pipelines = exp.compare_models(
                    sort='MAPE',turbo=False,n_select=3
                )
                metrics1 = exp.pull()[:3]
                st.write(metrics1)
                # Tune the best pipelines:
                best_pipelines_tuned = [exp.tune_model(model) for model in best_pipelines]
                metrics2= exp.pull()
                st.subheader('The forecast is Made by Blending the Above 3 Fine Tuned Models')
                st.write(metrics2)
                # Blended Model
                blended_model = exp.blend_models(
                    best_pipelines_tuned, method="gmean")
                # Create the predictions using the blended model and plot the forecasts:,
                horizon = st.sidebar.number_input("Future Forecast Period (days)", min_value=1, max_value=60, value=5,
                                  step=1)
                y_pred = exp.predict_model(blended_model,fh=horizon)
                # final Model
                final_model = exp.finalize_model(blended_model)
                y_pred_new = exp.predict_model(final_model,fh=horizon)
                y_pred_new.reset_index(inplace=True)
                y_pred_new.rename(columns={'index':'Date','y_pred':'Price Forecast'},inplace=True)
                st.dataframe(y_pred_new)
                #Plot Graph
                exp.plot_model(estimator=final_model,display_format='streamlit')
