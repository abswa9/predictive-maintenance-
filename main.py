import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("WORKING")
st.write("OK")
BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "train_FD001.txt"
TEST_PATH = BASE_DIR / "test_FD001.txt"
RUL_PATH = BASE_DIR / "RUL_FD001.txt"

FEATURES = ['cycle', 'op1', 'op2', 'op3'] + [f'sensor{i}' for i in range(1, 22)]
COLUMNS = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor{i}' for i in range(1, 22)]


@st.cache_data
def load_data():
    df_train = pd.read_csv(TRAIN_PATH, sep=r'\s+', header=None)
    df_test = pd.read_csv(TEST_PATH, sep=r'\s+', header=None)
    df_rul = pd.read_csv(RUL_PATH, sep=r'\s+', header=None)

    df_train.columns = COLUMNS
    df_test.columns = COLUMNS
    df_rul.columns = ['RUL']

    max_cycle = df_train.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']
    df_train = df_train.merge(max_cycle, on='engine_id', how='left')
    df_train['RUL'] = df_train['max_cycle'] - df_train['cycle']
    df_train['RUL_capped'] = df_train['RUL'].clip(upper=125)

    return df_train, df_test, df_rul


@st.cache_resource
def train_model(df_train: pd.DataFrame):
    X = df_train[FEATURES]
    y = df_train['RUL_capped']
    groups = df_train['engine_id']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    importance_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return model, X_val, y_val, y_pred, mae, rmse, importance_df


def plot_real_vs_predicted(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_true.reset_index(drop=True), y_pred, alpha=0.4)
    ax.plot([0, 125], [0, 125], linestyle='--')
    ax.set_xlabel('Real RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title('Real vs Predicted RUL')
    return fig


def plot_feature_importance(importance_df):
    top10 = importance_df.head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top10['Feature'], top10['Importance'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Most Important Features')
    return fig


def plot_engine_curve(df_train, model, engine_id):
    engine_data = df_train[df_train['engine_id'] == engine_id].copy()
    engine_data = engine_data.sort_values('cycle')
    preds = model.predict(engine_data[FEATURES])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(engine_data['cycle'], engine_data['RUL_capped'], label='Real RUL')
    ax.plot(engine_data['cycle'], preds, label='Predicted RUL')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('RUL')
    ax.set_title(f'Engine {engine_id} - Real vs Predicted RUL')
    ax.legend()
    return fig


def risk_label(predicted_rul: float):
    if predicted_rul <= 20:
        return 'High risk - maintenance needed soon'
    if predicted_rul <= 50:
        return 'Medium risk - monitor closely'
    return 'Low risk - normal condition'


try:
    df_train, df_test, df_rul = load_data()
    model, X_val, y_val, y_pred, mae, rmse, importance_df = train_model(df_train)
except Exception as e:
    st.error(f"Data or model loading failed: {e}")
    st.stop()

st.subheader('Model Performance')
col1, col2 = st.columns(2)
col1.metric('MAE', f'{mae:.2f}')
col2.metric('RMSE', f'{rmse:.2f}')

col3, col4 = st.columns(2)
with col3:
    st.pyplot(plot_real_vs_predicted(y_val, y_pred))
with col4:
    st.pyplot(plot_feature_importance(importance_df))

st.subheader('Engine-Level Analysis')
engine_ids = sorted(df_train['engine_id'].unique().tolist())
selected_engine = st.selectbox('Select engine', engine_ids, index=0)
st.pyplot(plot_engine_curve(df_train, model, selected_engine))

st.subheader('Manual Prediction')
st.write('Adjust the values below to estimate the remaining useful life for an engine state.')

with st.form('manual_prediction_form'):
    input_data = {}
    col_left, col_right = st.columns(2)

    with col_left:
        input_data['cycle'] = st.number_input('cycle', min_value=1, value=50)
        input_data['op1'] = st.number_input('op1', value=0.0, format='%.4f')
        input_data['op2'] = st.number_input('op2', value=0.0, format='%.4f')
        input_data['op3'] = st.number_input('op3', value=100.0, format='%.4f')
        for i in range(1, 11):
            median_val = float(df_train[f'sensor{i}'].median())
            input_data[f'sensor{i}'] = st.number_input(f'sensor{i}', value=median_val, format='%.4f')

    with col_right:
        for i in range(11, 22):
            median_val = float(df_train[f'sensor{i}'].median())
            input_data[f'sensor{i}'] = st.number_input(f'sensor{i}', value=median_val, format='%.4f')

    submitted = st.form_submit_button('Predict RUL')

if submitted:
    input_df = pd.DataFrame([input_data])[FEATURES]
    predicted_rul = float(model.predict(input_df)[0])
    predicted_rul = max(0.0, predicted_rul)

    st.success(f'Predicted RUL: {predicted_rul:.2f} cycles')
    st.info(risk_label(predicted_rul))

    st.write('Input snapshot')
    st.dataframe(input_df)

st.subheader('Top Features')
st.dataframe(importance_df.reset_index(drop=True))

st.caption('Tip: Run this app with: streamlit run app.py')
           