# app.py
# Streamlit dashboard for Global Video Game Sales + simple Linear Regression forecasting

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px

st.set_page_config(page_title="Global Video Game Sales Dashboard", layout="wide")

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # 尝试标准列名；兼容不同版本数据集
    rename_map = {
        'Name':'Game', 'Platform':'Platform', 'Year':'Year', 'Year_of_Release': 'Year',
        'Genre':'Genre', 'Publisher':'Publisher',
        'NA_Sales':'NA_Sales', 'EU_Sales':'EU_Sales', 'JP_Sales':'JP_Sales',
        'Other_Sales':'Other_Sales', 'Global_Sales':'Global_Sales',
        'User_Score':'User_Score', 'Rank':'Rank'
    }
    for k,v in list(rename_map.items()):
        if k not in df.columns and v in df.columns:
            continue
    # 统一化：如果存在 Year_of_Release 就改成 Year
    if 'Year' not in df.columns and 'Year_of_Release' in df.columns:
        df['Year'] = df['Year_of_Release']
    # 类型转化
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    sales_cols = ['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']
    for c in sales_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # 填充与清理
    df = df.dropna(subset=['Year'])
    df = df[df['Year'] > 1975]  # 常见数据质量修剪
    # 如果无 Global_Sales，临时合成
    if 'Global_Sales' not in df.columns:
        df['Global_Sales'] = df.get('NA_Sales',0)+df.get('EU_Sales',0)+df.get('JP_Sales',0)+df.get('Other_Sales',0)
    # 保留常见列
    keep = [c for c in ['Game','Platform','Year','Genre','Publisher',
                        'NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales','User_Score','Rank'] if c in df.columns]
    return df[keep].copy()

def kpi_block(df_filtered: pd.DataFrame):
    total_games = df_filtered['Game'].nunique() if 'Game' in df_filtered.columns else len(df_filtered)
    total_sales = df_filtered['Global_Sales'].sum()
    years = (int(df_filtered['Year'].min()), int(df_filtered['Year'].max()))
    c1,c2,c3 = st.columns(3)
    c1.metric("Titles (unique)", f"{total_games:,}")
    c2.metric("Global Sales (M units)", f"{total_sales:,.1f}")
    c3.metric("Year Range", f"{years[0]}–{years[1]}")

def line_global_trend(df_filtered: pd.DataFrame):
    yearly = df_filtered.groupby('Year', as_index=False)['Global_Sales'].sum().sort_values('Year')
    fig = px.line(yearly, x='Year', y='Global_Sales', title="Global Sales Trend by Year")
    st.plotly_chart(fig, use_container_width=True)

def bar_region_compare(df_filtered: pd.DataFrame):
    region_cols = [c for c in ['NA_Sales','EU_Sales','JP_Sales','Other_Sales'] if c in df_filtered.columns]
    if not region_cols:
        st.info("Regional columns not found.")
        return
    agg = df_filtered.groupby('Year', as_index=False)[region_cols].sum().sort_values('Year')
    fig = px.bar(agg, x='Year', y=region_cols, barmode='stack', title="Regional Sales Comparison by Year")
    st.plotly_chart(fig, use_container_width=True)

def pie_genre(df_filtered: pd.DataFrame):
    if 'Genre' not in df_filtered.columns:
        return
    g = df_filtered.groupby('Genre', as_index=False)['Global_Sales'].sum().sort_values('Global_Sales', ascending=False)
    fig = px.pie(g, names='Genre', values='Global_Sales', title="Genre Distribution (Global Sales)")
    st.plotly_chart(fig, use_container_width=True)

def bar_top_platforms(df_filtered: pd.DataFrame, topn=10):
    if 'Platform' not in df_filtered.columns:
        return
    p = df_filtered.groupby('Platform', as_index=False)['Global_Sales'].sum().sort_values('Global_Sales', ascending=False).head(topn)
    fig = px.bar(p, x='Platform', y='Global_Sales', title=f"Top {topn} Platforms by Global Sales")
    st.plotly_chart(fig, use_container_width=True)

def simple_linear_forecast(df: pd.DataFrame, group_by: str, group_value, horizon: int = 5):
    """按 Year-Global_Sales 做线性回归，预测未来 horizon 年"""
    use = df.copy()
    if group_by and group_value and group_by in df.columns:
        use = use[use[group_by] == group_value]
    yearly = use.groupby('Year', as_index=False)['Global_Sales'].sum().sort_values('Year')
    if len(yearly) < 3:
        return None, None, None  # 数据太少
    X = yearly[['Year']].values
    y = yearly['Global_Sales'].values
    lr = LinearRegression()
    lr.fit(X, y)
    yhat = lr.predict(X)
    r2 = r2_score(y, yhat)
    mae = mean_absolute_error(y, yhat)
    # 未来年份
    last_year = int(yearly['Year'].max())
    future_years = np.arange(last_year+1, last_year+horizon+1).reshape(-1,1)
    future_pred = lr.predict(future_years)
    hist_df = yearly.assign(Predicted=yhat)
    future_df = pd.DataFrame({"Year": future_years.flatten(), "Predicted": future_pred})
    return hist_df, future_df, {"R2": r2, "MAE": mae, "coef": float(lr.coef_[0]), "intercept": float(lr.intercept_)}

# ---------------- UI ----------------
st.title("🎮 Global Video Game Sales — Interactive Analytics & Forecast")

# 侧栏：数据载入
st.sidebar.header("Data")
default_path = "vgsales.csv"  # 你可以把 Kaggle 文件命名为 vgsales.csv
data_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if data_file is not None:
    df = load_data(data_file)
else:
    try:
        df = load_data(default_path)
        st.sidebar.caption(f"Using default: {default_path}")
    except Exception as e:
        st.error("Please upload a dataset CSV with common columns like Year/Genre/Platform/... ")
        st.stop()

# 侧栏：过滤器
st.sidebar.header("Filters")
years = (int(df['Year'].min()), int(df['Year'].max()))
year_range = st.sidebar.slider("Year Range", min_value=years[0], max_value=years[1], value=years)
platforms = sorted(df['Platform'].dropna().unique().tolist()) if 'Platform' in df.columns else []
genres = sorted(df['Genre'].dropna().unique().tolist()) if 'Genre' in df.columns else []
publishers = sorted(df['Publisher'].dropna().unique().tolist()) if 'Publisher' in df.columns else []

selected_platforms = st.sidebar.multiselect("Platforms", platforms, default=platforms[:5] if platforms else [])
selected_genres = st.sidebar.multiselect("Genres", genres, default=genres[:5] if genres else [])
selected_publishers = st.sidebar.multiselect("Publishers (optional)", publishers, default=[])

# 过滤数据
mask = (df['Year'].between(year_range[0], year_range[1]))
if selected_platforms and 'Platform' in df.columns:
    mask &= df['Platform'].isin(selected_platforms)
if selected_genres and 'Genre' in df.columns:
    mask &= df['Genre'].isin(selected_genres)
if selected_publishers and 'Publisher' in df.columns and len(selected_publishers) > 0:
    mask &= df['Publisher'].isin(selected_publishers)

df_f = df[mask].copy()
kpi_block(df_f)

# 图表区
col1, col2 = st.columns([1.2, 1])
with col1:
    line_global_trend(df_f)
with col2:
    pie_genre(df_f)

bar_region_compare(df_f)
bar_top_platforms(df_f, topn=10)

st.markdown("---")
st.subheader("🔮 Simple Forecast (Linear Regression)")
mode = st.radio("Forecast by:", ["Overall", "By Genre", "By Platform"], horizontal=True)
horizon = st.slider("Forecast Horizon (years)", 1, 10, 5)

gb_col = None
gb_val = None
if mode == "By Genre" and 'Genre' in df.columns:
    gb_col = "Genre"
    gb_val = st.selectbox("Select Genre", genres)
elif mode == "By Platform" and 'Platform' in df.columns:
    gb_col = "Platform"
    gb_val = st.selectbox("Select Platform", platforms)

hist, future, stats = simple_linear_forecast(df_f, gb_col, gb_val, horizon=horizon)
if hist is None:
    st.info("Not enough data points to build a forecast. Try expanding your filters.")
else:
    # 展示 R2 / MAE
    c1, c2 = st.columns(2)
    c1.metric("R² (in-sample)", f"{stats['R2']:.3f}")
    c2.metric("MAE (in-sample)", f"{stats['MAE']:.3f}")
    # 合并历史+未来并画图
    plot_df = pd.concat([
        hist[['Year','Global_Sales']].rename(columns={'Global_Sales':'Actual'}).assign(Type='Actual'),
        hist[['Year','Predicted']].rename(columns={'Predicted':'Value'}).assign(Type='Fitted'),
        future.rename(columns={'Predicted':'Value'}).assign(Type='Forecast')
    ], ignore_index=True)
    # 整理列
    if 'Actual' in plot_df.columns:
        plot_df = plot_df.melt(id_vars=['Year','Type'], value_vars=['Actual','Value'], var_name='Series', value_name='Global_Sales')
        plot_df = plot_df.dropna(subset=['Global_Sales'])
    fig = px.line(plot_df, x='Year', y='Global_Sales', color='Type', title="Linear Regression — Historical Fit & Forecast")
    st.plotly_chart(fig, use_container_width=True)

st.caption("Data source: TheDevastator (2023). Global Video Game Sales. Kaggle.")
