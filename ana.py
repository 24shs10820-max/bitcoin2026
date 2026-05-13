import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# 페이지 설정
st.set_page_config(page_title="비트코인 데이터 분석 대시보드", layout="wide")

@st.cache_data
def load_data(file_path):
    # CSV 파일 읽기 (업로드된 데이터 확인 결과 구분자가 ';' 임)
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path, delimiter=';')
    
    # 'timeOpen' 컬럼을 datetime 객체로 변환
    df['timeOpen'] = pd.to_datetime(df['timeOpen'])
    
    # 최신 데이터가 위로 오는 경우가 많으므로 날짜순으로 정렬
    df = df.sort_values('timeOpen')
    
    # 분석에 필요한 수치형 컬럼 변환 (오류 발생 시 NaN 처리)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'marketCap']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 결측치 제거
    df = df.dropna(subset=['close', 'timeOpen'])
    
    return df

# 내일 가격 예측 함수 (선형회귀 모델)
def predict_next_day(df):
    # 날짜를 숫자로 변환 (학습을 위해)
    df = df.copy()
    df['date_num'] = df['timeOpen'].map(datetime.toordinal)
    
    X = df[['date_num']].values
    y = df['close'].values
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X, y)
    
    # 내일 날짜 계산
    next_day_num = np.array([[df['date_num'].max() + 1]])
    prediction = model.predict(next_day_num)[0]
    
    return prediction

# 메인 타이틀
st.title("📊 비트코인(BTC) 가격 분석 및 내일 예측 대시보드")

# 데이터 로드
file_name = 'coin.csv'
df = load_data(file_name)

if df is not None:
    # 사이드바: 설정 및 필터링
    st.sidebar.header("🔍 데이터 필터링")
    
    min_date = df['timeOpen'].min().to_pydatetime()
    max_date = df['timeOpen'].max().to_pydatetime()
    
    # 날짜 범위 슬라이더
    date_range = st.sidebar.slider(
        "분석 기간 선택",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    
    # 선택된 기간으로 데이터 필터링
    filtered_df = df[(df['timeOpen'] >= date_range[0]) & (df['timeOpen'] <= date_range[1])].copy()

    # 1. 상단 주요 지표 (Metrics)
    if not filtered_df.empty:
        latest_data = filtered_df.iloc[-1]
        
        # 전일 데이터 찾기 (변동폭 계산용)
        if len(filtered_df) > 1:
            prev_data = filtered_df.iloc[-2]
            price_diff = latest_data['close'] - prev_data['close']
            price_diff_pct = (price_diff / prev_data['close']) * 100
        else:
            price_diff, price_diff_pct = 0, 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("현재 종가", f"{latest_data['close']:,.0f} ₩", f"{price_diff:,.0f} ({price_diff_pct:.2f}%)")
        m2.metric("기간 최고가", f"{filtered_df['high'].max():,.0f} ₩")
        m3.metric("기간 최저가", f"{filtered_df['low'].min():,.0f} ₩")
        m4.metric("평균 거래량", f"{filtered_df['volume'].mean():,.0e}")

    st.markdown("---")

    # 2. 내일 가격 예측 섹션 (Linear Regression)
    st.subheader("🔮 AI 내일 가격 예측 (선형회귀 모델)")
    
    if len(filtered_df) > 5:  # 최소한의 데이터가 있을 때만 예측
        predicted_price = predict_next_day(filtered_df)
        current_price = filtered_df.iloc[-1]['close']
        change = predicted_price - current_price
        change_pct = (change / current_price) * 100
        
        p_col1, p_col2 = st.columns([1, 2])
        
        with p_col1:
            st.write(f"**대상 날짜:** {(filtered_df['timeOpen'].max() + timedelta(days=1)).strftime('%Y-%m-%d')}")
            if change > 0:
                st.success(f"📈 **상승 예측**: 약 {predicted_price:,.0f} ₩")
                st.write(f"현재가 대비 약 **{change_pct:+.2f}%** 상승할 것으로 보입니다.")
            else:
                st.error(f"📉 **하락 예측**: 약 {predicted_price:,.0f} ₩")
                st.write(f"현재가 대비 약 **{change_pct:+.2f}%** 하락할 것으로 보입니다.")
            
            st.caption("※ 선형회귀 모델은 과거 추세만을 반영하므로 실제 시장 변동과는 차이가 클 수 있습니다. 투자 참고용으로만 사용하세요.")
            
        with p_col2:
            # 예측 시각화 (현재 데이터 끝에 예측 점 추가)
            pred_fig = go.Figure()
            pred_fig.add_trace(go.Scatter(x=filtered_df['timeOpen'].tail(30), y=filtered_df['close'].tail(30), name='최근 가격'))
            pred_fig.add_trace(go.Scatter(
                x=[filtered_df['timeOpen'].max() + timedelta(days=1)], 
                y=[predicted_price], 
                mode='markers', 
                marker=dict(size=12, color='red' if change < 0 else 'green'),
                name='내일 예측치'
            ))
            pred_fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), template='plotly_white')
            st.plotly_chart(pred_fig, use_container_width=True)
    else:
        st.info("예측을 위해 더 많은 데이터 기간을 선택해 주세요.")

    st.markdown("---")

    # 3. 메인 가격 차트 (캔들스틱)
    st.subheader("📈 캔들스틱 및 이동평균선 (MA)")
    
    # 이동평균선 계산 (20일, 50일)
    filtered_df['MA20'] = filtered_df['close'].rolling(window=20).mean()
    filtered_df['MA50'] = filtered_df['close'].rolling(window=50).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=filtered_df['timeOpen'],
        open=filtered_df['open'],
        high=filtered_df['high'],
        low=filtered_df['low'],
        close=filtered_df['close'],
        name='BTC 가격'
    ))
    fig.add_trace(go.Scatter(x=filtered_df['timeOpen'], y=filtered_df['MA20'], name='MA20', line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=filtered_df['timeOpen'], y=filtered_df['MA50'], name='MA50', line=dict(color='royalblue', width=1)))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        height=500,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. 하단 섹션: 거래량 및 데이터 상세보기
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📊 거래량 추이")
        vol_fig = go.Figure(go.Bar(
            x=filtered_df['timeOpen'],
            y=filtered_df['volume'],
            marker_color='lightslategrey',
            name='거래량'
        ))
        vol_fig.update_layout(template='plotly_white', height=350, margin=dict(t=10))
        st.plotly_chart(vol_fig, use_container_width=True)

    with col_right:
        st.subheader("📋 최근 데이터 (Top 10)")
        display_df = filtered_df[['timeOpen', 'close', 'volume']].sort_values('timeOpen', ascending=False).head(10)
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    # 데이터 내보내기 기능
    st.sidebar.markdown("---")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 필터링된 데이터 다운로드",
        data=csv,
        file_name='bitcoin_filtered_data.csv',
        mime='text/csv',
    )

else:
    st.warning(f"'{file_name}' 파일을 찾을 수 없습니다. 파이썬 스크립트와 동일한 폴더에 파일을 위치시켜 주세요.")
    st.info("파일 이름이 정확한지 확인해 보세요: 'coin.csv'")
