# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 23:18:50 2026

@author: e010
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# ==========================================
# 1. 網頁基本設定與載入模型
# ==========================================
st.set_page_config(page_title="失智症 5 年風險預測系統", layout="wide")

# 設定 Matplotlib 支援中文顯示 (依據您的作業系統可能需要微調)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

@st.cache_resource
def load_model():
    with open('cox_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    cph_model = load_model()
except FileNotFoundError:
    st.error("找不到模型檔案 'cox_model.pkl'。請確認檔案是否存在於同一目錄下！")
    st.stop()

# ==========================================
# 2. 側邊欄：建立互動式輸入表單
# ==========================================
st.sidebar.header("📝 輸入病患臨床資料")

# 定義一個小函數來將「是/否」轉換回模型需要的 1/0
def yes_no_to_int(val):
    return 1 if val == "是 (1)" else 0

with st.sidebar.form("patient_form"):
    st.subheader("基本資料")
    age = st.number_input("年齡 (Age)", min_value=40, max_value=120, value=70)
    gender = st.selectbox("性別 (Gender)", ["男 (0)", "女 (1)"]) # 依據您原始編碼調整
    education = st.number_input("教育年數/程度 (Education)", min_value=0, max_value=30, value=12)
    
    st.subheader("慢性病史")
    htn = st.selectbox("高血壓 (HTN)", ["否 (0)", "是 (1)"])
    dm = st.selectbox("糖尿病 (DM)", ["否 (0)", "是 (1)"])
    cad = st.selectbox("冠心病 (CAD)", ["否 (0)", "是 (1)"])
    cva = st.selectbox("中風病史 (CVA)", ["否 (0)", "是 (1)"])
    
    st.subheader("用藥狀況")
    anti_htn = st.selectbox("降血壓藥 (Anti_HTN)", ["否 (0)", "是 (1)"])
    anti_dm = st.selectbox("降血糖藥 (Anti_DM)", ["否 (0)", "是 (1)"])
    anti_plt = st.selectbox("抗血小板藥 (AntiPLT)", ["否 (0)", "是 (1)"])
    antidementia = st.selectbox("失智症藥物 (Antidementia)", ["否 (0)", "是 (1)"])
    
    st.subheader("臨床評估與抽血數值")
    ldl = st.number_input("低密度脂蛋白 (LDL)", min_value=10, max_value=300, value=100)
    hdl = st.number_input("高密度脂蛋白 (HDL)", min_value=10, max_value=150, value=50)
    tg = st.number_input("三酸甘油脂 (TG)", min_value=10, max_value=1000, value=150)
    casi = st.number_input("CASI 認知分數", min_value=0.0, max_value=100.0, value=80.0)
    haiadl = st.number_input("HAIADL 日常活動分數", min_value=0, max_value=50, value=5)
    npi_sb = st.number_input("NPI_SB 精神行為症狀分數", min_value=0, max_value=144, value=2)
    cfs = st.number_input("CFS 臨床衰弱量表", min_value=1, max_value=9, value=3)

    # 提交按鈕
    submit_button = st.form_submit_button(label="開始預測")

# ==========================================
# 3. 主畫面：處理資料與視覺化預測結果
# ==========================================
st.title("🧠 個人化失智症發生風險預測系統")
st.write("請在左側面板輸入病患的詳細資料，按下「開始預測」後，系統將自動計算未來 1 到 5 年的罹病機率。")

if submit_button:
    # 將使用者的輸入整理成 DataFrame (欄位名稱必須與訓練模型時一模一樣)
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [1 if gender == "女 (1)" else 0],
        'Education': [education],
        'HTN': [yes_no_to_int(htn)],
        'DM': [yes_no_to_int(dm)],
        'CAD': [yes_no_to_int(cad)],
        'CVA': [yes_no_to_int(cva)],
        'Anti_HTN': [yes_no_to_int(anti_htn)],
        'Anti_DM': [yes_no_to_int(anti_dm)],
        'AntiPLT': [yes_no_to_int(anti_plt)],
        'Antidementia': [yes_no_to_int(antidementia)],
        'TG': [tg],
        'LDL': [ldl],
        'HDL': [hdl],
        'CASI': [casi],
        'HAIADL': [haiadl],
        'NPI_SB': [npi_sb],
        'CFS': [cfs]
    })
    
    st.markdown("---")
    st.subheader("📊 預測結果報告")
    
    # 執行模型預測
    time_points = [1, 2, 3, 4, 5]
    survival_probs = cph_model.predict_survival_function(input_data, times=time_points)
    
    # 計算失智發生機率並轉換為百分比
    dementia_probs = 1 - survival_probs
    dementia_probs_percent = (dementia_probs * 100).round(2)
    
    # 將結果分成兩欄顯示 (左邊文字表格，右邊圖表)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**未來 1~5 年累積發生機率：**")
        # 整理顯示用的表格
        display_df = dementia_probs_percent.copy()
        display_df.columns = ["預測罹病機率 (%)"]
        display_df.index = [f"第 {y} 年" for y in time_points]
        st.table(display_df)
        
        # 簡單的臨床警示邏輯 (可依需求調整)
        risk_year_5 = display_df.iloc[-1, 0]
        if risk_year_5 > 50:
            st.error(f"⚠️ **高風險警示**：五年內罹病機率高達 {risk_year_5}%，建議密切追蹤並介入治療。")
        elif risk_year_5 > 20:
            st.warning(f"⚠️ **中度風險**：五年內罹病機率為 {risk_year_5}%，建議定期回診評估。")
        else:
            st.success(f"✅ **低風險**：五年內罹病機率為 {risk_year_5}%，請維持良好生活習慣。")

    with col2:
        # 繪製風險軌跡圖
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time_points, dementia_probs.iloc[:, 0], marker='o', color='#d62728', lw=2, markersize=8)
        
        ax.set_title('個人化 5 年風險軌跡', fontsize=16, fontweight='bold', pad=10)
        ax.set_xlabel('未來追蹤時間 (年)', fontsize=12)
        ax.set_ylabel('累積發生機率 (0~1.0)', fontsize=12)
        
        ax.set_xticks(time_points)
        ax.set_xticklabels([f'第 {y} 年' for y in time_points])
        ax.set_ylim([0, 1.05])
        ax.grid(alpha=0.3, linestyle='--')
        
        st.pyplot(fig)