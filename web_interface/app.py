import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import time

# 設定頁面配置
st.set_page_config(
    page_title="多模態AI智慧城市交通優化系統",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #1f77b4, #ff7f0e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 1rem 0;
}
.status-good { color: #28a745; }
.status-warning { color: #ffc107; }
.status-danger { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """載入示例數據"""
    np.random.seed(42)
    
    # 交通流量數據
    timestamps = pd.date_range(start='2025-06-10 00:00', periods=24, freq='H')
    traffic_data = pd.DataFrame({
        'timestamp': timestamps,
        'intersection_1': np.random.normal(150, 30, 24),
        'intersection_2': np.random.normal(180, 40, 24),
        'intersection_3': np.random.normal(120, 25, 24),
        'intersection_4': np.random.normal(200, 50, 24)
    })
    
    return traffic_data

def main():
    # 主標題
    st.markdown('<h1 class="main-header">🚦 多模態AI智慧城市交通優化系統</h1>', unsafe_allow_html=True)
    
    # 側邊欄控制面板
    with st.sidebar:
        st.header("🎛️ 系統控制面板")
        
        # 系統狀態
        st.subheader("系統狀態")
        system_status = st.selectbox("選擇系統模式", ["自動模式", "手動模式", "維護模式"])
        
        if system_status == "自動模式":
            st.success("✅ AI系統正在自動優化交通流量")
        elif system_status == "手動模式":
            st.warning("⚠️ 手動控制模式已啟用")
        else:
            st.error("🔧 系統維護中")
        
        # 多模態數據源
        st.subheader("多模態數據源")
        data_sources = {
            "交通攝影機": st.checkbox("交通攝影機", value=True),
            "GPS軌跡": st.checkbox("GPS軌跡數據", value=True),
            "氣象資訊": st.checkbox("氣象資訊", value=True),
            "社群媒體": st.checkbox("社群媒體文本", value=False),
            "感測器": st.checkbox("IoT感測器", value=True)
        }
        
        # 強化學習參數
        st.subheader("強化學習參數")
        learning_rate = st.slider("學習率", 0.0001, 0.01, 0.0003, format="%.4f")
        epsilon = st.slider("探索率", 0.01, 1.0, 0.1, format="%.2f")
        
        # 即時控制
        st.subheader("即時控制")
        if st.button("🚀 啟動AI優化", type="primary"):
            st.success("AI優化已啟動！")
        
        if st.button("⏹️ 停止優化"):
            st.info("AI優化已停止")
        
        if st.button("🔄 重置系統"):
            st.warning("系統正在重置...")
    
    # 主要顯示區域
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 即時監控", "🤖 AI模型狀態", "🚦 交通控制", "📈 效能分析", "⚙️ 系統設定"
    ])
    
    with tab1:
        st.header("📊 即時交通監控")
        
        # 載入數據
        traffic_data = load_sample_data()
        
        # 關鍵指標
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_traffic = np.random.randint(150, 300)
            st.metric(
                "當前車流量", 
                f"{current_traffic} 輛/小時",
                delta=f"{np.random.randint(-20, 20)} 輛/小時"
            )
        
        with col2:
            avg_speed = np.random.uniform(25, 45)
            st.metric(
                "平均車速",
                f"{avg_speed:.1f} km/h",
                delta=f"{np.random.uniform(-5, 5):.1f} km/h"
            )
        
        with col3:
            wait_time = np.random.uniform(20, 60)
            st.metric(
                "平均等待時間",
                f"{wait_time:.1f} 秒",
                delta=f"{np.random.uniform(-10, 10):.1f} 秒"
            )
        
        with col4:
            efficiency = np.random.uniform(75, 95)
            st.metric(
                "系統效率",
                f"{efficiency:.1f}%",
                delta=f"{np.random.uniform(-5, 5):.1f}%"
            )
        
        # 交通流量趨勢圖
        st.subheader("📈 24小時交通流量趨勢")
        
        fig_traffic = go.Figure()
        
        for intersection in ['intersection_1', 'intersection_2', 'intersection_3', 'intersection_4']:
            fig_traffic.add_trace(go.Scatter(
                x=traffic_data['timestamp'],
                y=traffic_data[intersection],
                mode='lines+markers',
                name=f'路口 {intersection.split("_")[1]}',
                line=dict(width=3)
            ))
        
        fig_traffic.update_layout(
            title="各路口車流量變化",
            xaxis_title="時間",
            yaxis_title="車流量 (輛/小時)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_traffic, use_container_width=True)
        
        # 交通熱力圖
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🗺️ 交通密度熱力圖")
            
            # 生成模擬熱力圖數據
            grid_size = 10
            density_data = np.random.exponential(2, (grid_size, grid_size))
            
            fig_heatmap = px.imshow(
                density_data,
                color_continuous_scale='Reds',
                title="實時交通密度分佈"
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            st.subheader("🚦 交通燈狀態")
            
            # 交通燈狀態表格
            light_status = pd.DataFrame({
                '路口ID': ['TL_001', 'TL_002', 'TL_003', 'TL_004'],
                '當前相位': ['綠燈', '紅燈', '綠燈', '黃燈'],
                '剩餘時間': ['25秒', '45秒', '15秒', '3秒'],
                '等待車輛': [12, 28, 8, 15],
                'AI建議': ['延長', '正常', '縮短', '切換']
            })
            
            # 美化表格顯示
            for idx, row in light_status.iterrows():
                status_color = {
                    '綠燈': 'status-good',
                    '黃燈': 'status-warning', 
                    '紅燈': 'status-danger'
                }[row['當前相位']]
                
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{row['路口ID']}</strong><br>
                    <span class="{status_color}">● {row['當前相位']}</span> - {row['剩餘時間']}<br>
                    等待車輛: {row['等待車輛']} 輛<br>
                    AI建議: <strong>{row['AI建議']}</strong>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.header("🤖 AI模型運行狀態")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 模型效能指標")
            
            # 模型效能數據
            model_metrics = {
                '預測准確率': 94.2,
                '收斂速度': 87.5,
                '決策效率': 91.8,
                '學習穩定性': 89.3
            }
            
            for metric, value in model_metrics.items():
                progress_color = 'normal'
                if value >= 90:
                    progress_color = 'success'
                elif value >= 80:
                    progress_color = 'warning'
                else:
                    progress_color = 'error'
                
                st.metric(metric, f"{value}%")
                st.progress(value / 100)
        
        with col2:
            st.subheader("🔬 多模態融合狀態")
            
            fusion_data = pd.DataFrame({
                '數據源': ['交通攝影機', 'GPS軌跡', '氣象資訊', '社群媒體', 'IoT感測器'],
                '數據量 (MB/h)': [1200, 800, 50, 300, 150],
                '處理延遲 (ms)': [45, 23, 12, 67, 18],
                '融合權重': [0.35, 0.25, 0.15, 0.10, 0.15]
            })
            
            # 融合權重餅圖
            fig_pie = px.pie(
                fusion_data, 
                values='融合權重', 
                names='數據源',
                title="多模態數據融合權重分配"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # 訓練損失曲線
        st.subheader("📉 模型訓練進度")
        
        # 生成模擬訓練數據
        epochs = list(range(1, 101))
        actor_loss = [1.0 * np.exp(-x/20) + 0.1 + np.random.normal(0, 0.05) for x in epochs]
        critic_loss = [0.8 * np.exp(-x/25) + 0.08 + np.random.normal(0, 0.03) for x in epochs]
        
        fig_training = go.Figure()
        fig_training.add_trace(go.Scatter(x=epochs, y=actor_loss, name='Actor Loss', line=dict(color='blue')))
        fig_training.add_trace(go.Scatter(x=epochs, y=critic_loss, name='Critic Loss', line=dict(color='red')))
        
        fig_training.update_layout(
            title="強化學習訓練損失",
            xaxis_title="訓練輪數",
            yaxis_title="損失值",
            height=400
        )
        
        st.plotly_chart(fig_training, use_container_width=True)
    
    with tab3:
        st.header("🚦 智能交通信號控制")
        
        # 交通控制面板
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 控制策略")
            control_strategy = st.selectbox(
                "選擇控制策略",
                ["AI自適應控制", "定時控制", "感應控制", "手動控制"]
            )
            
            optimization_goal = st.selectbox(
                "優化目標",
                ["最小化等待時間", "最大化通過量", "均衡化流量", "減少排放"]
            )
        
        with col2:
            st.subheader("⚡ 即時調整")
            
            # 緊急控制按鈕
            if st.button("🚨 緊急車輛優先", type="primary"):
                st.success("緊急車輛綠色通道已啟動！")
            
            if st.button("🔧 重新路徑規劃"):
                st.info("正在重新計算最優路徑...")
            
            if st.button("📊 流量重分配"):
                st.warning("正在執行流量重新分配...")
        
        with col3:
            st.subheader("📱 手動控制")
            
            selected_intersection = st.selectbox(
                "選擇路口",
                ["路口 TL_001", "路口 TL_002", "路口 TL_003", "路口 TL_004"]
            )
            
            manual_phase = st.selectbox(
                "設置信號相位",
                ["南北直行", "東西直行", "左轉", "全紅"]
            )
            
            if st.button("✅ 執行手動控制"):
                st.success(f"{selected_intersection} 已設置為 {manual_phase}")
        
        # 控制效果預測
        st.subheader("🔮 控制效果預測")
        
        # 生成預測數據
        time_horizon = list(range(1, 31))  # 30分鐘預測
        current_scenario = [100 + 10*np.sin(t/5) + np.random.normal(0, 5) for t in time_horizon]
        optimized_scenario = [80 + 8*np.sin(t/5) + np.random.normal(0, 3) for t in time_horizon]
        
        fig_prediction = go.Figure()
        fig_prediction.add_trace(go.Scatter(
            x=time_horizon, 
            y=current_scenario, 
            name='當前策略', 
            line=dict(color='red', dash='dash')
        ))
        fig_prediction.add_trace(go.Scatter(
            x=time_horizon, 
            y=optimized_scenario, 
            name='AI優化策略', 
            line=dict(color='green')
        ))
        
        fig_prediction.update_layout(
            title="未來30分鐘等待時間預測對比",
            xaxis_title="時間 (分鐘)",
            yaxis_title="平均等待時間 (秒)",
            height=400
        )
        
        st.plotly_chart(fig_prediction, use_container_width=True)
    
    with tab4:
        st.header("📈 系統效能分析")
        
        # 效能提升統計
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("⏱️ 時間效益")
            time_improvements = {
                "平均等待時間減少": "27.34%",
                "通勤時間縮短": "18.7%", 
                "紅燈等待減少": "31.2%"
            }
            
            for metric, improvement in time_improvements.items():
                st.metric(metric, improvement, delta=f"+{improvement}")
        
        with col2:
            st.subheader("🌱 環境效益")
            env_improvements = {
                "CO2排放減少": "15.8%",
                "燃油消耗降低": "12.4%",
                "空氣品質改善": "8.9%"
            }
            
            for metric, improvement in env_improvements.items():
                st.metric(metric, improvement, delta=f"+{improvement}")
        
        with col3:
            st.subheader("💰 經濟效益")
            economic_improvements = {
                "運輸成本節省": "¥2.1M/年",
                "燃料費用降低": "¥1.3M/年",
                "維護成本減少": "¥0.8M/年"
            }
            
            for metric, improvement in economic_improvements.items():
                st.metric(metric, improvement, delta=f"+{improvement}")
        
        # 長期趨勢分析
        st.subheader("📊 長期效能趨勢")
        
        # 生成30天的效能數據
        dates = pd.date_range(start='2025-05-10', periods=30, freq='D')
        efficiency_trend = 70 + 20 * (1 - np.exp(-np.arange(30)/10)) + np.random.normal(0, 2, 30)
        
        fig_trend = px.line(
            x=dates, 
            y=efficiency_trend,
            title="系統效率提升趨勢 (30天)",
            labels={'x': '日期', 'y': '系統效率 (%)'}
        )
        fig_trend.update_traces(line=dict(width=3, color='#1f77b4'))
        fig_trend.update_layout(height=400)
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab5:
        st.header("⚙️ 系統設定與配置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 AI模型參數")
            
            with st.expander("強化學習設定"):
                new_lr = st.number_input("學習率", value=0.0003, format="%.4f")
                new_gamma = st.slider("折扣因子", 0.8, 0.99, 0.95)
                new_epsilon = st.slider("探索率衰減", 0.01, 1.0, 0.1)
                
                if st.button("💾 保存RL設定"):
                    st.success("強化學習參數已更新！")
            
            with st.expander("多模態融合設定"):
                text_weight = st.slider("文本權重", 0.0, 1.0, 0.3)
                image_weight = st.slider("影像權重", 0.0, 1.0, 0.4)
                sensor_weight = st.slider("感測器權重", 0.0, 1.0, 0.3)
                
                if st.button("💾 保存融合設定"):
                    st.success("多模態融合參數已更新！")
        
        with col2:
            st.subheader("📡 數據源配置")
            
            with st.expander("API設定"):
                openai_key = st.text_input("OpenAI API Key", type="password")
                weather_api = st.text_input("氣象API Key", type="password")
                maps_api = st.text_input("地圖API Key", type="password")
                
                if st.button("🔗 測試API連接"):
                    st.info("正在測試API連接...")
                    time.sleep(2)
                    st.success("所有API連接正常！")
            
            with st.expander("系統監控"):
                enable_logging = st.checkbox("啟用詳細日誌", value=True)
                log_level = st.selectbox("日誌級別", ["DEBUG", "INFO", "WARNING", "ERROR"])
                enable_alerts = st.checkbox("啟用異常告警", value=True)
                
                if st.button("📋 下載系統日誌"):
                    st.success("系統日誌下載已開始！")
        
        # 系統資訊
        st.subheader("💻 系統資訊")
        
        system_info = {
            "系統版本": "SmartTraffic AI v2.1.0",
            "部署環境": "Kubernetes Cluster",
            "運行時間": "72天 14小時 32分鐘",
            "CPU使用率": "45.2%",
            "記憶體使用": "6.8GB / 16GB",
            "GPU使用率": "78.3%",
            "網路延遲": "< 5ms",
            "數據處理量": "2.3TB/天"
        }
        
        col1, col2, col3, col4 = st.columns(4)
        items = list(system_info.items())
        
        for i, (key, value) in enumerate(items):
            col_idx = i % 4
            if col_idx == 0:
                col1.metric(key, value)
            elif col_idx == 1:
                col2.metric(key, value)
            elif col_idx == 2:
                col3.metric(key, value)
            else:
                col4.metric(key, value)

if __name__ == "__main__":
    main()
