import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# Cấu hình trang Dashboard
st.set_page_config(page_title="QC Data Analysis Dashboard", layout="wide")

st.title("📊 Hệ thống Phân tích Tương quan & Xác định Khoảng Cơ tính An toàn")
st.markdown("---")

# 1. Tải file Excel lên
uploaded_file = st.file_uploader("Tải file Excel dữ liệu của bạn lên (Định dạng: .xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Đọc dữ liệu
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip() # Xóa khoảng trắng thừa
    
    # 2. Tiền xử lý dữ liệu
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['Total_Count'] = df[count_cols].sum(axis=1)
    df = df[df['Total_Count'] > 0].copy() # Lọc dữ liệu hợp lệ
    
    # Tính Điểm chất lượng (Thang 1-5)
    df['Quality_Score'] = (5 * df['A+B+數'] + 4 * df['A-B+數'] + 
                           3 * df['A-B數'] + 2 * df['A-B-數'] + 1 * df['B+數']) / df['Total_Count']

    # Tạo 3 Tabs
    tab1, tab2, tab3 = st.tabs(["1. Bảng Thống kê & Tỉ lệ", "2. Phân tích Tương quan", "3. Khoảng An toàn (Phân phối chuẩn)"])

    # --- TAB 1: THỐNG KÊ KẾT QUẢ ---
    with tab1:
        st.header("1. Thống kê Phân bổ Chất lượng theo Độ dày")
        
        # Tạo bảng thống kê tổng hợp
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Tổng cuộn kiểm tra'] = summary_df[count_cols].sum(axis=1)
        
        # Tính % A+B+ để đưa vào bảng
        summary_df['Tỉ lệ A+B+ (%)'] = (summary_df['A+B+數'] / summary_df['Tổng cuộn kiểm tra'] * 100).round(2)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Bảng số liệu tổng hợp")
            st.dataframe(summary_df, use_container_width=True)
            
        with col2:
            st.subheader("Biểu đồ tỉ lệ phần trăm")
            # Trực quan hóa bằng Stacked Bar Chart cho gọn gàng
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            summary_df.set_index('厚度歸類')[count_cols].apply(lambda x: x*100/sum(x), axis=1).plot(
                kind='bar', stacked=True, ax=ax1, colormap='RdYlGn_r' # Xanh lá (Tốt) đến Đỏ (Kém)
            )
            ax1.set_ylabel("Tỉ lệ (%)")
            ax1.set_xlabel("Độ dày (厚度歸類)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig1)

    # --- TAB 2: MỐI TƯƠNG QUAN ---
    with tab2:
        st.header("2. Mối tương quan giữa Cơ tính và Điểm Chất lượng")
        features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
        
        col_corr1, col_corr2 = st.columns([1, 2])
        
        with col_corr1:
            st.subheader("Bảng Hệ số Tương quan (Pearson)")
            st.markdown("*(Hệ số càng gần 1 hoặc -1 thì tương quan càng mạnh)*")
            # Tính toán ma trận tương quan chỉ cho các cột cần thiết
            corr_matrix = df[['Quality_Score'] + features].corr()[['Quality_Score']].drop('Quality_Score')
            corr_matrix.columns = ['Độ tương quan với Chất lượng']
            
            # Tô màu bảng để sếp dễ nhìn (Xanh = Tương quan thuận, Đỏ = Tương quan nghịch)
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=0), use_container_width=True)
            
        with col_corr2:
            st.subheader("Biểu đồ Phân tán (Scatter Plot)")
            selected_feature_corr = st.selectbox("Chọn thông số cơ tính để xem chi tiết biểu đồ:", features)
            
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x=selected_feature_corr, y='Quality_Score', hue='厚度歸類', palette='Set1', alpha=0.8, ax=ax2)
            ax2.axhline(4.0, color='red', linestyle='--', label='Ngưỡng An toàn (4.0 điểm)')
            ax2.set_title(f"Tương quan giữa {selected_feature_corr} và Chất lượng")
            ax2.set_ylabel("Điểm Chất lượng")
            ax2.legend()
            st.pyplot(fig2)

    # --- TAB 3: KHOẢNG CƠ TÍNH AN TOÀN (PHÂN PHỐI CHUẨN) ---
    with tab3:
        st.header("3. Xác định Khoảng Cơ tính An toàn (Dựa trên Phân phối chuẩn)")
        
        col_ctrl1, col_ctrl2 = st.columns([1, 3])
        
        with col_ctrl1:
            selected_thickness = st.selectbox("Lọc theo Độ dày:", df['厚度歸類'].unique())
            selected_feature = st.selectbox("Chọn Thông số phân tích:", features)
        
        # Lọc dữ liệu theo độ dày và chỉ lấy những cuộn thép có chất lượng TỐT (Điểm >= 4.0)
        df_thick = df[df['厚度歸類'] == selected_thickness]
        good_coils = df_thick[df_thick['Quality_Score'] >= 4.0][selected_feature].dropna()
        
        if len(good_coils) < 5:
            st.warning("Không đủ dữ liệu cuộn thép đạt chuẩn để vẽ phân phối chuẩn.")
        else:
            # Tính toán các thông số thống kê cho Phân phối chuẩn
            mu, std = stats.norm.fit(good_coils) # Trung bình (Mean) và Độ lệch chuẩn (Std Dev)
            
            # Đề xuất khoảng an toàn: Từ (Mean - 1.5*Std) đến (Mean + 1.5*Std) bao phủ ~86% dữ liệu tốt
            # Hoặc dùng Phân vị (Percentile 10 - 90) để linh hoạt hơn. Ở đây dùng Percentile cho thực tế nhà máy.
            p10 = np.percentile(good_coils, 10)
            p90 = np.percentile(good_coils, 90)
            
            with col_ctrl1:
                st.markdown("### Bảng Đề xuất")
                result_df = pd.DataFrame({
                    "Chỉ số": [selected_feature],
                    "Trung bình (μ)": [round(mu, 2)],
                    "Giới hạn Dưới (Min)": [round(p10, 2)],
                    "Giới hạn Trên (Max)": [round(p90, 2)]
                })
                st.dataframe(result_df.set_index("Chỉ số"))
                st.success(f"**Kết luận:** Để đạt chất lượng tốt đối với thép {selected_thickness}, hãy kiểm soát {selected_feature} trong khoảng từ **{round(p10, 2)} đến {round(p90, 2)}**.")

            with col_ctrl2:
                # Vẽ biểu đồ Phân phối chuẩn (Bell Curve)
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                
                # 1. Vẽ Histogram thực tế của các cuộn đạt chất lượng
                sns.histplot(good_coils, bins=15, stat='density', alpha=0.5, color='green', label='Dữ liệu thực tế (Các cuộn đạt chuẩn)', ax=ax3)
                
                # 2. Vẽ đường cong Phân phối chuẩn toán học (Bell Curve)
                xmin, xmax = ax3.get_xlim()
                x_axis = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x_axis, mu, std)
                ax3.plot(x_axis, p, 'k', linewidth=2, label=f'Đường cong Phân phối chuẩn\n(μ={mu:.1f}, σ={std:.1f})')
                
                # 3. Kẻ vạch Khoảng an toàn đề xuất
                ax3.axvline(p10, color='blue', linestyle='--', linewidth=2.5, label=f'Giới hạn Dưới ({p10:.1f})')
                ax3.axvline(p90, color='red', linestyle='--', linewidth=2.5, label=f'Giới hạn Trên ({p90:.1f})')
                
                # Tô màu vùng an toàn
                ax3.fill_between(x_axis, p, where=((x_axis >= p10) & (x_axis <= p90)), color='yellow', alpha=0.3, label='Vùng Cơ tính An toàn')
                
                ax3.set_title(f"Biểu đồ Phân phối chuẩn của {selected_feature} (Thép {selected_thickness} - Hàng Đạt)")
                ax3.set_xlabel(f"Giá trị {selected_feature}")
                ax3.set_ylabel("Mật độ phân phối (Density)")
                ax3.legend(loc='upper right')
                
                st.pyplot(fig3)

else:
    st.info("👆 Vui lòng tải file Excel dữ liệu của bạn ở thanh công cụ phía trên để bắt đầu hệ thống tự động phân tích.")
