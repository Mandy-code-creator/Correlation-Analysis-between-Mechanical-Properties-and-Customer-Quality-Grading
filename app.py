import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cấu hình trang Dashboard
st.set_page_config(page_title="QC Data Analysis Dashboard", layout="wide")

st.title("📊 Hệ thống Phân tích Tương quan Cơ tính & Đề xuất Tiêu chuẩn QC")
st.markdown("**Dự án:** Tối ưu hóa giới hạn kiểm soát (YS, TS, EL, YPE, HARDNESS) dựa trên đánh giá khách hàng.")

# 1. Tải file Excel lên
uploaded_file = st.file_uploader("Tải file Excel dữ liệu của bạn lên (Định dạng: .xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Đọc dữ liệu
    df = pd.read_excel(uploaded_file)
    
    # Xóa khoảng trắng thừa trong tên cột (nếu có) để tránh lỗi
    df.columns = df.columns.str.strip()
    
    # 2. Tiền xử lý dữ liệu
    # Tính tổng số lượng điểm đánh giá trên mỗi cuộn
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    
    # Đảm bảo các cột này là số
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['Total_Count'] = df[count_cols].sum(axis=1)
    
    # Lọc bỏ các dòng không có dữ liệu đánh giá
    df = df[df['Total_Count'] > 0].copy()
    
    # Tính Điểm chất lượng (Thang 1-5)
    df['Quality_Score'] = (5 * df['A+B+數'] + 
                           4 * df['A-B+數'] + 
                           3 * df['A-B數'] + 
                           2 * df['A-B-數'] + 
                           1 * df['B+數']) / df['Total_Count']

    # Tạo các Tab cho Dashboard
    tab1, tab2, tab3 = st.tabs(["1. Tổng quan Tỉ lệ", "2. Phân tích Tương quan", "3. Đề xuất Tiêu chuẩn Mới"])

    # --- TAB 1: TỔNG QUAN ---
    with tab1:
        st.header("Bức tranh Tổng thể về Chất lượng")
        
        # Nhóm theo Độ dày (厚度歸類) và tính tổng các loại
        summary_df = df.groupby('厚度歸類')[count_cols].sum()
        
        col1, col2 = st.columns(2)
        
        for idx, thickness in enumerate(summary_df.index):
            with col1 if idx % 2 == 0 else col2:
                st.subheader(f"Độ dày: {thickness}")
                data = summary_df.loc[thickness]
                
                # Vẽ biểu đồ tròn
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, 
                       colors=['#2ca02c', '#98df8a', '#ffbb78', '#ff9896', '#d62728'])
                ax.axis('equal')
                st.pyplot(fig)

    # --- TAB 2: TƯƠNG QUAN ---
    with tab2:
        st.header("Mối tương quan giữa Cơ tính và Điểm Chất lượng")
        st.markdown("Đường nét đứt màu đỏ thể hiện mốc chất lượng Tốt (Điểm 4.0 trở lên).")
        
        features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
        
        # Tạo lưới biểu đồ
        fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
        for i, feature in enumerate(features):
            row, col = divmod(i, 3)
            ax = axes[row, col]
            
            # Scatter plot phân loại theo độ dày
            sns.scatterplot(data=df, x=feature, y='Quality_Score', hue='厚度歸類', 
                            palette='Set1', alpha=0.7, ax=ax)
            
            ax.set_title(f'{feature} vs Điểm chất lượng')
            ax.set_ylabel('Điểm chất lượng (1-5)')
            ax.axhline(4.0, color='red', linestyle='--', alpha=0.5)
            
        fig2.delaxes(axes[1, 2]) # Xóa ô trống
        plt.tight_layout()
        st.pyplot(fig2)

    # --- TAB 3: ĐỀ XUẤT TIÊU CHUẨN (GIÁ TRỊ CỐT LÕI) ---
    with tab3:
        st.header("Đề xuất Giới hạn Kiểm soát Mới (Data-Driven)")
        st.markdown("Thuật toán tự động trích xuất các cuộn thép **đạt điểm xuất sắc (> 4.5)**, sau đó tính toán dải phân vị từ 10% đến 90% để loại bỏ nhiễu, từ đó tìm ra **Vùng tiêu chuẩn vàng**.")
        
        # Chọn độ dày để phân tích
        selected_thickness = st.selectbox("Chọn độ dày để phân tích tiêu chuẩn:", df['厚度歸類'].unique())
        
        df_thick = df[df['厚度歸類'] == selected_thickness]
        
        # Lọc tập "Dữ liệu vàng" (Điểm chất lượng >= 4.5)
        golden_df = df_thick[df_thick['Quality_Score'] >= 4.5]
        
        if len(golden_df) < 5:
            st.warning(f"Không đủ dữ liệu cuộn thép xuất sắc cho độ dày {selected_thickness} để tính toán đề xuất (Cần ít nhất 5 cuộn).")
        else:
            proposals = []
            for feature in ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']:
                # Tính Phân vị 10 và 90
                p10 = golden_df[feature].quantile(0.10)
                p90 = golden_df[feature].quantile(0.90)
                
                proposals.append({
                    "Thông số": feature,
                    "Giới hạn dưới đề xuất (Min)": round(p10, 2),
                    "Giới hạn trên đề xuất (Max)": round(p90, 2)
                })
                
            proposal_df = pd.DataFrame(proposals)
            
            st.subheader(f"Bảng Ma Trận Tiêu Chuẩn Mới cho thép {selected_thickness}")
            st.dataframe(proposal_df, use_container_width=True)
            
            # Trực quan hóa sự phân bố của 1 thông số cụ thể
            st.markdown("---")
            st.subheader("Biểu đồ Phân bố so sánh (Trực quan hóa cho sếp)")
            feature_to_plot = st.selectbox("Chọn thông số để xem phân bố:", ['YS', 'TS', 'EL', 'YPE', 'HARDNESS'])
            
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            
            # Vẽ phân bố của Hàng Tốt và Hàng Kém
            sns.kdeplot(data=df_thick[df_thick['Quality_Score'] >= 4.0], x=feature_to_plot, fill=True, label="Nhóm Tốt (Đạt)", color="green", ax=ax3)
            sns.kdeplot(data=df_thick[df_thick['Quality_Score'] < 4.0], x=feature_to_plot, fill=True, label="Nhóm Kém (Bị phàn nàn)", color="red", ax=ax3)
            
            # Vẽ đường ranh giới mới đề xuất
            min_prop = proposal_df[proposal_df['Thông số'] == feature_to_plot]['Giới hạn dưới đề xuất (Min)'].values[0]
            max_prop = proposal_df[proposal_df['Thông số'] == feature_to_plot]['Giới hạn trên đề xuất (Max)'].values[0]
            
            ax3.axvline(min_prop, color='blue', linestyle='--', linewidth=2, label=f'Đề xuất Min ({min_prop})')
            ax3.axvline(max_prop, color='blue', linestyle='--', linewidth=2, label=f'Đề xuất Max ({max_prop})')
            
            ax3.set_title(f"Phân bố {feature_to_plot} thực tế và Vùng đề xuất (Màu xanh dương)")
            ax3.legend()
            st.pyplot(fig3)

else:
    st.info("Vui lòng upload file Excel (có chứa các cột như: 使用日期, 鋼捲號碼, 厚度歸類, YS, TS, A+B+數,...) để bắt đầu phân tích.")
