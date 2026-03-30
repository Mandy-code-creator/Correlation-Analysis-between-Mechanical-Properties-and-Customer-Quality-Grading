import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# Cấu hình trang Dashboard
st.set_page_config(page_title="QC Data Analysis Dashboard", layout="wide")

st.title("📊 Hệ thống Phân tích & Định hình Cơ tính theo Cấp độ Chất lượng")
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
    
    # Tính Điểm chất lượng trung bình (Thang 1-5) để dùng cho Tab 2
    df['Quality_Score'] = (5 * df['A+B+數'] + 4 * df['A-B+數'] + 
                           3 * df['A-B數'] + 2 * df['A-B-數'] + 1 * df['B+數']) / df['Total_Count']

    # Tạo 3 Tabs
    tab1, tab2, tab3 = st.tabs(["1. Bảng Thống kê & Tỉ lệ", "2. Ma trận Tương quan", "3. Phân tích Phân phối theo Cấp độ"])

    # --- TAB 1: THỐNG KÊ KẾT QUẢ ---
    with tab1:
        st.header("1. Thống kê Phân bổ Chất lượng tổng hợp")
        
        # Tạo bảng thống kê tổng hợp theo độ dày
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Tổng cuộn kiểm tra'] = summary_df[count_cols].sum(axis=1)
        
        # Tính % A+B+
        summary_df['Tỉ lệ A+B+ (%)'] = (summary_df['A+B+數'] / summary_df['Tổng cuộn kiểm tra'] * 100).round(2)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Bảng số liệu tổng hợp theo Độ dày")
            st.dataframe(summary_df, use_container_width=True)
            
        with col2:
            st.subheader("Biểu đồ tỉ lệ phần trăm")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            summary_df.set_index('厚度歸類')[count_cols].apply(lambda x: x*100/sum(x), axis=1).plot(
                kind='bar', stacked=True, ax=ax1, colormap='RdYlGn_r'
            )
            ax1.set_ylabel("Tỉ lệ (%)")
            ax1.set_xlabel("Độ dày (厚度歸類)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig1)

    # --- TAB 2: MA TRẬN TƯƠNG QUAN ---
    with tab2:
        st.header("2. Ma trận Hệ số Tương quan")
        features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
        
        # Tính toán ma trận tương quan chỉ cho các cột cần thiết
        corr_matrix = df[['Quality_Score'] + features].corr()[['Quality_Score']].drop('Quality_Score')
        corr_matrix.columns = ['Độ tương quan với Điểm Chất lượng Tổng']
        
        col_corr1, col_corr2 = st.columns([1, 2])
        
        with col_corr1:
            st.subheader("Bảng Hệ số Tương quan (Pearson)")
            st.markdown("*(Hệ số càng gần -1 thì cột cơ tính đó càng cao, chất lượng càng giảm)*")
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=0), use_container_width=True)
            
        with col_corr2:
            st.subheader("Giải thích")
            max_corr_feature = corr_matrix.idxmin()[0]
            st.info(f"Dựa trên dữ liệu, thông số **{max_corr_feature}** có ảnh hưởng tiêu cực nhất đến điểm chất lượng. Khi {max_corr_feature} tăng cao, chất lượng có xu hướng giảm xuống.")

    # --- TAB 3: PHÂN TÍCH PHÂN PHỐI THEO CẤP ĐỘ (NÂNG CẤP) ---
    with tab3:
        st.header("3. Định hình Cơ tính theo Cấp độ Chất lượng")
        
        # Ánh xạ tên cấp độ với tên cột
        grade_mapping = {
            'A+B+ (Xuất sắc)': 'A+B+數',
            'A-B+ (Tốt)': 'A-B+數',
            'A-B (Trung bình)': 'A-B數',
            'A-B- (Kém)': 'A-B-數',
            'B+ (Thứ phẩm)': 'B+數'
        }
        
        # Thanh điều khiển
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            selected_thickness = st.selectbox("1. Lọc theo Độ dày:", df['厚度歸類'].unique(), key='tab3_thick')
        with col_ctrl2:
            selected_grade_label = st.selectbox("2. Chọn Cấp độ Chất lượng để phân tích:", list(grade_mapping.keys()), index=3) # Mặc định chọn A-B-
            selected_grade_col = grade_mapping[selected_grade_label]
        with col_ctrl3:
            selected_feature = st.selectbox("3. Chọn Thông số Cơ tính:", features, key='tab3_feat')
        
        st.markdown("---")
        
        # Xử lý dữ liệu: Chỉ lấy các cuộn có tồn tại cấp độ chất lượng được chọn
        df_thick = df[df['厚度歸類'] == selected_thickness]
        # Lọc dữ liệu và bỏ các giá trị null
        grade_data = df_thick[[selected_feature, selected_grade_col]].dropna()
        # Chỉ lấy những cuộn mà số lượng cấp độ đó > 0
        grade_data = grade_data[grade_data[selected_grade_col] > 0]
        
        if len(grade_data) < 2:
            st.warning(f"Không đủ dữ liệu thống kê cho cấp độ **{selected_grade_label}** với độ dày {selected_thickness}.")
        else:
            # Dữ liệu và Trọng số (Số lượng điểm chất lượng)
            values = grade_data[selected_feature].values
            weights = grade_data[selected_grade_col].values
            total_weight = weights.sum()

            # --- TÍNH TOÁN CÁC CHỈ SỐ THỐNG KÊ CÓ TRỌNG SỐ ---
            # 1. Trung bình có trọng số (Weighted Mean)
            weighted_mean = np.average(values, weights=weights)
            
            # 2. Độ lệch chuẩn có trọng số (Weighted Std Dev)
            weighted_var = np.average((values - weighted_mean)**2, weights=weights)
            weighted_std = np.sqrt(weighted_var)
            
            # 3. Phân vị có trọng số (Percentile 10 - 90) để xác định khoảng phân bố chủ yếu
            # Cần sắp xếp dữ liệu để tính phân vị
            sorter = np.argsort(values)
            values_sorted = values[sorter]
            weights_sorted = weights[sorter]
            cum_weights = np.cumsum(weights_sorted)
            
            p10_idx = np.searchsorted(cum_weights, 0.10 * total_weight)
            p90_idx = np.searchsorted(cum_weights, 0.90 * total_weight)
            
            # Đảm bảo index không vượt quá giới hạn
            p10_idx = min(p10_idx, len(values_sorted) - 1)
            p90_idx = min(p90_idx, len(values_sorted) - 1)
            
            weighted_p10 = values_sorted[p10_idx]
            weighted_p90 = values_sorted[p90_idx]

            # --- TRỰC QUAN HÓA BẰNG BIỂU ĐỒ PHÂN PHỐI CHUẨN ---
            col_chart, col_stats = st.columns([2, 1])
            
            with col_chart:
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                
                # 1. Vẽ Histogram thực tế (có trọng số)
                sns.histplot(x=values, weights=weights, bins=20, stat='density', alpha=0.4, color='blue', label='Dữ liệu thực tế (Có trọng số)', ax=ax3)
                
                # 2. Vẽ đường cong Phân phối chuẩn toán học (Bell Curve) dựa trên mean và std có trọng số
                xmin, xmax = ax3.get_xlim()
                x_axis = np.linspace(xmin, xmax, 100)
                # Đảm bảo Std Dev không phải là 0 để tránh lỗi toán học
                if weighted_std > 0:
                    p = stats.norm.pdf(x_axis, weighted_mean, weighted_std)
                    ax3.plot(x_axis, p, 'k', linewidth=2, label=f'Đường cong Bell Curve lý thuyết\n(μ={weighted_mean:.1f}, σ={weighted_std:.1f})')
                    
                    # Tô màu vùng phân bố chủ yếu (10% - 90%)
                    ax3.fill_between(x_axis, p, where=((x_axis >= weighted_p10) & (x_axis <= weighted_p90)), color='red', alpha=0.2, label='Khoảng phân bố chủ yếu (10%-90%)')
                
                # 3. Kẻ vạch các ranh giới phân vị
                ax3.axvline(weighted_p10, color='red', linestyle='--', linewidth=2, label=f'Phân vị 10% ({weighted_p10:.1f})')
                ax3.axvline(weighted_p90, color='red', linestyle='--', linewidth=2, label=f'Phân vị 90% ({weighted_p90:.1f})')
                
                ax3.set_title(f"Phân phối {selected_feature} của Cấp độ: {selected_grade_label} (Thép {selected_thickness})")
                ax3.set_xlabel(f"Giá trị {selected_feature}")
                ax3.set_ylabel("Mật độ phân phối")
                ax3.legend(loc='upper right')
                st.pyplot(fig3)
                
            with col_stats:
                st.subheader("📊 Bảng kết quả thống kê")
                st.markdown(f"**Thông số:** {selected_feature}")
                st.markdown(f"**Tổng số điểm đánh giá được tích lũy:** {total_weight:.0f}")
                
                stats_df = pd.DataFrame({
                    "Chỉ số Thống kê (Có trọng số)": ["Trung bình (μ)", "Độ lệch chuẩn (σ)", "Giới hạn Dưới (P10)", "Giới hạn Trên (P90)"],
                    "Giá trị": [round(weighted_mean, 2), round(weighted_std, 2), round(weighted_p10, 2), round(weighted_p90, 2)]
                })
                st.dataframe(stats_df.set_index("Chỉ số Thống kê (Có trọng số)"), use_container_width=True)
                
                # Đưa ra nhận xét dựa trên so sánh giữa Hàng Tốt và Hàng Kém (Cần dữ liệu A+B+ để so sánh)
                # Ở đây ta tạm đưa ra kết luận đơn giản cho cấp độ hiện tại
                st.success(f"**Nhận xét:**\nHàng bị khách đánh giá là **{selected_grade_label}** phần lớn (80%) có giá trị {selected_feature} nằm trong khoảng **{round(weighted_p10, 2)} đến {round(weighted_p90, 2)}**.")

else:
    st.info("Vui lòng tải file Excel dữ liệu của bạn ở thanh công cụ phía trên để bắt đầu phân tích.")
