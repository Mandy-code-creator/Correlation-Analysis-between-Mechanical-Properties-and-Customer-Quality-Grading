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
    tab1, tab2, tab3 = st.tabs(["1. Bảng Thống kê & Tỉ lệ", "2. Ma trận Tương quan", "3. Phân tích Dữ liệu Toàn cảnh"])

    # --- TAB 1: THỐNG KÊ KẾT QUẢ ---
    with tab1:
        st.header("1. Thống kê Phân bổ Chất lượng tổng hợp")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Tổng cuộn kiểm tra'] = summary_df[count_cols].sum(axis=1)
        summary_df['Tỉ lệ A+B+ (%)'] = (summary_df['A+B+數'] / summary_df['Tổng cuộn kiểm tra'] * 100).round(2)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Bảng số liệu tổng hợp theo Độ dày")
            st.dataframe(summary_df, use_container_width=True)
            
        with col2:
            st.subheader("Biểu đồ tỉ lệ phần trăm")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            
            # XỬ LÝ LỖI FONT & VẼ BIỂU ĐỒ
            # 1. Tạo dataframe tỷ lệ và đổi tên cột để tránh lỗi tiếng Trung
            plot_df = summary_df.set_index('厚度歸類')[count_cols].apply(lambda x: x*100/sum(x), axis=1)
            plot_df.columns = ['A+B+', 'A-B+', 'A-B', 'A-B-', 'B+']
            
            # 2. Vẽ biểu đồ Stacked Bar
            plot_df.plot(kind='bar', stacked=True, ax=ax1, colormap='RdYlGn_r', edgecolor='white', linewidth=0.5)
            ax1.set_ylabel("Tỉ lệ (%)")
            ax1.set_xlabel("Độ dày") # Đổi tên trục X để sửa lỗi font
            
            # 3. THÊM SỐ PHẦN TRĂM VÀO GIỮA CỘT
            for container in ax1.containers:
                # Điều kiện: Chỉ in số nếu chiều cao (tỉ lệ %) > 3% để tránh đè chữ
                labels = [f"{v.get_height():.1f}%" if v.get_height() > 3 else "" for v in container]
                ax1.bar_label(container, labels=labels, label_type='center', fontsize=9, fontweight='bold', color='#333333')
            
            # Cấu hình lại chú thích (Legend)
            plt.legend(title="Cấp độ", bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig1)

    # --- TAB 2: MA TRẬN TƯƠNG QUAN ---
    with tab2:
        st.header("2. Ma trận Hệ số Tương quan")
        features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
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

    # --- TAB 3: PHÂN TÍCH THEO ĐỘ DÀY & CẤP ĐỘ CHẤT LƯỢNG (NORMAL CURVE) ---
    with tab3:
        st.header("3. So sánh Cơ tính phân rã theo Độ Dày và Cấp Độ")
        st.markdown("Biểu đồ cột thể hiện dữ liệu thực tế. **Các đường cong là phân phối chuẩn (Normal Bell Curve)** được tính toán toán học từ độ lệch chuẩn và giá trị trung bình của từng nhóm, loại bỏ nhiễu dữ liệu.")
        
        grade_mapping = {
            'A+B+ (Xuất sắc)': 'A+B+數',
            'A-B+ (Tốt)': 'A-B+數',
            'A-B (Trung bình)': 'A-B數',
            'A-B- (Kém)': 'A-B-數',
            'B+ (Thứ phẩm)': 'B+數'
        }
        features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728'] 
        
        thickness_list = df['厚度歸類'].dropna().unique()
        thickness_list = sorted(thickness_list, key=lambda x: str(x))
        num_thick = len(thickness_list)
        
        for feature in features:
            st.subheader(f"📊 Thông số: {feature}")
            fig, axes = plt.subplots(nrows=1, ncols=num_thick, figsize=(5 * num_thick, 4.5), squeeze=False)
            
            for i, thickness in enumerate(thickness_list):
                ax = axes[0, i]
                df_thick = df[df['厚度歸類'] == thickness]
                has_data = False
                
                min_val = df_thick[feature].min()
                max_val = df_thick[feature].max()
                
                if pd.notna(min_val) and pd.notna(max_val):
                    if min_val == max_val: 
                        min_val -= 1
                        max_val += 1
                    bin_range = (min_val, max_val)
                    bin_width = (max_val - min_val) / 20  
                else:
                    bin_range = None
                    bin_width = 1
                
                for (grade_label, grade_col), color in zip(grade_mapping.items(), colors):
                    temp_df = df_thick[[feature, grade_col]].dropna()
                    temp_df = temp_df[temp_df[grade_col] > 0]
                    
                    if len(temp_df) > 3: 
                        has_data = True
                        values = temp_df[feature].values
                        weights = temp_df[grade_col].values
                        total_weight = weights.sum()
                        
                        # Vẽ phân bố dữ liệu thực (Histogram)
                        sns.histplot(
                            data=temp_df,
                            x=feature,
                            weights=grade_col,
                            label=grade_label,
                            color=color,
                            bins=20,                 
                            binrange=bin_range,      
                            kde=False,               
                            stat="count",            
                            alpha=0.25,
                            linewidth=0.5,
                            ax=ax
                        )
                        
                        # Tính toán và vẽ Normal Curve
                        weighted_mean = np.average(values, weights=weights)
                        weighted_var = np.average((values - weighted_mean)**2, weights=weights)
                        weighted_std = np.sqrt(weighted_var)
                        
                        if weighted_std > 0:
                            x_axis = np.linspace(min_val, max_val, 150)
                            pdf = stats.norm.pdf(x_axis, weighted_mean, weighted_std)
                            scaled_pdf = pdf * total_weight * bin_width
                            ax.plot(x_axis, scaled_pdf, color=color, linewidth=2, alpha=0.9)
                            
                        # Vẽ đường nét đứt Trung bình
                        ax.axvline(weighted_mean, color=color, linestyle='--', linewidth=1.5, alpha=0.9)
                
                if has_data:
                    ax.set_title(f"Độ dày: {thickness}", fontsize=12, fontweight='bold')
                    ax.set_xlabel(f"Giá trị {feature}")
                    ax.set_ylabel("Số lượng cuộn" if i == 0 else "") 
                    ax.grid(axis='y', linestyle=':', alpha=0.6)
                    
                    if i == num_thick - 1:
                        handles, labels = ax.get_legend_handles_labels()
                        unique_labels = list(grade_mapping.keys())
                        unique_handles = [h for h, l in zip(handles, labels) if l in unique_labels]
                        if unique_handles:
                            ax.legend(unique_handles, unique_labels, title="Cấp độ chất lượng", bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    ax.set_title(f"Độ dày: {thickness}\n(Không có dữ liệu)", fontsize=11, color='gray')
                    ax.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("---")

else:
    st.info("Vui lòng tải file Excel dữ liệu của bạn ở thanh công cụ phía trên để bắt đầu phân tích.")
