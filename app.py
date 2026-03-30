import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# Cấu hình trang Dashboard
st.set_page_config(page_title="QC Data Analysis Dashboard - Enhanced", layout="wide")

st.title("📊 Hệ thống Phân tích & Định hình Cơ tính theo Cấp độ Chất lượng - Phiên bản Rõ ràng")
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
    tab1, tab2, tab3 = st.tabs(["1. Bảng Thống kê & Tỉ lệ", "2. Ma trận Tương quan", "3. Phân tích Dữ liệu Toàn cảnh (SPC Clear View)"])

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
            
            # 1. Lấy dữ liệu và đổi tên cột cho đẹp
            plot_df = summary_df.set_index('厚度歸類')[count_cols]
            plot_df.columns = ['A+B+', 'A-B+', 'A-B', 'A-B-', 'B+']
            
            # Đồng bộ màu sắc với Tab 3
            pie_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']
            
            # 2. Khởi tạo chuỗi biểu đồ tròn
            thicknesses = plot_df.index
            n_pies = len(thicknesses)
            
            # Tăng kích thước Figsize đáng kể để Pie charts to hơn
            fig1, axes1 = plt.subplots(1, n_pies, figsize=(4.5 * n_pies, 4.5))
            
            # Xử lý trường hợp chỉ có 1 độ dày
            if n_pies == 1:
                axes1 = [axes1]
                
            # 3. Vẽ từng biểu đồ tròn
            for ax, thick in zip(axes1, thicknesses):
                data = plot_df.loc[thick]
                
                mask = data > 0
                if mask.any():
                    ax.pie(
                        data[mask], 
                        autopct=lambda p: f'{p:.1f}%' if p > 3 else '', # Chỉ hiện số nếu > 3%
                        startangle=90, 
                        colors=[c for c, m in zip(pie_colors, mask) if m],
                        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
                    )
                ax.set_title(f"Độ dày: {thick}", fontsize=14, fontweight='bold')
                
            fig1.legend(plot_df.columns, title="Cấp độ chất lượng", bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=11)
            plt.tight_layout()
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

    # --- TAB 3: PHÂN TÍCH THEO ĐỘ DÀY & CẤP ĐỘ CHẤT LƯỢNG (BẢN CLAR VIEW - TỰ ĐỘNG PHÓNG TO) ---
    with tab3:
        st.header("3. Phân tích Phân phối Toàn cảnh (SPC - Tự động Phóng to)")
        st.markdown("Hệ thống tự động phân rã tất cả dữ liệu theo **Độ dày** (Subplots) và vẽ **Cấp độ chất lượng** phân bố chồng lên nhau. Biểu đồ **tự động phóng to trục X** để vùng phân phối chính trông rõ ràng nhất.")
        
        grade_mapping = {
            'A+B+ (Xuất sắc)': 'A+B+數',
            'A-B+ (Tốt)': 'A-B+數',
            'A-B (Trung bình)': 'A-B數',
            'A-B- (Kém)': 'A-B-數',
            'B+ (Thứ phẩm)': 'B+數'
        }
        features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
        # Đồng bộ màu sắc
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728'] 
        
        thickness_list = df['厚度歸類'].dropna().unique()
        thickness_list = sorted(thickness_list, key=lambda x: str(x))
        num_thick = len(thickness_list)
        
        for feature in features:
            st.markdown(f"### 📊 Thông số cơ tính: **{feature}**")
            
            # --- TÍNH TOÁN PHẠM VI DỮ LIỆU TỔNG THỂ CHO FEATURE NÀY ĐỂ AUTO-ZOOM ---
            all_feature_data = df[[feature] + count_cols].dropna()
            total_values = []
            for grade_col in count_cols:
                mask = all_feature_data[grade_col] > 0
                if mask.any():
                    vals = all_feature_data[mask][feature].values
                    wgts = all_feature_data[mask][grade_col].values
                    # Lặp lại giá trị theo trọng số để tạo tập dữ liệu tổng thể đúng
                    total_values.extend(np.repeat(vals, wgts.astype(int)))
            
            if len(total_values) > 10:
                # Tính khoảng phân vị 1% - 99% để loại bỏ outliers gây nhiễu và phóng to vùng chính
                xmin_total, xmax_total = np.percentile(total_values, [1, 99])
                
                # Làm tròn phạm vi một chút cho đẹp (ví dụ YS 342-498 => 340-500)
                round_factor = 5 if feature in ['YS', 'TS'] else 0.5
                xmin_total = np.floor(xmin_total / round_factor) * round_factor
                xmax_total = np.ceil(xmax_total / round_factor) * round_factor
                
                if xmin_total == xmax_total: # Xử lý lỗi nếu các giá trị đều giống y hệt nhau
                    xmin_total -= round_factor
                    xmax_total += round_factor
                
                bin_width = (xmax_total - xmin_total) / 20  
                bin_range = (xmin_total, xmax_total)
            else:
                bin_range = None
                bin_width = 1
            
            # --- CẤU HÌNH SUBPLOTS TO, RÕ RÀNG (CHIA NHIỀU HÀNG NẾU CẦN) ---
            # Tăng figsize đáng kể, chiều cao Subplot to hơn
            n_cols = num_thick
            n_rows = (n_cols + 1) // 2 # Tối đa 2 Subplot/hàng
            
            # Tăng height đáng kể
            fig, axes = plt.subplots(nrows=n_rows, ncols=min(2, n_cols), figsize=(12 if n_cols == 1 else 20, 6 * n_rows), squeeze=False)
            
            axes_flat = axes.flatten()
            
            for i, thickness in enumerate(thickness_list):
                if i >= len(axes_flat): break # Đề phòng lỗi
                ax = axes_flat[i]
                df_thick = df[df['厚度歸類'] == thickness]
                has_data = False
                
                # Vẽ từng cấp độ chất lượng
                for (grade_label, grade_col), color in zip(grade_mapping.items(), colors):
                    temp_df = df_thick[[feature, grade_col]].dropna()
                    temp_df = temp_df[temp_df[grade_col] > 0]
                    
                    if len(temp_df) > 3: 
                        has_data = True
                        values = temp_df[feature].values
                        weights = temp_df[grade_col].values
                        total_weight = weights.sum()
                        
                        # Vẽ phân bố dữ liệu thực (Histogram) - ĐÃ CẢI THIỆN CHIỀU CAO CỘT
                        sns.histplot(
                            data=temp_df,
                            x=feature,
                            weights=grade_col,
                            label=grade_label,
                            color=color,
                            bins=20,                 
                            binrange=bin_range,      # Sử dụng phạm vi Auto-Zoom
                            kde=False,               
                            stat="count",            # Trục Y là số lượng cuộn
                            alpha=0.25,
                            linewidth=0.5,
                            ax=ax
                        )
                        
                        # Tính toán tham số Thống kê có trọng số
                        weighted_mean = np.average(values, weights=weights)
                        weighted_var = np.average((values - weighted_mean)**2, weights=weights)
                        weighted_std = np.sqrt(weighted_var)
                        
                        # Vẽ đường Phân phối chuẩn toán học (Normal Curve)
                        if weighted_std > 0 and bin_range is not None:
                            x_axis = np.linspace(bin_range[0], bin_range[1], 150)
                            pdf = stats.norm.pdf(x_axis, weighted_mean, weighted_std)
                            # Đổi hệ số PDF sang quy mô đếm số lượng cột
                            scaled_pdf = pdf * total_weight * bin_width
                            # Vẽ đường mượt mà, TO VÀ RÕ RÀNG
                            ax.plot(x_axis, scaled_pdf, color=color, linewidth=2.5, alpha=0.95)
                            
                        # Vẽ đường nét đứt Trung bình
                        ax.axvline(weighted_mean, color=color, linestyle='--', linewidth=1.5, alpha=0.9)
                
                # Trang trí trục và tiêu đề TO RÕ RÀNG
                if has_data and bin_range is not None:
                    ax.set_title(f"Độ dày: {thickness}", fontsize=15, fontweight='bold')
                    ax.set_xlabel(f"Giá trị {feature}", fontsize=12)
                    # Nhãn trục Y TO RÕ RÀNG
                    ax.set_ylabel("Số lượng cuộn", fontsize=12) 
                    ax.set_xlim(bin_range) # ÁP DỤNG AUTO-ZOOM
                    ax.grid(axis='y', linestyle=':', alpha=0.7)
                    
                    # Chỉ hiển thị Legend chung ở Subplot cuối cùng để không bị rối
                    if i == num_thick - 1:
                        handles, labels = ax.get_legend_handles_labels()
                        unique_labels = list(grade_mapping.keys())
                        unique_handles = [h for h, l in zip(handles, labels) if l in unique_labels]
                        if unique_handles:
                            ax.legend(unique_handles, unique_labels, title="Cấp độ chất lượng", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                else:
                    ax.set_title(f"Độ dày: {thickness}\n(Không có dữ liệu)", fontsize=13, color='gray')
                    ax.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("---")

else:
    st.info("Vui lòng tải file Excel dữ liệu của bạn ở thanh công cụ phía trên để bắt đầu phân tích.")
