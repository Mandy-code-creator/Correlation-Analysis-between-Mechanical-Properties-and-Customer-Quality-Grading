import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# Cấu hình trang Dashboard - Ép sử dụng toàn bộ chiều rộng màn hình
st.set_page_config(page_title="QC Data Analysis Dashboard - Max Size", layout="wide")

st.title("📊 Hệ thống Phân tích & Định hình Cơ tính theo Cấp độ Chất lượng")
st.markdown("---")

# 1. Tải file Excel lên
uploaded_file = st.file_uploader("Tải file Excel dữ liệu của bạn lên (Định dạng: .xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Đọc dữ liệu
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip() # Xóa khoảng trắng thừa
    
    # 2. Tiền xử lý dữ liệu (Data Preprocessing)
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    # --- BỘ LỌC CHỐNG NHIỄU DỮ LIỆU CƠ TÍNH ---
    # Chuyển các giá trị bằng 0 hoặc âm thành giá trị Rỗng (NaN) để biểu đồ tự động bỏ qua
    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
    for feat in mech_features:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors='coerce')
            df.loc[df[feat] <= 0, feat] = np.nan
    # ------------------------------------------
            
    df['Total_Count'] = df[count_cols].sum(axis=1)
    df = df[df['Total_Count'] > 0].copy() # Lọc dữ liệu hợp lệ
    
    # Tính Điểm chất lượng trung bình
    df['Quality_Score'] = (5 * df['A+B+數'] + 4 * df['A-B+數'] + 
                           3 * df['A-B數'] + 2 * df['A-B-數'] + 1 * df['B+數']) / df['Total_Count']

    # Tạo 3 Tabs
    tab1, tab2, tab3 = st.tabs(["1. Bảng Thống kê & Tỉ lệ", "2. Ma trận Tương quan", "3. Phân tích Dữ liệu Toàn cảnh (Hiển thị LỚN)"])

    # --- TAB 1: THỐNG KÊ KẾT QUẢ ---
    with tab1:
        st.header("1. Thống kê Phân bổ Chất lượng tổng hợp")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Tổng cuộn kiểm tra'] = summary_df[count_cols].sum(axis=1)
        
        # TÍNH TOÁN PHẦN TRĂM CHO TẤT CẢ CÁC CẤP ĐỘ
        summary_df['% A+B+'] = (summary_df['A+B+數'] / summary_df['Tổng cuộn kiểm tra'] * 100).round(2)
        summary_df['% A-B+'] = (summary_df['A-B+數'] / summary_df['Tổng cuộn kiểm tra'] * 100).round(2)
        summary_df['% A-B'] = (summary_df['A-B數'] / summary_df['Tổng cuộn kiểm tra'] * 100).round(2)
        summary_df['% A-B-'] = (summary_df['A-B-數'] / summary_df['Tổng cuộn kiểm tra'] * 100).round(2)
        summary_df['% B+'] = (summary_df['B+數'] / summary_df['Tổng cuộn kiểm tra'] * 100).round(2)
        
        # Tạo tỷ lệ chia cột (Bảng số liệu chiếm 60% màn hình, Biểu đồ chiếm 40% màn hình) để đủ chỗ hiển thị bảng
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.subheader("Bảng số liệu tổng hợp theo Độ dày")
            
            # FORMAT BẢNG ĐỂ HIỂN THỊ CHUYÊN NGHIỆP HƠN
            display_df = summary_df.copy()
            
            # --- ĐÃ CẬP NHẬT: Đổi tên cột sang Tiếng Anh cho rõ ràng, chuẩn mực ---
            display_df.rename(columns={
                '厚度歸類': 'Thickness',
                'Tổng cuộn kiểm tra': 'Total Coils',
                'A+B+數': 'Count A+B+',
                'A-B+數': 'Count A-B+',
                'A-B數': 'Count A-B',
                'A-B-數': 'Count A-B-',
                'B+數': 'Count B+'
            }, inplace=True)
            
            # Sắp xếp lại thứ tự hiển thị của bảng
            display_cols = [
                'Thickness', 'Total Coils', 
                'Count A+B+', '% A+B+', 
                'Count A-B+', '% A-B+', 
                'Count A-B', '% A-B', 
                'Count A-B-', '% A-B-', 
                'Count B+', '% B+'
            ]
            
            # Hiển thị bảng
            st.dataframe(display_df[display_cols], use_container_width=True)
            
        with col2:
            st.subheader("Biểu đồ tỉ lệ phần trăm")
            
            plot_df = summary_df.set_index('厚度歸類')[count_cols]
            plot_df.columns = ['A+B+', 'A-B+', 'A-B', 'A-B-', 'B+']
            pie_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']
            
            thicknesses = plot_df.index
            n_pies = len(thicknesses)
            
            n_cols_pie = min(2, n_pies)
            n_rows_pie = (n_pies + 1) // 2
            
            fig1, axes1 = plt.subplots(n_rows_pie, n_cols_pie, figsize=(14, 7 * n_rows_pie))
            
            if n_pies == 1:
                axes1_flat = [axes1]
            else:
                axes1_flat = axes1.flatten()
                
            for i, thick in enumerate(thicknesses):
                ax = axes1_flat[i]
                data = plot_df.loc[thick]
                mask = data > 0
                if mask.any():
                    ax.pie(
                        data[mask], 
                        autopct=lambda p: f'{p:.1f}%' if p > 3 else '', 
                        startangle=90, 
                        colors=[c for c, m in zip(pie_colors, mask) if m],
                        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                        textprops={'fontsize': 14, 'fontweight': 'bold'}
                    )
                ax.set_title(f"Độ dày: {thick}", fontsize=18, fontweight='bold')
                
            for j in range(i + 1, len(axes1_flat)):
                axes1_flat[j].axis('off')
                
            fig1.legend(plot_df.columns, title="Cấp độ chất lượng", bbox_to_anchor=(1.0, 0.5), loc="center left", fontsize=14, title_fontsize=16)
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)

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

    # --- TAB 3: PHÂN TÍCH THEO ĐỘ DÀY (KHÔNG CÒN NHIỄU DỮ LIỆU) ---
    with tab3:
        st.header("3. Phân tích Phân phối Toàn cảnh (Hiển thị LỚN)")
        st.markdown("Biểu đồ đã tự động loại bỏ các cuộn thép có cơ tính **bị trống hoặc bằng 0** để đảm bảo đường phân phối chuẩn chính xác tuyệt đối.")
        
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
            st.markdown(f"### 📊 Phân phối Thông số: **{feature}**")
            
            all_feature_data = df[[feature] + count_cols].dropna()
            total_values = []
            for grade_col in count_cols:
                mask = all_feature_data[grade_col] > 0
                if mask.any():
                    vals = all_feature_data[mask][feature].values
                    wgts = all_feature_data[mask][grade_col].values
                    total_values.extend(np.repeat(vals, wgts.astype(int)))
            
            if len(total_values) > 10:
                xmin_total, xmax_total = np.percentile(total_values, [0.5, 99.5]) 
                round_factor = 5 if feature in ['YS', 'TS'] else 0.5
                xmin_total = np.floor(xmin_total / round_factor) * round_factor
                xmax_total = np.ceil(xmax_total / round_factor) * round_factor
                if xmin_total == xmax_total: 
                    xmin_total -= round_factor
                    xmax_total += round_factor
                
                bin_width = (xmax_total - xmin_total) / 20  
                bin_range = (xmin_total, xmax_total)
            else:
                bin_range = None
                bin_width = 1
            
            fig, axes = plt.subplots(nrows=num_thick, ncols=1, figsize=(16, 7 * num_thick))
            
            if num_thick == 1:
                axes = [axes]
                
            for i, thickness in enumerate(thickness_list):
                ax = axes[i]
                df_thick = df[df['厚度歸類'] == thickness]
                has_data = False
                
                for (grade_label, grade_col), color in zip(grade_mapping.items(), colors):
                    temp_df = df_thick[[feature, grade_col]].dropna()
                    temp_df = temp_df[temp_df[grade_col] > 0]
                    
                    if len(temp_df) > 3: 
                        has_data = True
                        values = temp_df[feature].values
                        weights = temp_df[grade_col].values
                        total_weight = weights.sum()
                        
                        sns.histplot(
                            data=temp_df, x=feature, weights=grade_col, label=grade_label,
                            color=color, bins=20, binrange=bin_range, kde=False,               
                            stat="count", alpha=0.3, linewidth=1, ax=ax
                        )
                        
                        weighted_mean = np.average(values, weights=weights)
                        weighted_var = np.average((values - weighted_mean)**2, weights=weights)
                        weighted_std = np.sqrt(weighted_var)
                        
                        if weighted_std > 0 and bin_range is not None:
                            x_axis = np.linspace(bin_range[0], bin_range[1], 150)
                            pdf = stats.norm.pdf(x_axis, weighted_mean, weighted_std)
                            scaled_pdf = pdf * total_weight * bin_width
                            ax.plot(x_axis, scaled_pdf, color=color, linewidth=3, alpha=1.0)
                            
                        ax.axvline(weighted_mean, color=color, linestyle='--', linewidth=2, alpha=0.8)
                
                if has_data and bin_range is not None:
                    ax.set_title(f"Độ dày: {thickness}", fontsize=20, fontweight='bold', pad=15)
                    ax.set_xlabel(f"Giá trị {feature}", fontsize=16)
                    ax.set_ylabel("Số lượng cuộn", fontsize=16) 
                    ax.set_xlim(bin_range) 
                    ax.grid(axis='y', linestyle=':', alpha=0.7)
                    ax.tick_params(axis='both', which='major', labelsize=14)
                    
                    handles, labels = ax.get_legend_handles_labels()
                    unique_labels = list(grade_mapping.keys())
                    unique_handles = [h for h, l in zip(handles, labels) if l in unique_labels]
                    if unique_handles:
                        ax.legend(unique_handles, unique_labels, title="Cấp độ chất lượng", fontsize=14, title_fontsize=14, loc='upper right')
                else:
                    ax.set_title(f"Độ dày: {thickness}\n(Không đủ dữ liệu hợp lệ)", fontsize=18, color='gray')
                    ax.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True) 
            st.markdown("---")

else:
    st.info("Vui lòng tải file Excel dữ liệu của bạn ở thanh công cụ phía trên để bắt đầu phân tích.")
