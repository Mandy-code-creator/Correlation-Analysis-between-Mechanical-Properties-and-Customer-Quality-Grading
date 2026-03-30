import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QC Data Analysis Dashboard - Optimized", layout="wide")

st.title("📊 Hệ thống Phân tích & Định hình Cơ tính theo Cấp độ Chất lượng")
st.markdown("---")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Tải file Excel dữ liệu của bạn lên (Định dạng: .xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    # --- 2. DATA PREPROCESSING ---
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    count_cols = [col for col in count_cols if col in df.columns]  # đảm bảo cột tồn tại
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
    mech_features = [feat for feat in mech_features if feat in df.columns]
    for feat in mech_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
        df.loc[df[feat] <= 0, feat] = np.nan

    df['Total_Count'] = df[count_cols].sum(axis=1)
    df = df[df['Total_Count'] > 0].copy()

    # --- 3. QUALITY SCORE ---
    df['Quality_Score'] = (5 * df.get('A+B+數',0) + 4 * df.get('A-B+數',0) + 
                           3 * df.get('A-B數',0) + 2 * df.get('A-B-數',0) + 1 * df.get('B+數',0)) / df['Total_Count']

    # --- 4. CREATE TABS ---
    tab1, tab2, tab3 = st.tabs(["1. Bảng Thống kê & Tỉ lệ", "2. Ma trận Tương quan", "3. Phân tích Toàn cảnh"])

    # --- TAB 1 ---
    with tab1:
        st.header("1. Thống kê Phân bổ Chất lượng tổng hợp")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Tổng cuộn kiểm tra'] = summary_df[count_cols].sum(axis=1)

        for col in count_cols:
            summary_df[f'% {col}'] = (summary_df[col] / summary_df['Tổng cuộn kiểm tra'] * 100).round(2)

        col1, col2 = st.columns([1.5, 1])

        with col1:
            st.subheader("Bảng số liệu tổng hợp theo Độ dày")
            display_df = summary_df.copy()
            display_df.rename(columns={'厚度歸類': 'Thickness'}, inplace=True)
            st.dataframe(display_df, use_container_width=True)

            # Download button
            towrite = io.BytesIO()
            display_df.to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button("📥 Tải bảng thống kê Excel", data=towrite, file_name="Summary_QC.xlsx")

        with col2:
            st.subheader("Biểu đồ tỉ lệ phần trăm")
            plot_df = summary_df.set_index('厚度歸類')[count_cols]
            plot_df.columns = ['A+B+', 'A-B+', 'A-B', 'A-B-', 'B+']
            pie_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']

            thicknesses = plot_df.index
            n_pies = len(thicknesses)
            n_cols_pie = min(2, n_pies)
            n_rows_pie = (n_pies + n_cols_pie - 1) // n_cols_pie

            fig1, axes1 = plt.subplots(n_rows_pie, n_cols_pie, figsize=(14, 7 * n_rows_pie))
            axes1_flat = np.array(axes1).flatten() if n_pies > 1 else [axes1]

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

    # --- TAB 2 ---
    with tab2:
        st.header("2. Ma trận Hệ số Tương quan")
        corr_matrix = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        corr_matrix.columns = ['Độ tương quan với Điểm Chất lượng Tổng']

        col_corr1, col_corr2 = st.columns([1, 2])
        with col_corr1:
            st.subheader("Bảng Hệ số Tương quan (Pearson)")
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=0), use_container_width=True)

        with col_corr2:
            st.subheader("Giải thích")
            if not corr_matrix.empty:
                max_corr_feature = corr_matrix.idxmin()[0]
                st.info(f"Thông số **{max_corr_feature}** ảnh hưởng tiêu cực nhất đến điểm chất lượng.")

    # --- TAB 3 ---
    with tab3:
        st.header("3. Phân tích Phân phối Toàn cảnh")
        st.markdown("Các cuộn thép cơ tính <= 0 đã được loại bỏ để đảm bảo tính chuẩn xác.")

        grade_mapping = {
            'A+B+ (Xuất sắc)': 'A+B+數',
            'A-B+ (Tốt)': 'A-B+數',
            'A-B (Trung bình)': 'A-B數',
            'A-B- (Kém)': 'A-B-數',
            'B+ (Thứ phẩm)': 'B+數'
        }

        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)

        for feature in mech_features:
            with st.expander(f"📊 Phân phối Thông số: {feature}", expanded=False):
                fig, axes = plt.subplots(len(thickness_list), 1, figsize=(16, 6 * len(thickness_list)))
                axes = axes if len(thickness_list) > 1 else [axes]

                for i, thickness in enumerate(thickness_list):
                    ax = axes[i]
                    df_thick = df[df['厚度歸類'] == thickness]
                    has_data = False

                    for (grade_label, grade_col), color in zip(grade_mapping.items(), colors):
                        if grade_col not in df_thick.columns:
                            continue
                        temp_df = df_thick[[feature, grade_col]].dropna()
                        temp_df = temp_df[temp_df[grade_col] > 0]
                        if len(temp_df) > 3:
                            has_data = True
                            values = temp_df[feature].values
                            weights = temp_df[grade_col].values
                            total_weight = weights.sum()

                            sns.histplot(
                                data=temp_df, x=feature, weights=grade_col, label=grade_label,
                                color=color, bins=20, kde=False, stat="count", alpha=0.3, ax=ax
                            )

                            weighted_mean = np.average(values, weights=weights)
                            weighted_std = np.sqrt(np.average((values - weighted_mean)**2, weights=weights))

                            if weighted_std > 0:
                                x_axis = np.linspace(values.min(), values.max(), 150)
                                pdf = stats.norm.pdf(x_axis, weighted_mean, weighted_std)
                                scaled_pdf = pdf * total_weight * ((values.max()-values.min())/20)
                                ax.plot(x_axis, scaled_pdf, color=color, linewidth=3, alpha=0.8)
                            ax.axvline(weighted_mean, color=color, linestyle='--', linewidth=2, alpha=0.8)

                    if has_data:
                        ax.set_title(f"Độ dày: {thickness}", fontsize=16, fontweight='bold')
                        ax.set_xlabel(feature)
                        ax.set_ylabel("Số lượng cuộn")
                        ax.grid(axis='y', linestyle=':', alpha=0.7)
                        ax.legend(title="Cấp độ chất lượng")
                    else:
                        ax.set_title(f"Độ dày: {thickness} - Không đủ dữ liệu", fontsize=14, color='gray')
                        ax.axis('off')

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

else:
    st.info("Vui lòng tải file Excel dữ liệu của bạn để bắt đầu phân tích.")
