import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import io # ĐÃ THÊM: Bắt buộc phải có để tải file Excel

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QC Mechanical Properties Dashboard", layout="wide")
st.title("📊 QC Mechanical Properties Analysis")
st.markdown("---")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()  # remove extra spaces

    # --- 2. DATA PREPROCESSING ---
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    count_cols = [col for col in count_cols if col in df.columns]
    
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
    mech_features = [feat for feat in mech_features if feat in df.columns]
    
    for feat in mech_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
        df.loc[df[feat] <= 0, feat] = np.nan

    df['Total_Count'] = df[count_cols].sum(axis=1)
    df = df[df['Total_Count'] > 0].copy()

    # --- QUALITY SCORE ---
    df['Quality_Score'] = (5*df.get('A+B+數', 0) + 4*df.get('A-B+數', 0) + 3*df.get('A-B數', 0) +
                           2*df.get('A-B-數', 0) + 1*df.get('B+數', 0)) / df['Total_Count']

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Summary & Percentages",
        "2. Correlation Matrix",
        "3. Weighted Distribution",
        "4. Optimal & Safe Limits"
    ])

    # --- TAB 1: SUMMARY ---
    with tab1:
        st.header("1. Summary by Thickness")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Total Coils'] = summary_df[count_cols].sum(axis=1)
        
        for col in count_cols:
            summary_df[f"% {col}"] = (summary_df[col]/summary_df['Total Coils']*100).round(2)
            
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.subheader("Summary Table")
            display_df = summary_df.copy()
            display_df.rename(columns={'厚度歸類': 'Thickness'}, inplace=True)
            st.dataframe(display_df, use_container_width=True)

            # Download button (Đã được cấp quyền bởi lệnh import io)
            towrite = io.BytesIO()
            display_df.to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button("📥 Download Excel", data=towrite, file_name="Summary_QC.xlsx")

        with col2:
            st.subheader("Percentage Pie Charts")
            plot_df = summary_df.set_index('厚度歸類')[count_cols]
            plot_df.columns = ['A+B+', 'A-B+', 'A-B', 'A-B-', 'B+']
            pie_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']

            thicknesses = plot_df.index
            n_pies = len(thicknesses)
            
            if n_pies > 0:
                n_cols_pie = min(2, n_pies)
                n_rows_pie = (n_pies + n_cols_pie - 1) // n_cols_pie

                fig1, axes1 = plt.subplots(n_rows_pie, n_cols_pie, figsize=(14, 7 * n_rows_pie))
                axes1_flat = np.array(axes1).flatten() if n_pies > 1 else [axes1]

                for i, thick in enumerate(thicknesses):
                    ax = axes1_flat[i]
                    data = plot_df.loc[thick]
                    mask = data > 0
                    if mask.any():
                        ax.pie(data[mask], autopct=lambda p: f'{p:.1f}%' if p>3 else '', startangle=90, 
                               colors=[c for c,m in zip(pie_colors, mask) if m], 
                               wedgeprops={'edgecolor':'white','linewidth':2}, 
                               textprops={'fontsize':14,'fontweight':'bold'})
                    ax.set_title(f"Thickness: {thick}", fontsize=18, fontweight='bold')

                for j in range(i+1, len(axes1_flat)):
                    axes1_flat[j].axis('off')

                fig1.legend(plot_df.columns, title="Quality Grade", bbox_to_anchor=(1.0,0.5), loc="center left", fontsize=14, title_fontsize=16)
                plt.tight_layout()
                st.pyplot(fig1, use_container_width=True)

    # --- TAB 2: CORRELATION ---
    with tab2:
        st.header("2. Correlation Matrix")
        corr_matrix = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        corr_matrix.columns = ['Correlation with Quality Score']
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=0), use_container_width=True)

    # --- TAB 4: OPTIMAL & SAFE LIMITS (Tính trước để vẽ ở Tab 3) ---
    with tab4:
        st.header("4. Optimal & Safe Limits")
        limits = []
        for feat in mech_features:
            # --- Optimal Limit by percentile ---
            all_vals = []
            for grade_col in count_cols:
                temp = df[[feat, grade_col]].dropna()
                temp = temp[temp[grade_col]>0]
                if len(temp)>0:
                    vals = temp[feat].values
                    wgts = temp[grade_col].values
                    all_vals.extend(np.repeat(vals, wgts.astype(int)))
                    
            if len(all_vals) > 10:
                lower_opt = np.percentile(all_vals, 2.5)
                upper_opt = np.percentile(all_vals, 97.5)
            else:
                lower_opt, upper_opt = np.nan, np.nan

            # --- Safe Limit from weighted mean/std of good grades only ---
            all_good_vals = []
            for grade_col in ['A+B+數','A-B+數']:
                if grade_col in df.columns:
                    temp = df[[feat, grade_col]].dropna()
                    temp = temp[temp[grade_col]>0]
                    if len(temp)>0:
                        vals = temp[feat].values
                        wgts = temp[grade_col].values
                        all_good_vals.extend(np.repeat(vals, wgts.astype(int)))
                        
            if len(all_good_vals) > 5:
                w_mean = np.average(all_good_vals)
                w_std = np.std(all_good_vals)
                lower_safe = w_mean - 2*w_std
                upper_safe = w_mean + 2*w_std
            else:
                lower_safe, upper_safe = np.nan, np.nan

            limits.append([feat, lower_opt, upper_opt, lower_safe, upper_safe])

        limits_df = pd.DataFrame(limits, columns=['Parameter', 'Optimal Lower','Optimal Upper', 'Safe Lower','Safe Upper'])
        st.dataframe(limits_df, use_container_width=True)

    # --- TAB 3: WEIGHTED DISTRIBUTION WITH LIMITS ---
    with tab3:
        st.header("3. Weighted Distribution by Thickness with Limits")
        grade_mapping = {
            'A+B+ (Excellent)': 'A+B+數',
            'A-B+ (Good)': 'A-B+數',
            'A-B (Average)': 'A-B數',
            'A-B- (Poor)': 'A-B-數',
            'B+ (Reject)': 'B+數'
        }
        colors = ['#2ca02c','#1f77b4','#ff7f0e','#9467bd','#d62728']
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)

        # Dictionary of limits
        limits_dict = {row['Parameter']:(row['Optimal Lower'],row['Optimal Upper'],row['Safe Lower'],row['Safe Upper'])
                       for idx,row in limits_df.iterrows()}

        for feature in mech_features:
            with st.expander(f"📊 Distribution: {feature}", expanded=True):
                if len(thickness_list) == 0:
                    continue
                    
                fig, axes = plt.subplots(len(thickness_list), 1, figsize=(16, 6*len(thickness_list)))
                axes = axes if len(thickness_list)>1 else [axes]

                for i, thickness in enumerate(thickness_list):
                    ax = axes[i]
                    df_thick = df[df['厚度歸類']==thickness]
                    has_data = False

                    for (grade_label, grade_col), color in zip(grade_mapping.items(), colors):
                        if grade_col not in df_thick.columns:
                            continue
                        temp_df = df_thick[[feature, grade_col]].dropna()
                        temp_df = temp_df[temp_df[grade_col]>0]
                        if len(temp_df) > 3:
                            has_data = True
                            values = temp_df[feature].values
                            weights = temp_df[grade_col].values
                            total_weight = weights.sum()
                            
                            sns.histplot(temp_df, x=feature, weights=grade_col, label=grade_label,
                                         color=color, bins=20, kde=False, stat='count', alpha=0.3, ax=ax)

                            # Weighted distribution line
                            weighted_mean = np.average(values, weights=weights)
                            weighted_std = np.sqrt(np.average((values-weighted_mean)**2, weights=weights))
                            
                            if weighted_std > 0:
                                x_axis = np.linspace(values.min(), values.max(), 150)
                                pdf = stats.norm.pdf(x_axis, weighted_mean, weighted_std)
                                # ĐÃ SỬA: Chống lỗi values.max() == values.min() làm triệt tiêu đường cong
                                val_range = max(values.max() - values.min(), 1)
                                scaled_pdf = pdf * total_weight * (val_range / 20)
                                ax.plot(x_axis, scaled_pdf, color=color, linewidth=2, alpha=0.8)
                                
                            ax.axvline(weighted_mean, color=color, linestyle='--', linewidth=1, alpha=0.8)

                    # Draw limits
                    if feature in limits_dict:
                        opt_lower, opt_upper, safe_lower, safe_upper = limits_dict[feature]
                        if not np.isnan(opt_lower):
                            ax.axvline(opt_lower, color='blue', linestyle='-.', linewidth=2.5, label='Optimal Lower')
                        if not np.isnan(opt_upper):
                            ax.axvline(opt_upper, color='blue', linestyle='-.', linewidth=2.5, label='Optimal Upper')
                        if not np.isnan(safe_lower):
                            ax.axvline(safe_lower, color='green', linestyle='-', linewidth=2.5, label='Safe Lower')
                        if not np.isnan(safe_upper):
                            ax.axvline(safe_upper, color='green', linestyle='-', linewidth=2.5, label='Safe Upper')

                    if not has_data:
                        ax.set_title(f"Thickness: {thickness} - Not enough data", color='gray')
                        ax.axis('off')
                    else:
                        ax.set_title(f"Thickness: {thickness}")
                        ax.set_xlabel(feature)
                        ax.set_ylabel("Number of Coils")
                        ax.grid(axis='y', linestyle=':', alpha=0.7)
                        
                        # ĐÃ SỬA LỖI LẶP LEGEND: Gộp các chú thích bị trùng lặp
                        handles, labels = ax.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        ax.legend(by_label.values(), by_label.keys(), title="Grade / Limit", loc='upper right')

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

else:
    st.info("Please upload an Excel file to start analysis.")
