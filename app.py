import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Mechanical Properties QC Dashboard", layout="wide")

st.title("📊 Mechanical Properties QC Analysis Dashboard")
st.markdown("---")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your Excel data (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    exported_figures = {} # Dictionary to store charts for Excel export

    # --- 2. DATA PREPROCESSING ---
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    count_cols = [col for col in count_cols if col in df.columns]
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
    mech_features = [feat for feat in mech_features if feat in df.columns]
    for feat in mech_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
        # Clean data: remove 0 or negative values
        df.loc[df[feat] <= 0, feat] = np.nan

    df['Total_Count'] = df[count_cols].sum(axis=1)
    df = df[df['Total_Count'] > 0].copy()

    # --- QUALITY SCORE ---
    df['Quality_Score'] = (5 * df.get('A+B+數',0) + 4 * df.get('A-B+數',0) + 
                           3 * df.get('A-B數',0) + 2 * df.get('A-B-數',0) + 1 * df.get('B+數',0)) / df['Total_Count']

    # --- CREATE TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Summary Table & Pie Charts", 
        "2. Correlation Matrix", 
        "3. Weighted Distribution by Thickness",
        "4. Optimal Mechanical Limits"
    ])

    # --- TAB 1 ---
    with tab1:
        st.header("1. Summary by Thickness")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Total Coils'] = summary_df[count_cols].sum(axis=1)

        for col in count_cols:
            summary_df[f'% {col}'] = (summary_df[col] / summary_df['Total Coils'] * 100).round(2)

        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.subheader("Summary Table")
            display_df = summary_df.copy()
            display_df.rename(columns={
                '厚度歸類': 'Thickness',
                'A+B+數': 'Count A+B+',
                'A-B+數': 'Count A-B+',
                'A-B數': 'Count A-B',
                'A-B-數': 'Count A-B-',
                'B+數': 'Count B+'
            }, inplace=True)
            st.dataframe(display_df, use_container_width=True)

        with col2:
            st.subheader("Percentage Pie Charts")
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
                    ax.pie(data[mask], autopct=lambda p: f'{p:.1f}%' if p>3 else '', startangle=90, colors=[c for c,m in zip(pie_colors, mask) if m], wedgeprops={'edgecolor':'white','linewidth':2}, textprops={'fontsize':14,'fontweight':'bold'})
                ax.set_title(f"Thickness: {thick}", fontsize=18, fontweight='bold')

            for j in range(i+1, len(axes1_flat)):
                axes1_flat[j].axis('off')

            fig1.legend(plot_df.columns, title="Quality Grade", bbox_to_anchor=(1.0,0.5), loc="center left", fontsize=14, title_fontsize=16)
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            exported_figures['Quality_Proportions'] = fig1

    # --- TAB 2 ---
    with tab2:
        st.header("2. Correlation Matrix")
        corr_matrix = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        corr_matrix.columns = ['Correlation with Total Quality Score']

        col_corr1, col_corr2 = st.columns([1,2])
        with col_corr1:
            st.subheader("Pearson Correlation")
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=0), use_container_width=True)
        with col_corr2:
            st.subheader("Interpretation")
            if not corr_matrix.empty:
                max_corr_feature = corr_matrix.idxmin()[0]
                st.info(f"Parameter **{max_corr_feature}** has the most negative effect on total quality score.")

    # --- TAB 3 ---
    with tab3:
        st.header("3. Weighted Distribution by Thickness")
        grade_mapping = {
            'A+B+ (Excellent)': 'A+B+數',
            'A-B+ (Good)': 'A-B+數',
            'A-B (Average)': 'A-B數',
            'A-B- (Poor)': 'A-B-數',
            'B+ (Reject)': 'B+數'
        }
        colors = ['#2ca02c','#1f77b4','#ff7f0e','#9467bd','#d62728']
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)

        for feature in mech_features:
            with st.expander(f"📊 Distribution: {feature}", expanded=False):
                fig, axes = plt.subplots(len(thickness_list), 1, figsize=(16,6*len(thickness_list)))
                axes = axes if len(thickness_list) >1 else [axes]

                for i, thickness in enumerate(thickness_list):
                    ax = axes[i]
                    df_thick = df[df['厚度歸類']==thickness]
                    has_data = False
                    for (grade_label, grade_col), color in zip(grade_mapping.items(), colors):
                        if grade_col not in df_thick.columns: continue
                        temp_df = df_thick[[feature, grade_col]].dropna()
                        temp_df = temp_df[temp_df[grade_col]>0]
                        if len(temp_df)>3:
                            has_data = True
                            values = temp_df[feature].values
                            weights = temp_df[grade_col].values
                            total_weight = weights.sum()
                            sns.histplot(temp_df, x=feature, weights=grade_col, label=grade_label, color=color, bins=20, kde=False, stat='count', alpha=0.3, ax=ax)
                            weighted_mean = np.average(values, weights=weights)
                            weighted_std = np.sqrt(np.average((values-weighted_mean)**2, weights=weights))
                            if weighted_std>0:
                                x_axis = np.linspace(values.min(), values.max(),150)
                                pdf = stats.norm.pdf(x_axis, weighted_mean, weighted_std)
                                scaled_pdf = pdf * total_weight * ((values.max()-values.min())/20)
                                ax.plot(x_axis, scaled_pdf, color=color, linewidth=3, alpha=0.8)
                            ax.axvline(weighted_mean, color=color, linestyle='--', linewidth=2, alpha=0.8)
                    if has_data:
                        ax.set_title(f"Thickness: {thickness}")
                        ax.set_xlabel(feature)
                        ax.set_ylabel("Number of Coils")
                        ax.grid(axis='y', linestyle=':', alpha=0.7)
                        ax.legend(title="Quality Grade")
                    else:
                        ax.set_title(f"Thickness: {thickness} - Not enough data", color='gray')
                        ax.axis('off')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                exported_figures[f'{feature}_Distribution'] = fig

    # --- TAB 4 ---
    with tab4:
        st.header("4. Optimal Mechanical Limits")
        limits = []
        for feat in mech_features:
            all_vals = []
            # Drop NaNs beforehand so np.percentile doesn't fail
            valid_df = df.dropna(subset=[feat]) 
            
            for grade_col in count_cols:
                mask = valid_df[grade_col] > 0
                if mask.any():
                    vals = valid_df.loc[mask, feat].values
                    w = valid_df.loc[mask, grade_col].values
                    all_vals.extend(np.repeat(vals, w.astype(int)))
                    
            if len(all_vals) > 0:
                arr = np.array(all_vals)
                lower = np.percentile(arr, 2.5)
                upper = np.percentile(arr, 97.5)
                mean = np.mean(arr)
                std = np.std(arr)
                limits.append({'Parameter': feat, 'Lower Limit': round(lower, 2), 'Upper Limit': round(upper, 2), 'Weighted Mean': round(mean, 2), 'Weighted Std': round(std, 2)})
                
        limits_df = pd.DataFrame(limits)
        st.dataframe(limits_df, use_container_width=True)
        st.markdown("*Optimal limits are based on weighted 2.5%–97.5% percentile of the actual distribution.*")

    # --- FULL EXCEL EXPORT ---
    st.markdown("## 📥 Export Full Report")
    
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        display_df.to_excel(writer, sheet_name='Summary_Data', index=False)
        corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
        if not limits_df.empty:
            limits_df.to_excel(writer, sheet_name='Optimal_Limits', index=False)
        
        workbook = writer.book
        worksheet_charts = workbook.add_worksheet('Generated_Charts')
        worksheet_charts.write('A1', 'QC Analysis Charts')
        
        row_idx = 2
        for chart_name, fig_obj in exported_figures.items():
            img_buffer = io.BytesIO()
            fig_obj.savefig(img_buffer, format='png', bbox_inches='tight')
            worksheet_charts.write(row_idx, 1, chart_name.replace('_', ' '))
            worksheet_charts.insert_image(row_idx + 2, 1, '', {'image_data': img_buffer, 'x_scale': 0.6, 'y_scale': 0.6})
            row_idx += 40
            
    excel_buffer.seek(0)
    st.download_button(
        label="📥 Download Full Excel Report (Data & Charts)",
        data=excel_buffer,
        file_name="QC_Analysis_Full_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Please upload your Excel data to start analysis.")
