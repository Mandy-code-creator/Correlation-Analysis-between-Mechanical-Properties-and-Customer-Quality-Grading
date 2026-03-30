import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import io

# Page Configuration
st.set_page_config(page_title="QC Data Analysis Dashboard", layout="wide")

st.title("📊 Quality Grade & Mechanical Properties Analysis System")
st.markdown("---")

# 1. File Upload
uploaded_file = st.file_uploader("Upload your Excel data file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Read Data
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip() 
    
    # Dictionary to store generated figures for Excel export
    exported_figures = {}
    
    # 2. Data Preprocessing
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    # Data Cleaning: Convert 0 or negative values to NaN to avoid noise
    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
    for feat in mech_features:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors='coerce')
            df.loc[df[feat] <= 0, feat] = np.nan
            
    df['Total_Count'] = df[count_cols].sum(axis=1)
    df = df[df['Total_Count'] > 0].copy() 
    
    # Calculate Average Quality Score (Scale 1-5)
    df['Quality_Score'] = (5 * df['A+B+數'] + 4 * df['A-B+數'] + 
                           3 * df['A-B數'] + 2 * df['A-B-數'] + 1 * df['B+數']) / df['Total_Count']

    # Create 3 Tabs
    tab1, tab2, tab3 = st.tabs(["1. Statistics & Proportions", "2. Correlation Matrix", "3. Full Distribution Analysis (Large View)"])

    # --- TAB 1: STATISTICS ---
    with tab1:
        st.header("1. Overall Quality Distribution")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Total Coils'] = summary_df[count_cols].sum(axis=1)
        
        # Calculate Percentages
        summary_df['% A+B+'] = (summary_df['A+B+數'] / summary_df['Total Coils'] * 100).round(2)
        summary_df['% A-B+'] = (summary_df['A-B+數'] / summary_df['Total Coils'] * 100).round(2)
        summary_df['% A-B'] = (summary_df['A-B數'] / summary_df['Total Coils'] * 100).round(2)
        summary_df['% A-B-'] = (summary_df['A-B-數'] / summary_df['Total Coils'] * 100).round(2)
        summary_df['% B+'] = (summary_df['B+數'] / summary_df['Total Coils'] * 100).round(2)
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.subheader("Aggregated Data by Thickness")
            
            # Format DataFrame for display and export
            display_df = summary_df.copy()
            display_df.rename(columns={
                '厚度歸類': 'Thickness',
                'A+B+數': 'Count A+B+',
                'A-B+數': 'Count A-B+',
                'A-B數': 'Count A-B',
                'A-B-數': 'Count A-B-',
                'B+數': 'Count B+'
            }, inplace=True)
            
            display_cols = [
                'Thickness', 'Total Coils', 
                'Count A+B+', '% A+B+', 
                'Count A-B+', '% A-B+', 
                'Count A-B', '% A-B', 
                'Count A-B-', '% A-B-', 
                'Count B+', '% B+'
            ]
            
            st.dataframe(display_df[display_cols], use_container_width=True)
            
        with col2:
            st.subheader("Percentage Distribution Chart")
            
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
                ax.set_title(f"Thickness: {thick}", fontsize=18, fontweight='bold')
                
            for j in range(i + 1, len(axes1_flat)):
                axes1_flat[j].axis('off')
                
            fig1.legend(plot_df.columns, title="Quality Grade", bbox_to_anchor=(1.0, 0.5), loc="center left", fontsize=14, title_fontsize=16)
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            
            # Save Pie chart for export
            exported_figures['Quality_Proportions'] = fig1

    # --- TAB 2: CORRELATION MATRIX ---
    with tab2:
        st.header("2. Correlation Matrix")
        corr_matrix = df[['Quality_Score'] + features].corr()[['Quality_Score']].drop('Quality_Score')
        corr_matrix.columns = ['Correlation with Total Quality Score']
        
        col_corr1, col_corr2 = st.columns([1, 2])
        with col_corr1:
            st.subheader("Pearson Correlation Table")
            st.markdown("*(A coefficient closer to -1 indicates higher mechanical property value correlates with lower quality)*")
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=0), use_container_width=True)
            
        with col_corr2:
            st.subheader("Explanation")
            max_corr_feature = corr_matrix.idxmin()[0]
            st.info(f"Based on the data, the parameter **{max_corr_feature}** has the most negative impact on quality. As {max_corr_feature} increases, the quality score tends to decrease.")

    # --- TAB 3: DISTRIBUTION ANALYSIS ---
    with tab3:
        st.header("3. Full Distribution Analysis (Large View)")
        st.markdown("Invalid or missing mechanical properties (≤ 0) are automatically removed to ensure highly accurate normal distribution curves.")
        
        grade_mapping = {
            'A+B+ (Excellent)': 'A+B+數',
            'A-B+ (Good)': 'A-B+數',
            'A-B (Average)': 'A-B數',
            'A-B- (Poor)': 'A-B-數',
            'B+ (Secondary)': 'B+數'
        }
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728'] 
        
        thickness_list = df['厚度歸類'].dropna().unique()
        thickness_list = sorted(thickness_list, key=lambda x: str(x))
        num_thick = len(thickness_list)
        
        for feature in features:
            st.markdown(f"### 📊 Distribution of Parameter: **{feature}**")
            
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
                    ax.set_title(f"Thickness: {thickness}", fontsize=20, fontweight='bold', pad=15)
                    ax.set_xlabel(f"Value of {feature}", fontsize=16)
                    ax.set_ylabel("Number of Coils", fontsize=16) 
                    ax.set_xlim(bin_range) 
                    ax.grid(axis='y', linestyle=':', alpha=0.7)
                    ax.tick_params(axis='both', which='major', labelsize=14)
                    
                    handles, labels = ax.get_legend_handles_labels()
                    unique_labels = list(grade_mapping.keys())
                    unique_handles = [h for h, l in zip(handles, labels) if l in unique_labels]
                    if unique_handles:
                        ax.legend(unique_handles, unique_labels, title="Quality Grade", fontsize=14, title_fontsize=14, loc='upper right')
                else:
                    ax.set_title(f"Thickness: {thickness}\n(Insufficient valid data)", fontsize=18, color='gray')
                    ax.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True) 
            st.markdown("---")
            
            # Save Distribution chart for export
            exported_figures[f'{feature}_Distribution'] = fig

    # --- EXCEL EXPORT SECTION ---
    st.markdown("## 📥 Export Report")
    st.markdown("Download a comprehensive Excel report containing the data tables and all generated charts.")
    
    # Generate Excel in memory
    excel_buffer = io.BytesIO()
    
    # Use xlsxwriter engine to support image insertion
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        # Write DataFrames
        display_df[display_cols].to_excel(writer, sheet_name='Summary_Data', index=False)
        corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
        
        # Access the xlsxwriter workbook and worksheets
        workbook = writer.book
        worksheet_charts = workbook.add_worksheet('Generated_Charts')
        
        # Configure chart sheet
        worksheet_charts.write('A1', 'QC Analysis Charts')
        
        # Insert images
        row_idx = 2
        for chart_name, fig_obj in exported_figures.items():
            # Save figure to a temporary memory buffer
            img_buffer = io.BytesIO()
            fig_obj.savefig(img_buffer, format='png', bbox_inches='tight')
            
            # Add a title for the chart in Excel
            worksheet_charts.write(row_idx, 1, chart_name.replace('_', ' '))
            
            # Insert the image into the worksheet
            worksheet_charts.insert_image(row_idx + 2, 1, '', {'image_data': img_buffer, 'x_scale': 0.6, 'y_scale': 0.6})
            
            # Move row index down to make room for the next chart (approximate spacing)
            # Since charts are large (figsize 16x7), we skip about 45 rows
            row_idx += 50
            
    # Reset buffer pointer to the beginning
    excel_buffer.seek(0)
    
    # Streamlit Download Button
    st.download_button(
        label="📥 Download Full Excel Report (Data & Charts)",
        data=excel_buffer,
        file_name="QC_Analysis_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Please upload your Excel data file from the top menu to begin analysis.")
