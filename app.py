import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QC Mechanical Properties Dashboard", layout="wide")
st.title("📊 Quality Yield & Mechanical Properties Optimization")
st.markdown("---")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    # Store charts for Excel export
    exported_figures = {}

    # --- 2. DATA PREPROCESSING ---
    # Primary columns for quality grades
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    count_cols = [col for col in count_cols if col in df.columns]
    
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Mechanical properties columns
    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
    mech_features = [feat for feat in mech_features if feat in df.columns]
    
    for feat in mech_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
        # Clean data: remove invalid 0 or negative values
        df.loc[df[feat] <= 0, feat] = np.nan

    df['Total_Count'] = df[count_cols].sum(axis=1)
    df = df[df['Total_Count'] > 0].copy()

    # Calculate Overall Quality Score
    df['Quality_Score'] = (5*df.get('A+B+數', 0) + 4*df.get('A-B+數', 0) + 3*df.get('A-B數', 0) +
                           2*df.get('A-B-數', 0) + 1*df.get('B+數', 0)) / df['Total_Count']

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Statistics & Pie Charts",
        "2. Correlation Analysis",
        "3. Distribution & Thickness",
        "4. Yield Rate Optimization (Target Finding)"
    ])

    # --- TAB 1: SUMMARY ---
    with tab1:
        st.header("1. Quality Summary by Thickness")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Total Coils'] = summary_df[count_cols].sum(axis=1)
        
        for col in count_cols:
            summary_df[f'% {col}'] = (summary_df[col]/summary_df['Total Coils']*100).round(2)
            
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.subheader("Data Table (English Headings)")
            display_df = summary_df.copy()
            # Mapping to English for professional view
            en_cols = {'厚度歸類': 'Thickness', 'A+B+數': 'Count A+B+', 'A-B+數': 'Count A-B+', 
                       'A-B數': 'Count A-B', 'A-B-數': 'Count A-B-', 'B+數': 'Count B+'}
            display_df.rename(columns=en_cols, inplace=True)
            st.dataframe(display_df, use_container_width=True)

        with col2:
            st.subheader("Quality Distribution (Pie)")
            plot_df = summary_df.set_index('厚度歸類')[count_cols]
            plot_df.columns = ['A+B+', 'A-B+', 'A-B', 'A-B-', 'B+']
            pie_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']

            n_pies = len(plot_df)
            if n_pies > 0:
                fig1, axes1 = plt.subplots((n_pies+1)//2, min(2, n_pies), figsize=(14, 6 * ((n_pies+1)//2)))
                axes_flat = np.array(axes1).flatten() if n_pies > 1 else [axes1]
                for i, thick in enumerate(plot_df.index):
                    data = plot_df.loc[thick]
                    mask = data > 0
                    axes_flat[i].pie(data[mask], labels=None, autopct=lambda p: f'{p:.1f}%' if p>3 else '', 
                                     startangle=90, colors=[c for c,m in zip(pie_colors, mask) if m],
                                     wedgeprops={'edgecolor':'white','linewidth':2}, textprops={'fontsize':12, 'fontweight':'bold'})
                    axes_flat[i].set_title(f"Thickness: {thick}", fontsize=14, fontweight='bold')
                for j in range(i+1, len(axes_flat)): axes_flat[j].axis('off')
                fig1.legend(plot_df.columns, title="Quality Grade", loc="center right", fontsize=12)
                st.pyplot(fig1)
                exported_figures['Quality_Pie'] = fig1

    # --- TAB 2: CORRELATION ---
    with tab2:
        st.header("2. Mechanical Property Correlation")
        corr_matrix = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        corr_matrix.columns = ['Correlation with Quality']
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)
        st.info("💡 Negative correlation means higher property values may reduce the quality score.")

    # --- TAB 3: DISTRIBUTION BY THICKNESS ---
    with tab3:
        st.header("3. Distribution with Normal Curves")
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)
        grade_labels = ['A+B+', 'A-B+', 'A-B', 'A-B-', 'B+']
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']

        for feat in mech_features:
            with st.expander(f"View {feat} Distribution Charts", expanded=False):
                fig, axes = plt.subplots(len(thickness_list), 1, figsize=(16, 6*len(thickness_list)))
                axes = axes if len(thickness_list) > 1 else [axes]
                for i, thick in enumerate(thickness_list):
                    df_t = df[df['厚度歸類'] == thick]
                    for col, label, color in zip(count_cols, grade_labels, colors):
                        temp = df_t[[feat, col]].dropna()
                        temp = temp[temp[col] > 0]
                        if len(temp) > 2:
                            sns.histplot(temp, x=feat, weights=col, label=label, color=color, 
                                         bins=20, stat='count', alpha=0.3, ax=axes[i], kde=False)
                            # Normal curve overlay
                            m, s = np.average(temp[feat], weights=temp[col]), np.sqrt(np.average((temp[feat]-np.average(temp[feat], weights=temp[col]))**2, weights=temp[col]))
                            if s > 0:
                                x = np.linspace(temp[feat].min(), temp[feat].max(), 100)
                                axes[i].plot(x, stats.norm.pdf(x, m, s) * temp[col].sum() * ((temp[feat].max()-temp[feat].min())/20), color=color, lw=2)
                    axes[i].set_title(f"Thickness: {thick} - {feat}")
                    axes[i].legend()
                st.pyplot(fig)
                exported_figures[f'{feat}_Dist'] = fig

    # --- TAB 4: YIELD OPTIMIZATION (THE "SWEET SPOT") ---
    with tab4:
        st.header("4. Finding the Optimal Property Range")
        st.markdown("This analysis shows the **Yield Rate** of high-quality grades (A+B+ and A-B+) at different property levels.")
        
        target_grade_cols = ['A+B+數', 'A-B+數'] # Define what is "High Quality"
        
        for feat in mech_features:
            st.subheader(f"Optimal Range for {feat}")
            
            # 1. Prepare binned data
            temp_opt = df[[feat] + count_cols].dropna()
            # Dynamic binning
            bins = np.histogram_bin_edges(temp_opt[feat], bins=15)
            temp_opt['bin'] = pd.cut(temp_opt[feat], bins=bins)
            
            # 2. Aggregate counts per bin
            bin_summary = temp_opt.groupby('bin', observed=True)[count_cols].sum()
            bin_summary['Total'] = bin_summary.sum(axis=1)
            bin_summary['Good_Yield_%'] = (bin_summary[target_grade_cols].sum(axis=1) / bin_summary['Total'] * 100).fillna(0)
            bin_summary['Bin_Center'] = bin_summary.index.map(lambda x: x.mid)
            
            # 3. Plot Yield Curve
            fig_yield, ax_yield = plt.subplots(figsize=(14, 6))
            ax2 = ax_yield.twinx()
            
            # Bar chart for volume
            sns.barplot(x=bin_summary['Bin_Center'].astype(str), y=bin_summary['Total'], color='lightgray', ax=ax_yield, alpha=0.5, label='Production Volume')
            # Line chart for Yield %
            sns.lineplot(x=np.arange(len(bin_summary)), y=bin_summary['Good_Yield_%'], marker='o', color='green', linewidth=3, ax=ax2, label='A+B+ & A-B+ Yield %')
            
            ax_yield.set_ylabel("Total Coils (Volume)")
            ax2.set_ylabel("High Quality Yield Rate (%)")
            ax2.set_ylim(0, 105)
            ax_yield.set_xticklabels([f"{b.left:.1f}-{b.right:.1f}" for b in bin_summary.index], rotation=45)
            plt.title(f"Yield Rate Curve: How {feat} affects Quality")
            
            # Identify the "Sweet Spot" (Bins with > 90% Yield)
            sweet_spot = bin_summary[bin_summary['Good_Yield_%'] >= 85]
            if not sweet_spot.empty:
                st.success(f"✅ **Sweet Spot detected for {feat}:** Quality is highest between **{sweet_spot.index[0].left:.1f}** and **{sweet_spot.index[-1].right:.1f}**")
            
            st.pyplot(fig_yield)
            exported_figures[f'{feat}_Yield_Curve'] = fig_yield

    # --- EXPORT REPORT ---
    st.markdown("---")
    if st.button("Generate Full Excel Report"):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Summary_Data', index=False)
            # Add images to a specific sheet
            workbook = writer.book
            img_sheet = workbook.add_worksheet('Charts_Report')
            row = 0
            for name, f in exported_figures.items():
                img_data = io.BytesIO()
                f.savefig(img_data, format='png')
                img_sheet.write(row, 0, name)
                img_sheet.insert_image(row+1, 0, name, {'image_data': img_data, 'x_scale': 0.5, 'y_scale': 0.5})
                row += 35
        st.download_button("📥 Download Full Report", buf.getvalue(), "QC_Full_Report.xlsx", "application/vnd.ms-excel")

else:
    st.info("Please upload an Excel file to begin.")
