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

    exported_figures = {} 

    # --- 2. DATA PREPROCESSING ---
    # Primary columns for quality grades (Raw data columns)
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    count_cols = [col for col in count_cols if col in df.columns]
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Mechanical properties columns
    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
    mech_features = [feat for feat in mech_features if feat in df.columns]
    for feat in mech_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
        # Filter out invalid data (<= 0) to prevent noise
        df.loc[df[feat] <= 0, feat] = np.nan

    df['Total_Count'] = df[count_cols].sum(axis=1)
    df = df[df['Total_Count'] > 0].copy()

    # Calculate Quality Score for correlation analysis
    df['Quality_Score'] = (5 * df.get('A+B+數',0) + 4 * df.get('A-B+數',0) + 
                           3 * df.get('A-B數',0) + 2 * df.get('A-B-數',0) + 1 * df.get('B+數',0)) / df['Total_Count']

    # --- CREATE TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Summary & Pie Charts", 
        "2. Correlation Matrix", 
        "3. Weighted Distribution (Large View)",
        "4. Optimal Mechanical Limits (Target Finding)"
    ])

    # --- TAB 1: SUMMARY ---
    with tab1:
        st.header("1. Quality Summary by Thickness")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Total Coils'] = summary_df[count_cols].sum(axis=1)

        for col in count_cols:
            summary_df[f'% {col}'] = (summary_df[col] / summary_df['Total Coils'] * 100).round(2)

        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.subheader("Aggregated Summary Table")
            display_df = summary_df.copy()
            display_df.rename(columns={'厚度歸類': 'Thickness'}, inplace=True)
            st.dataframe(display_df, use_container_width=True)

        with col2:
            st.subheader("Distribution Pie Charts")
            plot_df = summary_df.set_index('厚度歸類')[count_cols]
            plot_df.columns = ['A+B+', 'A-B+', 'A-B', 'A-B-', 'B+']
            pie_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']

            thicknesses = plot_df.index
            n_pies = len(thicknesses)
            if n_pies > 0:
                fig1, axes1 = plt.subplots((n_pies+1)//2, min(2, n_pies), figsize=(14, 6 * ((n_pies+1)//2)))
                axes1_flat = np.array(axes1).flatten() if n_pies > 1 else [axes1]
                for i, thick in enumerate(thicknesses):
                    ax = axes1_flat[i]
                    data = plot_df.loc[thick]
                    mask = data > 0
                    ax.pie(data[mask], autopct=lambda p: f'{p:.1f}%' if p>3 else '', startangle=90, colors=[c for c,m in zip(pie_colors, mask) if m], wedgeprops={'edgecolor':'white','linewidth':2}, textprops={'fontsize':12,'fontweight':'bold'})
                    ax.set_title(f"Thickness: {thick}", fontsize=16, fontweight='bold')
                for j in range(i+1, len(axes1_flat)): ax.axis('off')
                fig1.legend(plot_df.columns, title="Grade", loc="center right", fontsize=12)
                st.pyplot(fig1)
                exported_figures['Quality_Pie'] = fig1

    # --- TAB 2: CORRELATION ---
    with tab2:
        st.header("2. Mechanical Properties Correlation")
        corr_matrix = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        corr_matrix.columns = ['Correlation Index']
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=0), use_container_width=True)
        st.info("💡 Values closer to -1 suggest that higher property values might lead to lower quality grades.")

    # --- TAB 3: DISTRIBUTION (LARGE VIEW) ---
    with tab3:
        st.header("3. Weighted Distribution with Normal Curves")
        grade_mapping = {'A+B+': 'A+B+數', 'A-B+': 'A-B+數', 'A-B': 'A-B數', 'A-B-': 'A-B-數', 'B+': 'B+數'}
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)

        for feature in mech_features:
            st.markdown(f"### 📊 Distribution of **{feature}**")
            fig, axes = plt.subplots(len(thickness_list), 1, figsize=(16, 7 * len(thickness_list)))
            axes = axes if len(thickness_list) > 1 else [axes]

            for i, thick in enumerate(thickness_list):
                ax = axes[i]
                df_t = df[df['厚度歸類'] == thick]
                for (label, col), color in zip(grade_mapping.items(), colors):
                    temp = df_t[[feature, col]].dropna()
                    temp = temp[temp[col] > 0]
                    if len(temp) > 2:
                        vals, wgts = temp[feature].values, temp[col].values
                        sns.histplot(x=vals, weights=wgts, label=label, color=color, bins=20, stat='count', alpha=0.25, ax=ax)
                        m, s = np.average(vals, weights=wgts), np.sqrt(np.average((vals-np.average(vals, weights=wgts))**2, weights=wgts))
                        if s > 0:
                            x = np.linspace(vals.min(), vals.max(), 100)
                            ax.plot(x, stats.norm.pdf(x, m, s) * wgts.sum() * ((vals.max()-vals.min())/20), color=color, lw=3)
                        ax.axvline(m, color=color, ls='--', lw=2)
                ax.set_title(f"Thickness: {thick}", fontsize=18, fontweight='bold')
                ax.set_ylabel("Coil Count")
                ax.legend(title="Quality Grade")
            st.pyplot(fig)
            exported_figures[f'{feature}_Chart'] = fig

    # --- TAB 4: OPTIMAL LIMITS (THE CORE LOGIC) ---
    with tab4:
        st.header("4. Target Optimization for High-Quality Yield")
        st.markdown("This analysis identifies the **Optimal Range** where the probability of achieving A+B+ or A-B+ grades is maximized.")

        opt_results = []
        for feat in mech_features:
            st.subheader(f"Optimal Range Analysis: {feat}")
            
            # Weighted expansion for percentile calculation
            all_good_vals = []
            for col in ['A+B+數', 'A-B+數']:
                if col in df.columns:
                    temp = df[[feat, col]].dropna()
                    temp = temp[temp[col] > 0]
                    all_good_vals.extend(np.repeat(temp[feat].values, temp[col].values.astype(int)))
            
            if len(all_good_vals) > 10:
                arr = np.array(all_good_vals)
                l_limit, u_limit = np.percentile(arr, 5), np.percentile(arr, 95)
                avg, std = np.mean(arr), np.std(arr)
                
                st.success(f"🎯 **Target Range for {feat}: {l_limit:.2f} ~ {u_limit:.2f}** (Based on 90% of High-Quality Coils)")
                
                opt_results.append({
                    'Parameter': feat, 'Optimal Lower': round(l_limit, 2), 
                    'Optimal Upper': round(u_limit, 2), 'Target Mean': round(avg, 2)
                })
                
                # Visualizing the Yield Curve
                fig_opt, ax_opt = plt.subplots(figsize=(14, 5))
                sns.kdeplot(arr, fill=True, color='green', ax=ax_opt, label='Good Quality Distribution')
                ax_opt.axvline(l_limit, color='red', ls='--', label='Lower Limit')
                ax_opt.axvline(u_limit, color='red', ls='--', label='Upper Limit')
                ax_opt.set_title(f"Optimal Operating Window for {feat}")
                ax_opt.legend()
                st.pyplot(fig_opt)
            else:
                st.warning(f"Not enough high-quality data to calculate limits for {feat}.")

        st.table(pd.DataFrame(opt_results))

    # --- EXPORT REPORT ---
    st.markdown("---")
    if st.button("📥 Generate Excel Report with Charts"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            if opt_results: pd.DataFrame(opt_results).to_excel(writer, sheet_name='Optimal_Limits', index=False)
            
            workbook = writer.book
            sheet = workbook.add_worksheet('Analysis_Charts')
            row = 0
            for name, fig in exported_figures.items():
                imgdata = io.BytesIO()
                fig.savefig(imgdata, format='png')
                sheet.write(row, 0, name)
                sheet.insert_image(row+1, 0, name, {'image_data': imgdata, 'x_scale': 0.5, 'y_scale': 0.5})
                row += 40
        st.download_button("Download Full Report", output.getvalue(), "QC_Report.xlsx")

else:
    st.info("Please upload your data file to begin.")
