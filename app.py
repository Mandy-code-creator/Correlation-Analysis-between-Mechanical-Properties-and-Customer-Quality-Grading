import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import io
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Mechanical Properties QC Dashboard", layout="wide")

st.title("📊 QC Analysis Dashboard - Sturges Binning Applied")
st.markdown("---")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your Excel data (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    exported_figures = {} 

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

    # Calculate Quality Score
    df['Quality_Score'] = (5 * df.get('A+B+數',0) + 4 * df.get('A-B+數',0) + 
                           3 * df.get('A-B數',0) + 2 * df.get('A-B-數',0) + 1 * df.get('B+數',0)) / df['Total_Count']

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Summary & Pie Charts", 
        "2. Correlation Matrix", 
        "3. Distribution (Sturges Rule)",
        "4. Optimal Mechanical Limits"
    ])

    # --- TAB 1 & 2 (Keep as previous version for brevity) ---
    with tab1:
        st.header("1. Summary by Thickness")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Total Coils'] = summary_df[count_cols].sum(axis=1)
        st.dataframe(summary_df, use_container_width=True)

    with tab2:
        st.header("2. Mechanical Correlation")
        corr_matrix = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)

    # --- TAB 3: DISTRIBUTION WITH STURGES RULE ---
    with tab3:
        st.header("3. Distribution Analysis with Sturges Binning")
        st.info("💡 Number of bins is calculated using Sturges Rule: $K = 1 + 3.32 \log_{10}N$")
        
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
                
                # --- CALCULATE STURGES BINS FOR THIS SUBSET ---
                N = df_t['Total_Count'].sum()
                if N > 0:
                    k_sturges = int(1 + 3.322 * math.log10(N))
                    k_sturges = max(k_sturges, 5) # Minimum 5 bins
                else:
                    k_sturges = 10

                for (label, col), color in zip(grade_mapping.items(), colors):
                    temp = df_t[[feature, col]].dropna()
                    temp = temp[temp[col] > 0]
                    if len(temp) > 2:
                        vals, wgts = temp[feature].values, temp[col].values
                        sns.histplot(x=vals, weights=wgts, label=label, color=color, bins=k_sturges, stat='count', alpha=0.25, ax=ax)
                        
                        # Normal Curve Overlay
                        m = np.average(vals, weights=wgts)
                        s = np.sqrt(np.average((vals-m)**2, weights=wgts))
                        if s > 0:
                            x = np.linspace(vals.min(), vals.max(), 100)
                            # Adjusting PDF scale based on bin width
                            bin_width = (vals.max() - vals.min()) / k_sturges
                            ax.plot(x, stats.norm.pdf(x, m, s) * wgts.sum() * bin_width, color=color, lw=3)
                        ax.axvline(m, color=color, ls='--', lw=2)
                
                ax.set_title(f"Thickness: {thick} (Bins: {k_sturges}, N: {int(N)})", fontsize=16, fontweight='bold')
                ax.legend(title="Quality Grade")
            st.pyplot(fig)
            exported_figures[f'{feature}_Sturges'] = fig

    # --- TAB 4: OPTIMAL LIMITS ---
    with tab4:
        st.header("4. Target Optimization (90% Coverage)")
        opt_results = []
        for feat in mech_features:
            all_good_vals = []
            for col in ['A+B+數', 'A-B+數']:
                if col in df.columns:
                    temp = df[[feat, col]].dropna()
                    temp = temp[temp[col] > 0]
                    all_good_vals.extend(np.repeat(temp[feat].values, temp[col].values.astype(int)))
            
            if len(all_good_vals) > 10:
                arr = np.array(all_good_vals)
                l_limit, u_limit = np.percentile(arr, 5), np.percentile(arr, 95)
                opt_results.append({'Parameter': feat, 'Optimal Lower': round(l_limit, 2), 'Optimal Upper': round(u_limit, 2)})
                st.success(f"🎯 **{feat} Target:** {l_limit:.2f} ~ {u_limit:.2f}")

        st.table(pd.DataFrame(opt_results))

    # --- EXPORT ---
    if st.button("📥 Export Full Report"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            workbook = writer.book
            sheet = workbook.add_worksheet('Charts')
            row = 0
            for name, fig in exported_figures.items():
                imgdata = io.BytesIO()
                fig.savefig(imgdata, format='png', bbox_inches='tight')
                sheet.insert_image(row, 0, name, {'image_data': imgdata, 'x_scale': 0.5, 'y_scale': 0.5})
                row += 40
        st.download_button("Download Report", output.getvalue(), "QC_Sturges_Report.xlsx")

else:
    st.info("Please upload your data file.")
