import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QC Mechanical Properties Optimizer", layout="wide")

st.title("📊 Mechanical Properties & Quality Yield Optimizer")
st.markdown("""
This system identifies the **Safe Operating Window** for mechanical properties, 
specifically addressing cases where a single coil produces multiple quality grades.
""")
st.markdown("---")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your Excel data (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Read data
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip() 

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

    # --- 3. CREATE TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Summary & Yields", 
        "2. Correlation Matrix", 
        "3. Weighted Distribution (Sturges)",
        "4. SAFE WINDOW OPTIMIZATION"
    ])

    # --- TAB 1: SUMMARY ---
    with tab1:
        st.header("1. Quality Summary by Thickness")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Total Coils'] = summary_df[count_cols].sum(axis=1)
        display_df = summary_df.copy()
        display_df.rename(columns={'厚度歸類': 'Thickness'}, inplace=True)
        st.dataframe(display_df, use_container_width=True)

    # --- TAB 2: CORRELATION ---
    with tab2:
        st.header("2. Mechanical Correlation Index")
        df['Quality_Score'] = (5*df.get('A+B+數', 0) + 4*df.get('A-B+數', 0) + 3*df.get('A-B數', 0) +
                               2*df.get('A-B-數', 0) + 1*df.get('B+數', 0)) / df['Total_Count']
        corr_matrix = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)

    # --- TAB 3: DISTRIBUTION ---
    with tab3:
        st.header("3. Distribution Analysis (Clear View)")
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
                N_total = df_t['Total_Count'].sum()
                k_bins = int(1 + 3.322 * math.log10(N_total)) if N_total > 0 else 10
                
                mean_labels = []
                for (label, col), color in zip(grade_mapping.items(), colors):
                    temp = df_t[[feature, col]].dropna()
                    temp = temp[temp[col] > 0]
                    if len(temp) > 2:
                        vals, wgts = temp[feature].values, temp[col].values
                        sns.histplot(x=vals, weights=wgts, label=label, color=color, bins=k_bins, stat='count', alpha=0.15, ax=ax)
                        m = np.average(vals, weights=wgts)
                        s = np.sqrt(np.average((vals-m)**2, weights=wgts)) if len(vals) > 1 else 1
                        if s > 0:
                            x_ext = np.linspace(m - 4*s, m + 4*s, 150)
                            bin_w = (vals.max() - vals.min()) / k_bins if vals.max() != vals.min() else 1
                            ax.plot(x_ext, stats.norm.pdf(x_ext, m, s) * wgts.sum() * bin_w, color=color, lw=2.5)
                        ax.axvline(m, color=color, ls='--', lw=2)
                        mean_labels.append({'val': m, 'color': color})
                if mean_labels:
                    mean_labels.sort(key=lambda x: x['val'])
                    y_max = ax.get_ylim()[1]
                    for idx, info in enumerate(mean_labels):
                        y_pos = (0.94 if idx % 2 == 0 else 0.86) * y_max
                        ax.text(info['val'], y_pos, f"{info['val']:.1f}", color=info['color'], 
                                fontsize=12, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                ax.set_title(f"Thickness: {thick} (N={int(N_total)})", fontsize=16, fontweight='bold')
                ax.legend(title="Quality Grade", bbox_to_anchor=(1.01, 1), loc='upper left')
            st.pyplot(fig)

    # --- TAB 4: SAFE WINDOW OPTIMIZATION ---
    with tab4:
        st.header("4. Safe Operating Window (Measured vs Safe Zone vs Spec Limit)")
        st.markdown("""
        This section compares the **measured mechanical properties** with the 
        **safe zone values (mean - 2σ)** and the **specification limits**.
        """)

        spec_limits = {
            "YS": (405, 500),
            "TS": (415, 550),
            "EL": (25, None),
            "YPE": (4, None)
        }

        status_list = []
        for feat in mech_features:
            mean_val = df[feat].mean(skipna=True)
            std_val = df[feat].std(skipna=True)
            safe_val = mean_val - 2*std_val if pd.notnull(std_val) else np.nan
            low, high = spec_limits.get(feat, (None, None))
            status = "✅ Safe"
            if low is not None and safe_val < low:
                status = "⚠ Risk (below limit)"
            if high is not None and safe_val > high:
                status = "⚠ Risk (above limit)"
            status_list.append({
                "Feature": feat,
                "Measured Mean": round(mean_val, 2),
                "Safe Zone (Mean - 2σ)": round(safe_val, 2),
                "Spec Limit": f"{low}–{high}" if high else f">={low}",
                "Status": status
            })
        status_df = pd.DataFrame(status_list)
        st.dataframe(status_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Success Probability Curve (A-B+ Grade)")
        target_grade = 'A-B+數' 
        for feat in mech_features:
            st.markdown(f"### Optimization Analysis: {feat}")
            temp_opt = df[[feat, target_grade, 'Total_Count']].dropna()
            if len(temp_opt) > 0:
                temp_opt['bin'] = pd.qcut(temp_opt[feat], q=12, duplicates='drop')
                bin_res = temp_opt.groupby('bin', observed=True).agg({
                    target_grade: 'sum',
                    'Total_Count': 'sum'
                })
                bin_res['Success_Rate'] = (bin_res[target_grade] / bin_res['Total_Count'] * 100).round(2)
                bin_res['Mid'] = bin_res.index.map(lambda x: x.mid)
                avg_rate = bin_res['Success_Rate'].mean()
                safe_bins = bin_res[bin_res['Success_Rate
