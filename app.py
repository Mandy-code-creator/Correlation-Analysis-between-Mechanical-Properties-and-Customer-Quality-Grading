import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import io
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QC Mechanical Properties Optimizer", layout="wide")

st.title("📊 Mechanical Properties & Quality Yield Optimizer (Full Version)")
st.markdown("---")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your Excel data (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
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

    # --- 3. TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Summary & Percentages", 
        "2. Correlation Matrix", 
        "3. Distribution Analysis",
        "4. SAFE WINDOW OPTIMIZATION"
    ])

    # --- TAB 1: SUMMARY (RESTORED PERCENTAGES) ---
    with tab1:
        st.header("1. Summary by Thickness")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Total Coils'] = summary_df[count_cols].sum(axis=1)
        
        # Restore Percentage Calculations
        for col in count_cols:
            summary_df[f"% {col}"] = (summary_df[col] / summary_df['Total Coils'] * 100).round(2)
            
        display_df = summary_df.copy()
        display_df.rename(columns={'厚度歸類': 'Thickness'}, inplace=True)
        
        # Reorder columns to show Count next to Percentage for each grade
        ordered_cols = ['Thickness', 'Total Coils']
        for col in count_cols:
            ordered_cols.append(col)
            ordered_cols.append(f"% {col}")
            
        st.subheader("Aggregated Quality Data")
        st.dataframe(display_df[ordered_cols], use_container_width=True)

    # --- TAB 2: CORRELATION ---
    with tab2:
        st.header("2. Correlation Index")
        df['Quality_Score'] = (5*df.get('A+B+數', 0) + 4*df.get('A-B+數', 0) + 3*df.get('A-B數', 0) + 2*df.get('A-B-數', 0) + 1*df.get('B+數', 0)) / df['Total_Count']
        corr = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        st.dataframe(corr.style.background_gradient(cmap='coolwarm'), use_container_width=True)

    # --- TAB 3: DISTRIBUTION ---
    with tab3:
        st.header("3. Distribution (Extended Tails & Sturges Rule)")
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)
        grade_mapping = {'A+B+': 'A+B+數', 'A-B+': 'A-B+數', 'A-B': 'A-B數', 'A-B-': 'A-B-數', 'B+': 'B+數'}
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']
        
        for feat in mech_features:
            st.markdown(f"### 📊 Distribution of **{feat}**")
            fig, axes = plt.subplots(len(thickness_list), 1, figsize=(16, 6*len(thickness_list)))
            axes = axes if len(thickness_list) > 1 else [axes]
            for i, thick in enumerate(thickness_list):
                ax = axes[i]
                df_t = df[df['厚度歸類'] == thick]
                N = df_t['Total_Count'].sum()
                k = int(1 + 3.322 * math.log10(N)) if N > 0 else 10
                
                mean_info = []
                for (label, col), color in zip(grade_mapping.items(), colors):
                    temp = df_t[[feat, col]].dropna()
                    temp = temp[temp[col] > 0]
                    if len(temp) > 2:
                        vals, wgts = temp[feat].values, temp[col].values
                        sns.histplot(x=vals, weights=wgts, label=label, color=color, bins=k, alpha=0.15, ax=ax)
                        m, s = np.average(vals, weights=wgts), np.sqrt(np.average((vals-np.average(vals, weights=wgts))**2, weights=wgts))
                        if s > 0:
                            x = np.linspace(m - 4*s, m + 4*s, 150)
                            bin_w = (vals.max() - vals.min()) / k if vals.max() != vals.min() else 1
                            ax.plot(x, stats.norm.pdf(x, m, s) * wgts.sum() * bin_w, color=color, lw=2.5)
                        ax.axvline(m, color=color, ls='--', lw=2)
                        mean_info.append({'val': m, 'color': color})
                
                if mean_info:
                    mean_info.sort(key=lambda x: x['val'])
                    for idx, info in enumerate(mean_info):
                        y_pos = (0.93 if idx % 2 == 0 else 0.85) * ax.get_ylim()[1]
                        ax.text(info['val'], y_pos, f"{info['val']:.1f}", color=info['color'], fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.6))
                
                ax.set_title(f"Thickness: {thick} - {feat}")
                ax.legend(loc='upper right')
            st.pyplot(fig)

    # --- TAB 4: SAFE WINDOW OPTIMIZATION (WITH MIN THRESHOLD) ---
    with tab4:
        st.header("4. Target Optimization with Technical Constraints")
        
        # User defined constraint for YPE
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            ype_min = st.number_input("Minimum YPE Requirement (%)", value=4.0, step=0.1)
        with col_in2:
            target_grade = st.selectbox("Optimize for Grade:", ['A-B+數', 'A+B+數', 'A-B數'])

        for feat in mech_features:
            st.subheader(f"Safe Window Optimization: {feat}")
            
            # Filter logic for technical standards
            if feat == 'YPE':
                temp_opt = df[df['YPE'] >= ype_min][[feat, target_grade, 'Total_Count']].dropna()
            else:
                temp_opt = df[[feat, target_grade, 'Total_Count']].dropna()

            if not temp_opt.empty:
                temp_opt['bin'] = pd.qcut(temp_opt[feat], q=10, duplicates='drop')
                bin_res = temp_opt.groupby('bin', observed=True).agg({target_grade: 'sum', 'Total_Count': 'sum'})
                bin_res['Success_Rate'] = (bin_res[target_grade] / bin_res['Total_Count'] * 100).round(2)
                bin_res['Mid'] = bin_res.index.map(lambda x: x.mid)
                
                avg_success = bin_res['Success_Rate'].mean()
                safe_bins = bin_res[bin_res['Success_Rate'] >= avg_success]
                
                if not safe_bins.empty:
                    low_s, high_s = safe_bins.index[0].left, safe_bins.index[-1].right
                    if feat == 'YPE': low_s = max(low_s, ype_min)
                    
                    st.success(f"✅ **Validated Safe Window for {feat}: {low_s:.1f} - {high_s:.1f}**")
                    
                    fig_s, ax_s = plt.subplots(figsize=(12, 4))
                    sns.lineplot(x=bin_res['Mid'].astype(float), y=bin_res['Success_Rate'], marker='o', color='green', lw=3)
                    ax_s.set_ylabel("Success Probability (%)")
                    ax_s.axvline(low_s, color='red', ls=':', label='Min Req')
                    ax_s.legend()
                    st.pyplot(fig_s)
            else:
                st.warning(f"No data available for {feat} meeting the threshold.")

else:
    st.info("Please upload an Excel file.")
