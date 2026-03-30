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

st.title("📊 Mechanical Properties & Quality Yield Optimizer (v2.0)")
st.markdown("""
**Update:** This version includes a **Minimum Threshold Filter** to ensure safe windows align with your technical standards (e.g., YPE > 4%).
""")
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
        "1. Summary", 
        "2. Correlation", 
        "3. Distribution",
        "4. SAFE WINDOW OPTIMIZATION"
    ])

    # (Tabs 1, 2, 3 remain consistent with previous logic for data integrity)
    with tab1:
        st.header("1. Summary by Thickness")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df.rename(columns={'厚度歸類': 'Thickness'}, inplace=True)
        st.dataframe(summary_df, use_container_width=True)

    with tab2:
        st.header("2. Correlation Index")
        df['Quality_Score'] = (5*df.get('A+B+數', 0) + 4*df.get('A-B+數', 0) + 3*df.get('A-B數', 0) + 2*df.get('A-B-數', 0) + 1*df.get('B+數', 0)) / df['Total_Count']
        corr = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        st.dataframe(corr.style.background_gradient(cmap='coolwarm'), use_container_width=True)

    with tab3:
        st.header("3. Distribution (Extended Tails)")
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)
        grade_mapping = {'A+B+': 'A+B+數', 'A-B+': 'A-B+數', 'A-B': 'A-B數', 'A-B-': 'A-B-數', 'B+': 'B+數'}
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']
        
        for feat in mech_features:
            fig, axes = plt.subplots(len(thickness_list), 1, figsize=(16, 6*len(thickness_list)))
            axes = axes if len(thickness_list) > 1 else [axes]
            for i, thick in enumerate(thickness_list):
                ax = axes[i]
                df_t = df[df['厚度歸類'] == thick]
                N = df_t['Total_Count'].sum()
                k = int(1 + 3.322 * math.log10(N)) if N > 0 else 10
                for (label, col), color in zip(grade_mapping.items(), colors):
                    temp = df_t[[feat, col]].dropna()
                    temp = temp[temp[col] > 0]
                    if len(temp) > 2:
                        vals, wgts = temp[feat].values, temp[col].values
                        sns.histplot(x=vals, weights=wgts, label=label, color=color, bins=k, alpha=0.15, ax=ax)
                        m, s = np.average(vals, weights=wgts), np.sqrt(np.average((vals-np.average(vals, weights=wgts))**2, weights=wgts))
                        if s > 0:
                            x = np.linspace(m - 4*s, m + 4*s, 150)
                            ax.plot(x, stats.norm.pdf(x, m, s) * wgts.sum() * ((vals.max()-vals.min())/k), color=color, lw=2.5)
                        ax.axvline(m, color=color, ls='--', lw=2)
                        ax.text(m, ax.get_ylim()[1]*0.9, f"{m:.1f}", color=color, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.6))
                ax.set_title(f"Thickness: {thick} - {feat}")
                ax.legend(loc='upper right')
            st.pyplot(fig)

    # --- TAB 4: FIXED SAFE WINDOW OPTIMIZATION ---
    with tab4:
        st.header("4. Target Optimization with Technical Constraints")
        st.info("💡 Adjust the 'Minimum Requirement' below to align with your factory standards.")

        # User input for minimum requirements (defaulting YPE to 4.0 as per your feedback)
        col_input1, col_input2 = st.columns(2)
        with col_input1:
            ype_min = st.number_input("Minimum YPE Requirement (%)", value=4.0, step=0.1)
        with col_input2:
            target_grade = st.selectbox("Optimize for Grade:", ['A-B+數', 'A+B+數', 'A-B數'])

        for feat in mech_features:
            st.subheader(f"Safe Window Optimization: {feat}")
            
            # Apply Filter: Only consider data meeting the technical standard
            # (Specifically for YPE, we apply the ype_min filter)
            if feat == 'YPE':
                temp_opt = df[df['YPE'] >= ype_min][[feat, target_grade, 'Total_Count']].dropna()
            else:
                temp_opt = df[[feat, target_grade, 'Total_Count']].dropna()

            if not temp_opt.empty:
                # Weighted Success Rate Analysis
                temp_opt['bin'] = pd.qcut(temp_opt[feat], q=10, duplicates='drop')
                bin_res = temp_opt.groupby('bin', observed=True).agg({target_grade: 'sum', 'Total_Count': 'sum'})
                bin_res['Success_Rate'] = (bin_res[target_grade] / bin_res['Total_Count'] * 100).round(2)
                bin_res['Mid'] = bin_res.index.map(lambda x: x.mid)
                
                # Calculation of Safe Window
                avg_success = bin_res['Success_Rate'].mean()
                safe_bins = bin_res[bin_res['Success_Rate'] >= avg_success]
                
                if not safe_bins.empty:
                    low_s, high_s = safe_bins.index[0].left, safe_bins.index[-1].right
                    # Final safeguard: ensure low_s is not below technical requirement
                    if feat == 'YPE': low_s = max(low_s, ype_min)
                    
                    st.success(f"✅ **Validated Safe Window for {feat}: {low_s:.1f} - {high_s:.1f}**")
                    
                    # Plot
                    fig_s, ax_s = plt.subplots(figsize=(12, 4))
                    sns.lineplot(x=bin_res['Mid'].astype(float), y=bin_res['Success_Rate'], marker='o', color='green', lw=3)
                    ax_s.set_ylabel("Success Probability (%)")
                    ax_s.set_xlabel(f"Value of {feat}")
                    ax_s.axvline(low_s, color='red', ls=':', label='Lower Limit')
                    ax_s.legend()
                    st.pyplot(fig_s)
            else:
                st.warning(f"No data available for {feat} that meets the {ype_min}% requirement.")

else:
    st.info("Please upload an Excel file to begin.")
