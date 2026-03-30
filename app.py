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
    # Primary columns for quality grades
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    count_cols = [col for col in count_cols if col in df.columns]
    
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Mechanical properties
    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
    mech_features = [feat for feat in mech_features if feat in df.columns]
    
    for feat in mech_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
        # Clean data: remove 0 or negative values to avoid noise
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
        # Quality score for correlation: A+B+ is 5, B+ is 1
        df['Quality_Score'] = (5*df.get('A+B+數', 0) + 4*df.get('A-B+數', 0) + 3*df.get('A-B數', 0) +
                               2*df.get('A-B-數', 0) + 1*df.get('B+數', 0)) / df['Total_Count']
        
        corr_matrix = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)

    # --- TAB 3: DISTRIBUTION (EXTENDED TAILS & NO OVERLAP) ---
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
                        
                        m, s = np.average(vals, weights=wgts), np.sqrt(np.average((vals-m)**2, weights=wgts)) if 'm' in locals() else 1
                        m = np.average(vals, weights=wgts) # Re-calculate cleanly
                        
                        if s > 0:
                            x_ext = np.linspace(m - 4*s, m + 4*s, 150)
                            bin_w = (vals.max() - vals.min()) / k_bins if vals.max() != vals.min() else 1
                            ax.plot(x_ext, stats.norm.pdf(x_ext, m, s) * wgts.sum() * bin_w, color=color, lw=2.5)
                        
                        ax.axvline(m, color=color, ls='--', lw=2)
                        mean_labels.append({'val': m, 'color': color})

                # Prevent label overlap
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

    # --- TAB 4: SAFE WINDOW OPTIMIZATION (SUCCESS PROBABILITY) ---
    with tab4:
        st.header("4. Safe Operating Window for A-B+ Grade")
        st.markdown("""
        Finding the range where the probability of producing **A-B+ or better** is maximized, 
        accounting for coils that split into multiple grades.
        """)

        target_grade = 'A-B+數' 
        
        for feat in mech_features:
            st.subheader(f"Optimization Analysis: {feat}")
            
            # Weighted binning analysis
            temp_opt = df[[feat, target_grade, 'Total_Count']].dropna()
            temp_opt['bin'] = pd.qcut(temp_opt[feat], q=12, duplicates='drop')
            
            bin_res = temp_opt.groupby('bin', observed=True).agg({
                target_grade: 'sum',
                'Total_Count': 'sum'
            })
            bin_res['Success_Rate'] = (bin_res[target_grade] / bin_res['Total_Count'] * 100).round(2)
            bin_res['Mid'] = bin_res.index.map(lambda x: x.mid)
            
            # Find Safe Range (where rate > average success rate)
            avg_rate = bin_res['Success_Rate'].mean()
            safe_bins = bin_res[bin_res['Success_Rate'] > avg_rate]
            
            if not safe_bins.empty:
                low_s, high_s = safe_bins.index[0].left, safe_bins.index[-1].right
                st.success(f"✅ **Safe Operating Window for {feat}: {low_s:.1f} - {high_s:.1f}**")
                st.info(f"Average probability of A-B+ in this window: {safe_bins['Success_Rate'].mean():.1f}%")
            
            # Plot Success Probability Curve
            fig_s, ax_s = plt.subplots(figsize=(12, 5))
            ax_vol = ax_s.twinx()
            
            # Production Volume (Bar)
            sns.barplot(x=bin_res['Mid'].astype(float), y=bin_res['Total_Count'], color='lightgray', ax=ax_vol, alpha=0.4)
            # Success Rate (Line)
            sns.lineplot(x=np.arange(len(bin_res)), y=bin_res['Success_Rate'], marker='o', color='green', lw=3, ax=ax_s)
            
            ax_s.set_ylabel("A-B+ Success Probability (%)", color='green', fontweight='bold')
            ax_vol.set_ylabel("Total Production Volume", color='gray')
            ax_s.set_ylim(0, 105)
            ax_s.set_xticks(np.arange(len(bin_res)))
            ax_s.set_xticklabels([f"{b.left:.1f}-{b.right:.1f}" for b in bin_res.index], rotation=45)
            plt.title(f"Success Probability Curve for {feat}")
            st.pyplot(fig_s)
            st.markdown("---")

else:
    st.info("Please upload an Excel file to start optimization.")
