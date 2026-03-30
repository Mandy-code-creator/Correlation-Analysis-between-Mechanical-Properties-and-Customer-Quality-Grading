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

    # --- TAB 3: DISTRIBUTION (PARALLEL VIEW YS-TS / EL-YPE) ---
    with tab3:
        st.header("3. Distribution Analysis (Parallel Clear View)")
        st.markdown("Charts are grouped in pairs for better correlation analysis: YS-TS and EL-YPE.")
        
        grade_mapping = {'A+B+': 'A+B+數', 'A-B+': 'A-B+數', 'A-B': 'A-B數', 'A-B-': 'A-B-數', 'B+': 'B+數'}
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)

        # Helper function to plot a single feature on a specific axis
        def plot_feature_dist(ax, data, feat, thick, group_name):
            N_t = data['Total_Count'].sum()
            k_b = int(1 + 3.322 * math.log10(N_t)) if N_t > 0 else 10
            k_b = max(k_b, 5)
            
            mean_inf = []
            for (label, col_n), color in zip(grade_mapping.items(), colors):
                temp_d = data[[feat, col_n]].dropna()
                temp_d = temp_d[temp_d[col_n] > 0]
                if len(temp_d) > 2:
                    vals_d, wgts_d = temp_d[feat].values, temp_d[col_n].values
                    # Histogram
                    sns.histplot(x=vals_d, weights=wgts_d, label=label, color=color, bins=k_b, 
                                 stat='count', alpha=0.15, ax=ax, edgecolor='none')
                    # Normal Curve (Extended Tails)
                    m_d = np.average(vals_d, weights=wgts_d)
                    s_d = np.sqrt(np.average((vals_d - m_d)**2, weights=wgts_d))
                    if s_d > 0:
                        x_range_d = np.linspace(m_d - 4*s_d, m_d + 4*s_d, 150)
                        bin_w_d = (vals_d.max() - vals_d.min()) / k_b if vals_d.max() != vals_d.min() else 1
                        ax.plot(x_range_d, stats.norm.pdf(x_range_d, m_d, s_d) * wgts_d.sum() * bin_w_d, 
                                color=color, lw=2.5, alpha=0.85)
                    # Mean Line
                    ax.axvline(m_d, color=color, ls='--', lw=2)
                    mean_inf.append({'val': m_d, 'color': color})

            # Anti-overlap label logic
            if mean_inf:
                mean_inf.sort(key=lambda x: x['val'])
                y_max_l = ax.get_ylim()[1]
                for idx_m, info_m in enumerate(mean_inf):
                    y_p = (0.94 if idx_m % 2 == 0 else 0.86) * y_max_l
                    ax.text(info_m['val'], y_p, f"{info_m['val']:.1f}", color=info_m['color'], 
                            fontsize=11, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            ax.set_title(f"Thickness: {thick} | {feat}", fontsize=15, fontweight='bold')
            ax.set_ylabel("Coil Count")
            ax.grid(axis='y', linestyle=':', alpha=0.6)
            
            # Simplified Legend for side-by-side view
            if i % 2 == 1: # Only on the right column
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), title="Grade", 
                          bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
            else:
                ax.legend().set_visible(False)

        # Loop through each thickness group and create parallel layouts
        for thickness in thickness_list:
            df_thickness = df[df['厚度歸類'] == thickness]
            st.markdown(f"## 📏 Analysis for Thickness: **{thickness}**")
            
            # --- ROW 1: YS vs TS ---
            col_ys, col_ts = st.columns(2)
            
            # YS Chart
            if 'YS' in mech_features:
                with col_ys:
                    fig_ys, ax_ys = plt.subplots(figsize=(10, 5))
                    # Pass 0 to indicate left column (no legend)
                    plot_feature_dist(ax_ys, df_thickness, 'YS', thickness, "YS-TS") 
                    st.pyplot(fig_ys)
            
            # TS Chart
            if 'TS' in mech_features:
                with col_ts:
                    fig_ts, ax_ts = plt.subplots(figsize=(10, 5))
                    # Pass 1 to indicate right column (show legend)
                    plot_feature_dist(ax_ts, df_thickness, 'TS', thickness, "YS-TS")
                    # Force legend visible on the right chart
                    handles_ts, labels_ts = ax_ts.get_legend_handles_labels()
                    by_label_ts = dict(zip(labels_ts, handles_ts))
                    ax_ts.legend(by_label_ts.values(), by_label_ts.keys(), title="Grade", 
                               bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
                    st.pyplot(fig_ts)

            # --- ROW 2: EL vs YPE ---
            col_el, col_ype = st.columns(2)
            
            # EL Chart
            if 'EL' in mech_features:
                with col_el:
                    fig_el, ax_el = plt.subplots(figsize=(10, 5))
                    plot_feature_dist(ax_el, df_thickness, 'EL', thickness, "EL-YPE")
                    st.pyplot(fig_el)
            
            # YPE Chart
            if 'YPE' in mech_features:
                with col_ype:
                    fig_ype, ax_ype = plt.subplots(figsize=(10, 5))
                    plot_feature_dist(ax_ype, df_thickness, 'YPE', thickness, "EL-YPE")
                    # Force legend visible on the right chart
                    handles_ype, labels_ype = ax_ype.get_legend_handles_labels()
                    by_label_ype = dict(zip(labels_ype, handles_ype))
                    ax_ype.legend(by_label_ype.values(), by_label_ype.keys(), title="Grade", 
                                bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
                    st.pyplot(fig_ype)
            
            st.markdown("---") # Separator between thicknesses
    # --- TAB 4: SAFE WINDOW OPTIMIZATION ---
    with tab4:
        st.header("4. Safe Operating Window by Thickness (with I-MR & Sigma Control)")
        sigma_factor = st.slider("Select Sigma Factor", 1.5, 3.5, 2.0, 0.5)

        spec_limits = {
            "YS": (405, 500),
            "TS": (415, 550),
            "EL": (25, None),
            "YPE": (4, None)
        }

        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)

        for thick in thickness_list:
            st.subheader(f"Thickness Category: {thick}")
            df_t = df[df['厚度歸類'] == thick]

            # Summary table
            status_list = []
            for feat in mech_features:
                mean_val = df_t[feat].mean(skipna=True)
                std_val = df_t[feat].std(skipna=True)
                safe_val = mean_val - sigma_factor*std_val if pd.notnull(std_val) else np.nan
                low, high = spec_limits.get(feat, (None, None))
                if low is not None and high is not None:
                    ctrl_low = low + 0.05*(high-low)
                    ctrl_high = high - 0.05*(high-low)
                    ctrl_limit = f"{round(ctrl_low,2)}–{round(ctrl_high,2)}"
                elif low is not None:
                    ctrl_limit = f">={low+ (0.05*low):.2f}"
                else:
                    ctrl_limit = "N/A"
                status = "✅ Safe"
                if low is not None and safe_val < low:
                    status = "⚠ Risk (below limit)"
                if high is not None and safe_val > high:
                    status = "⚠ Risk (above limit)"
                status_list.append({
                    "Feature": feat,
                    "Measured Mean": round(mean_val, 2),
                    f"Safe Zone (Mean - {sigma_factor}σ)": round(safe_val, 2),
                    "Spec Limit": f"{low}–{high}" if high else f">={low}",
                    "Proposed Control Limit": ctrl_limit,
                    "Status": status
                })
            status_df = pd.DataFrame(status_list)
            st.dataframe(status_df, use_container_width=True)

            # --- I-MR Chart for each feature ---
            for feat in mech_features:
                st.markdown(f"### I-MR Chart: {feat}")
                vals = df_t[feat].dropna().values
                if len(vals) > 1:
                    # Individuals chart
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                    ax1.plot(vals, marker='o', color='blue')
                    mean_val = np.mean(vals)
                    std_val = np.std(vals, ddof=1)
                    UCL = mean_val + sigma_factor*std_val
                    LCL = mean_val - sigma_factor*std_val
                    ax1.axhline(mean_val, color='green', linestyle='--', label='Mean')
                    ax1.axhline(UCL, color='red', linestyle='--', label=f'UCL ({sigma_factor}σ)')
                    ax1.axhline(LCL, color='red', linestyle='--', label=f'LCL ({sigma_factor}σ)')
                    ax1.set_title(f"Individuals Chart for {feat}")
                    ax1.legend()

                    # Moving Range chart
                    MR = np.abs(np.diff(vals))
                    ax2.plot(MR, marker='o', color='orange')
                    MR_mean = np.mean(MR)
                    ax2.axhline(MR_mean, color='green', linestyle='--', label='MR Mean')
                    ax2.set_title(f"Moving Range Chart for {feat}")
                    ax2.legend()

                    st.pyplot(fig)
