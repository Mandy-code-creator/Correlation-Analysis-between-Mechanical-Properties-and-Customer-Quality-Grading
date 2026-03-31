import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import math
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QC Mechanical Properties Optimizer", layout="wide")

st.title("📊 Mechanical Properties & Quality Yield Optimizer")
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

    # --- 3. CREATE TABS ---
    tab1, tab2, tab3 = st.tabs([
        "1. Summary & Yields", 
        "2. Distribution Analysis (Parallel View)",
        "3. PRODUCTION CONTROL LIMITS (EXECUTIVE VIEW)"
    ])

    # --- TAB 1: SUMMARY ---
    with tab1:
        st.header("1. Quality Summary by Thickness")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Total Coils'] = summary_df[count_cols].sum(axis=1)
        for col in count_cols:
            summary_df[f"% {col}"] = (summary_df[col] / summary_df['Total Coils'] * 100).fillna(0).round(2)
        display_df = summary_df.copy()
        display_df.rename(columns={'厚度歸類': 'Thickness'}, inplace=True)
        display_df.insert(0, 'No.', range(1, len(display_df) + 1))
        cols_to_int = count_cols + ['Total Coils']
        for c in cols_to_int:
            if c in display_df.columns:
                display_df[c] = display_df[c].astype(int)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- TAB 2: DISTRIBUTION ---
    with tab2:
        st.header("2. Distribution Analysis")
        grade_mapping = {'A+B+': 'A+B+數', 'A-B+': 'A-B+數', 'A-B': 'A-B數', 'A-B-': 'A-B-數', 'B+': 'B+數'}
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728'] 
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)

        def plot_feature_dist(ax, data, feat, thick, is_right_col=False):
            N_t = data['Total_Count'].sum()
            k_b = int(1 + 3.322 * math.log10(N_t)) if N_t > 0 else 10
            k_b = max(k_b, 5)
            mean_inf = []
            for (label, col_n), color in zip(grade_mapping.items(), colors):
                temp_d = data[[feat, col_n]].dropna()
                temp_d = temp_d[temp_d[col_n] > 0]
                if len(temp_d) > 2:
                    vals_d, wgts_d = temp_d[feat].values, temp_d[col_n].values
                    sns.histplot(x=vals_d, weights=wgts_d, label=label, color=color, bins=k_b, 
                                 stat='count', alpha=0.45, ax=ax, edgecolor='white', linewidth=0.5)
                    m_d = np.average(vals_d, weights=wgts_d)
                    s_d = np.sqrt(np.average((vals_d - m_d)**2, weights=wgts_d))
                    if s_d > 0:
                        x_range_d = np.linspace(m_d - 4*s_d, m_d + 4*s_d, 150)
                        bin_w_d = (vals_d.max() - vals_d.min()) / k_b if vals_d.max() != vals_d.min() else 1
                        ax.plot(x_range_d, stats.norm.pdf(x_range_d, m_d, s_d) * wgts_d.sum() * bin_w_d, color=color, lw=2.5)
                    ax.axvline(m_d, color=color, ls='--', lw=2)
                    mean_inf.append({'val': m_d, 'color': color})
            if mean_inf:
                mean_inf.sort(key=lambda x: x['val'])
                y_max_l = ax.get_ylim()[1]
                for idx_m, info_m in enumerate(mean_inf):
                    y_p = y_max_l * (0.95 - (idx_m % 4) * 0.08)
                    ax.text(info_m['val'], y_p, f"{info_m['val']:.0f}", color=info_m['color'], fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor=info_m['color'], boxstyle='round'))
            ax.set_title(f"{feat} (Thick: {thick})")
            if is_right_col: ax.legend(title="Grade", bbox_to_anchor=(1.05, 1), loc='upper left')

        for thickness in thickness_list:
            df_thickness = df[df['厚度歸類'] == thickness]
            st.markdown(f"## 📏 Thickness: {thickness}")
            col1, col2 = st.columns(2)
            if 'YS' in mech_features:
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    plot_feature_dist(ax, df_thickness, 'YS', thickness, False)
                    st.pyplot(fig)
            if 'TS' in mech_features:
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    plot_feature_dist(ax, df_thickness, 'TS', thickness, True)
                    st.pyplot(fig)

    # --- TAB 3: EXECUTIVE OPTIMIZATION ---
    with tab3:
        st.header("3. Production Control Limits & Goals")
        sigma_choice = st.radio("Select Sigma Factor for Mill Range", [2.0, 2.5, 3.0], index=0)

        spec_limits = {"YS": (405, 500), "TS": (415, 550), "EL": (25, None), "YPE": (4, None)}
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)
        all_export_data = []

        for thick in thickness_list:
            st.subheader(f"Thickness Category: {thick}")
            df_t = df[df['厚度歸類'] == thick]
            status_list = []
            good_grades = [c for c in ['A+B+數', 'A-B+數', 'A-B數'] if c in df_t.columns]

            for feat in mech_features:
                temp_calc = df_t[[feat] + good_grades].dropna(subset=[feat])
                if len(temp_calc) == 0: continue
                temp_calc['Good_Count'] = temp_calc[good_grades].sum(axis=1)
                temp_calc_good = temp_calc[temp_calc['Good_Count'] > 0]

                low, high = spec_limits.get(feat, (None, None))
                customer_spec = f"{int(low)}–{int(high)}" if low and high else (f">={int(low)}" if low else "N/A")

                if not temp_calc_good.empty:
                    vals_good = temp_calc_good[feat].values
                    wgts_good = temp_calc_good['Good_Count'].values
                    m_v = np.average(vals_good, weights=wgts_good)
                    s_v = np.sqrt(np.average((vals_good - m_v)**2, weights=wgts_good))
                    
                    # Data-Driven Release (3-sigma)
                    rel_l = max(m_v - 3*s_v, low) if low else m_v - 3*s_v
                    rel_h = min(m_v + 3*s_v, high) if high else m_v + 3*s_v
                    release_range = f"{int(round(rel_l))}–{int(round(rel_h))}" if high else f">={int(round(rel_l))}"
                    
                    # Target Goal
                    target_goal = int(round(m_v))
                    
                    # Tolerance Value (Sigma Factor * Sigma)
                    tolerance_val = int(round(sigma_choice * s_v))
                    
                    # Mill Range
                    mill_l = max(m_v - sigma_choice*s_v, rel_l)
                    mill_h = min(m_v + sigma_choice*s_v, rel_h)
                    mill_range = f"{int(round(mill_l))}–{int(round(mill_h))}" if high else f">={int(round(mill_l))}"
                else:
                    target_goal, release_range, mill_range, tolerance_val = "N/A", "N/A", "N/A", "N/A"

                seg_total = df_t[count_cols].sum().sum()
                seg_dist = "N/A"
                if seg_total > 0:
                    dist_parts = [f"{k.replace('數','')}: {int(round(df_t[k].sum()/seg_total*100))}%" for k in count_cols]
                    seg_dist = ", ".join(dist_parts)

                # --- SẮP XẾP THỨ TỰ BẢNG LOGIC ---
                status_list.append({
                    "Feature": feat,
                    "Customer Spec": customer_spec,
                    "Release Range": release_range,
                    "Target Goal": target_goal,
                    f"Tolerance (±{sigma_choice}σ)": tolerance_val, # THỂ HIỆN GIÁ TRỊ SIGMA
                    "Mill Range (Proposed)": mill_range,
                    "Segment Distribution": seg_dist
                })

            st.dataframe(pd.DataFrame(status_list), use_container_width=True, hide_index=True)
            st.markdown("---")
            
        # --- EXPORT ---
        if status_list:
            towrite = io.BytesIO()
            pd.DataFrame(all_export_data).to_excel(towrite, index=False)
            st.download_button("📥 Download Report", data=towrite.getvalue(), file_name="QC_Report.xlsx")

else:
    st.info("Please upload an Excel file.")
