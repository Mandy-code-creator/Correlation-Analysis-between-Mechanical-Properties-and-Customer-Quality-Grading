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
st.markdown("""
This system identifies the **Safe Operating Window** for mechanical properties, 
specifically addressing cases where a single coil produces multiple quality grades.
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
        
        for col in count_cols:
            summary_df[f"% {col}"] = (summary_df[col] / summary_df['Total Coils'] * 100).fillna(0).round(0).astype(int)
            
        display_df = summary_df.copy()
        display_df.rename(columns={'厚度歸類': 'Thickness'}, inplace=True)
        
        display_df.insert(0, 'STT', range(1, len(display_df) + 1))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- TAB 2: CORRELATION ---
    with tab2:
        st.header("2. Mechanical Correlation Index")
        df['Quality_Score'] = (5*df.get('A+B+數', 0) + 4*df.get('A-B+數', 0) + 3*df.get('A-B數', 0) +
                               2*df.get('A-B-數', 0) + 1*df.get('B+數', 0)) / df['Total_Count']
        corr_matrix = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)

    # --- TAB 3: DISTRIBUTION (PARALLEL VIEW) ---
    with tab3:
        st.header("3. Distribution Analysis (Parallel Clear View)")
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
                                 stat='count', alpha=0.15, ax=ax, edgecolor='none')
                    m_d = np.average(vals_d, weights=wgts_d)
                    s_d = np.sqrt(np.average((vals_d - m_d)**2, weights=wgts_d))
                    if s_d > 0:
                        x_range_d = np.linspace(m_d - 4*s_d, m_d + 4*s_d, 150)
                        bin_w_d = (vals_d.max() - vals_d.min()) / k_b if vals_d.max() != vals_d.min() else 1
                        ax.plot(x_range_d, stats.norm.pdf(x_range_d, m_d, s_d) * wgts_d.sum() * bin_w_d, 
                                color=color, lw=2.5, alpha=0.85)
                    ax.axvline(m_d, color=color, ls='--', lw=2)
                    mean_inf.append({'val': m_d, 'color': color})

            if mean_inf:
                mean_inf.sort(key=lambda x: x['val'])
                y_max_l = ax.get_ylim()[1]
                for idx_m, info_m in enumerate(mean_inf):
                    y_p = (0.94 if idx_m % 2 == 0 else 0.86) * y_max_l
                    ax.text(info_m['val'], y_p, f"{info_m['val']:.0f}", color=info_m['color'], 
                            fontsize=10, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            ax.set_title(f"{feat} (Thick: {thick})", fontsize=14, fontweight='bold')
            ax.set_ylabel("Count")
            ax.grid(axis='y', linestyle=':', alpha=0.6)
            
            if is_right_col: 
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), title="Grade", 
                          bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
            else:
                if ax.get_legend(): ax.get_legend().remove()

        for thickness in thickness_list:
            df_thickness = df[df['厚度歸類'] == thickness]
            st.markdown(f"## 📏 Thickness Category: **{thickness}**")
            
            col_ys, col_ts = st.columns(2)
            if 'YS' in mech_features:
                with col_ys:
                    fig_ys, ax_ys = plt.subplots(figsize=(10, 5))
                    plot_feature_dist(ax_ys, df_thickness, 'YS', thickness, is_right_col=False) 
                    st.pyplot(fig_ys)
            if 'TS' in mech_features:
                with col_ts:
                    fig_ts, ax_ts = plt.subplots(figsize=(10, 5))
                    plot_feature_dist(ax_ts, df_thickness, 'TS', thickness, is_right_col=True)
                    st.pyplot(fig_ts)

            col_el, col_ype = st.columns(2)
            if 'EL' in mech_features:
                with col_el:
                    fig_el, ax_el = plt.subplots(figsize=(10, 5))
                    plot_feature_dist(ax_el, df_thickness, 'EL', thickness, is_right_col=False)
                    st.pyplot(fig_el)
            if 'YPE' in mech_features:
                with col_ype:
                    fig_ype, ax_ype = plt.subplots(figsize=(10, 5))
                    plot_feature_dist(ax_ype, df_thickness, 'YPE', thickness, is_right_col=True)
                    st.pyplot(fig_ype)
            st.markdown("---")

    # --- TAB 4: SAFE WINDOW OPTIMIZATION ---
    with tab4:
        st.header("4. Safe Operating Window by Thickness (with I-MR & Segment Quality)")
        sigma_factor = st.radio("Select Sigma Factor", [2.0, 2.5, 3.0], index=0)

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

            status_list = []
            for feat in mech_features:
                vals = df_t[feat].dropna().values
                if len(vals) == 0:
                    continue

                # loại bỏ ngoại lai (±3σ)
                mean_val = np.mean(vals)
                std_val = np.std(vals, ddof=1)
                vals_clean = vals[(vals >= mean_val-3*std_val) & (vals <= mean_val+3*std_val)]

                mean_val = np.mean(vals_clean)
                std_val = np.std(vals_clean, ddof=1)
                safe_val = mean_val - sigma_factor*std_val
                low, high = spec_limits.get(feat, (None, None))

                # Proposed Control Limit (±5% inside spec)
                if low is not None and high is not None:
                    ctrl_low = low + 0.05*(high-low)
                    ctrl_high = high - 0.05*(high-low)
                    ctrl_limit = f"{round(ctrl_low,2)}–{round(ctrl_high,2)}"
                elif low is not None:
                    ctrl_limit = f">={low+ (0.05*low):.2f}"
                else:
                    ctrl_limit = "N/A"

                # I-MR Control Limit
                UCL_I = mean_val + sigma_factor*std_val
                LCL_I = mean_val - sigma_factor*std_val
                ctrl_imr = f"{round(LCL_I,2)}–{round(UCL_I,2)}"

                # CHỈNH SỬA LỖI N/A: Tính Xác Suất Trực Tiếp Trên Dữ Liệu
                target_grade = 'A-B+數'
                temp_opt = df_t[[feat, target_grade, 'Total_Count']].dropna()
                success_prob = None
                
                if len(temp_opt) > 0:
                    # Trích xuất tất cả các cuộn thép có cơ tính rơi vào vùng I-MR an toàn
                    coils_in_zone = temp_opt[(temp_opt[feat] >= LCL_I) & (temp_opt[feat] <= UCL_I)]
                    
                    if len(coils_in_zone) > 0:
                        total_target_yield = coils_in_zone[target_grade].sum()
                        total_all_yield = coils_in_zone['Total_Count'].sum()
                        
                        if total_all_yield > 0:
                            success_prob = (total_target_yield / total_all_yield) * 100

                # Segment Distribution
                seg_A_Bplusplus = df_t['A+B+數'].sum()
                seg_A_Bplus = df_t['A-B+數'].sum()
                seg_A_B = df_t['A-B數'].sum()
                seg_A_Bminus = df_t['A-B-數'].sum()
                seg_Bplus = df_t['B+數'].sum()
                seg_total = seg_A_Bplusplus + seg_A_Bplus + seg_A_B + seg_A_Bminus + seg_Bplus

                if seg_total > 0:
                    seg_dist = (
                        f"A+B+: {seg_A_Bplusplus/seg_total:.0%}, "
                        f"A-B+: {seg_A_Bplus/seg_total:.0%}, "
                        f"A-B: {seg_A_B/seg_total:.0%}, "
                        f"A-B-: {seg_A_Bminus/seg_total:.0%}, "
                        f"B+: {seg_Bplus/seg_total:.0%}"
                    )
                else:
                    seg_dist = "N/A"

                # Expected Quality Level
                if success_prob is not None:
                    if success_prob >= 70 and seg_total > 0 and (seg_A_Bplusplus+seg_A_Bplus)/seg_total >= 0.75:
                        exp_quality = "A-B+"
                    elif success_prob >= 40:
                        exp_quality = "A-B"
                    else:
                        exp_quality = "B or lower"
                else:
                    exp_quality = "Unknown"

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
                    "I-MR Control Limit": ctrl_imr,
                    "Success Probability in Control Zone (%)": round(success_prob, 2) if success_prob is not None else "N/A",
                    "Segment Distribution (A+B+, A-B+, A-B, A-B-, B+)": seg_dist,
                    "Expected Quality Level": exp_quality,
                    "Status": status
                })

            status_df = pd.DataFrame(status_list)
            st.dataframe(status_df, use_container_width=True)

            # --- I-MR Chart ---
            for feat in mech_features:
                vals = df_t[feat].dropna().values
                if len(vals) > 1:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                    mean_val = np.mean(vals)
                    std_val = np.std(vals, ddof=1)
                    UCL_I = mean_val + sigma_factor*std_val
                    LCL_I = mean_val - sigma_factor*std_val

                    ax1.plot(vals, marker='o', color='blue')
                    ax1.axhline(mean_val, color='green', linestyle='--', label='Mean')
                    ax1.axhline(UCL_I, color='red', linestyle='--', label=f'UCL I ({sigma_factor}σ)')
                    ax1.axhline(LCL_I, color='red', linestyle='--', label=f'LCL I ({sigma_factor}σ)')
                    ax1.set_title(f"Individuals Chart for {feat}")
                    ax1.legend()

                    MR = np.abs(np.diff(vals))
                    MR_mean = np.mean(MR)
                    d3, d4 = 0, 3.267 
                    UCL_MR = d4 * MR_mean
                    LCL_MR = d3 * MR_mean
                    ax2.plot(MR, marker='o', color='orange')
                    ax2.axhline(MR_mean, color='green', linestyle='--', label='MR Mean')
                    ax2.axhline(UCL_MR, color='red', linestyle='--', label='UCL MR')
                    ax2.axhline(LCL_MR, color='red', linestyle='--', label='LCL MR')
                    ax2.set_title(f"Moving Range Chart for {feat}")
                    ax2.legend()
                    st.pyplot(fig)

            # --- Success Probability Curves ---
            pairs = [("YS","TS"), ("EL","YPE")]
            target_grade = 'A-B+數'
            for f1, f2 in pairs:
                if f1 in df_t.columns and f2 in df_t.columns:
                    st.markdown(f"### Success Probability Curves: {f1} & {f2}")
                    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
                    for ax, feat in zip(axes, [f1, f2]):
                        temp_opt = df_t[[feat, target_grade, 'Total_Count']].dropna()
                        if len(temp_opt) > 0:
                            temp_opt['bin'] = pd.qcut(temp_opt[feat], q=12, duplicates='drop')
                            bin_res = temp_opt.groupby('bin', observed=True).agg({
                                target_grade: 'sum',
                                'Total_Count': 'sum'
                            })
                            bin_res['Success_Rate'] = (bin_res[target_grade] / bin_res['Total_Count'] * 100).round(2)
                            bin_res['Label'] = bin_res.index.map(lambda x: f"{x.left:.0f}-{x.right:.0f}")
                            x_positions = np.arange(len(bin_res))
                            
                            ax.bar(x_positions, bin_res['Total_Count'], color='lightgray', alpha=0.5, label="Volume")
                            ax2 = ax.twinx()
                            ax2.plot(x_positions, bin_res['Success_Rate'], marker='o', color='green', lw=2, label="Success %")
                            
                            ax.set_xticks(x_positions)
                            ax.set_xticklabels(bin_res['Label'], rotation=45, ha='right')
                            ax.set_xlabel(feat)
                            ax.set_ylabel("Volume", color='gray')
                            ax2.set_ylabel("Success %", color='green')
                            ax2.set_ylim(0, 105)
                    st.pyplot(fig)
                    st.markdown("---")

else:
    st.info("Please upload an Excel file to start analysis.")
