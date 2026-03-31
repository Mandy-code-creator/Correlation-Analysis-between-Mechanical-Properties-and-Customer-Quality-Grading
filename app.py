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
    tab1, tab2, tab3 = st.tabs([
        "1. Summary & Yields", 
        "2. Distribution Analysis (Parallel View)",
        "3. SAFE WINDOW OPTIMIZATION & EXPORT"
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
        
        cols_to_int = count_cols + ['Total Coils']
        for c in cols_to_int:
            if c in display_df.columns:
                display_df[c] = display_df[c].astype(int)
                
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- TAB 2: DISTRIBUTION (PARALLEL VIEW) ---
    with tab2:
        st.header("2. Distribution Analysis (Parallel Clear View)")
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

            # --- SỬA LỖI GHI ĐÈ NHÃN CHUYÊN SÂU ---
            if mean_inf:
                mean_inf.sort(key=lambda x: x['val'])
                y_max_l = ax.get_ylim()[1]
                for idx_m, info_m in enumerate(mean_inf):
                    # Trải đều độ cao theo 4 cấp bậc (95%, 87%, 79%, 71%) thay vì chỉ 2 cấp
                    y_p = y_max_l * (0.95 - (idx_m % 4) * 0.08)
                    
                    # Thêm boxstyle để đóng khung con số rõ ràng
                    ax.text(info_m['val'], y_p, f"{info_m['val']:.0f}", color=info_m['color'], 
                            fontsize=10, fontweight='bold', ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.85, edgecolor=info_m['color'], lw=1.5, boxstyle='round,pad=0.3'))

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
            st.markdown(f"## 📏 Analysis for Thickness: **{thickness}**")
            
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

    # --- TAB 3: SAFE WINDOW OPTIMIZATION & EXPORT ---
    with tab3:
        st.header("3. Safe Operating Window by Thickness (with I-MR & Export)")
        sigma_factor = st.radio("Select Sigma Factor", [2.0, 2.5, 3.0], index=0)

        spec_limits = {
            "YS": (405, 500), "TS": (415, 550), "EL": (25, None), "YPE": (4, None)
        }

        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)
        all_export_data = []

        for thick in thickness_list:
            st.subheader(f"Thickness Category: {thick}")
            df_t = df[df['厚度歸類'] == thick]

            status_list = []
            for feat in mech_features:
                vals = df_t[feat].dropna().values
                if len(vals) == 0: continue

                mean_val = np.mean(vals)
                std_val = np.std(vals, ddof=1)
                vals_clean = vals[(vals >= mean_val-3*std_val) & (vals <= mean_val+3*std_val)]

                mean_val = np.mean(vals_clean)
                std_val = np.std(vals_clean, ddof=1)
                safe_val = mean_val - sigma_factor*std_val
                low, high = spec_limits.get(feat, (None, None))

                if low is not None and high is not None:
                    spec_str = f"{int(low)}–{int(high)}"
                    ctrl_low = int(round(low + 0.05*(high-low)))
                    ctrl_high = int(round(high - 0.05*(high-low)))
                    ctrl_limit = f"{ctrl_low}–{ctrl_high}"
                elif low is not None:
                    spec_str = f">={int(low)}"
                    ctrl_limit = f">={int(round(low + (0.05*low)))}"
                else:
                    spec_str = "N/A"
                    ctrl_limit = "N/A"

                UCL_I = mean_val + sigma_factor*std_val
                LCL_I = mean_val - sigma_factor*std_val
                ctrl_imr = f"{int(round(LCL_I))}–{int(round(UCL_I))}"

                # Segment Distribution
                seg_A_Bplusplus = df_t['A+B+數'].sum()
                seg_A_Bplus = df_t['A-B+數'].sum()
                seg_A_B = df_t['A-B數'].sum()
                seg_A_Bminus = df_t['A-B-數'].sum()
                seg_Bplus = df_t['B+數'].sum()
                seg_total = seg_A_Bplusplus + seg_A_Bplus + seg_A_B + seg_A_Bminus + seg_Bplus

                if seg_total > 0:
                    seg_dist = (
                        f"A+B+: {int(round(seg_A_Bplusplus/seg_total*100))}%, "
                        f"A-B+: {int(round(seg_A_Bplus/seg_total*100))}%, "
                        f"A-B: {int(round(seg_A_B/seg_total*100))}%, "
                        f"A-B-: {int(round(seg_A_Bminus/seg_total*100))}%, "
                        f"B+: {int(round(seg_Bplus/seg_total*100))}%"
                    )
                else:
                    seg_dist = "N/A"

                target_grade = 'A-B+數'
                temp_opt = df_t[[feat, target_grade, 'Total_Count']].dropna()
                success_prob = None
                if len(temp_opt) > 0:
                    temp_opt['bin'] = pd.qcut(temp_opt[feat], q=12, duplicates='drop')
                    bin_res = temp_opt.groupby('bin', observed=True).agg({
                        target_grade: 'sum',
                        'Total_Count': 'sum'
                    })
                    bin_res['Success_Rate'] = (bin_res[target_grade] / bin_res['Total_Count'] * 100).fillna(0).round(0).astype(int)
                    bin_res['Mid'] = bin_res.index.map(lambda x: x.mid).astype(float)
                    bins_in_ctrl = bin_res[(bin_res['Mid'] >= LCL_I) & (bin_res['Mid'] <= UCL_I)]
                    if not bins_in_ctrl.empty:
                        success_prob = bins_in_ctrl['Success_Rate'].mean()

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

                row_data = {
                    "Thickness": thick,
                    "Feature": feat,
                    "Measured Mean": int(round(mean_val)),
                    f"Safe Zone (Mean - {sigma_factor}σ)": int(round(safe_val)),
                    "Spec Limit": spec_str,
                    "Proposed Control Limit": ctrl_limit,
                    "I-MR Control Limit": ctrl_imr,
                    "Segment Distribution": seg_dist,
                    "Expected Quality Level": exp_quality,
                    "Status": status
                }
                
                status_list.append(row_data)
                all_export_data.append(row_data)

            display_df_3 = pd.DataFrame(status_list)
            if not display_df_3.empty:
                display_df_3 = display_df_3.drop(columns=['Thickness'])
            st.dataframe(display_df_3, use_container_width=True, hide_index=True)

            # --- I-MR Chart ---
            for feat in mech_features:
                st.markdown(f"### I-MR Chart: {feat}")
                v = df_t[feat].dropna().values
                if len(v) > 1:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                    m_v, s_v = np.mean(v), np.std(v, ddof=1)
                    U, L = m_v + sigma_factor*s_v, m_v - sigma_factor*s_v
                    ax1.plot(v, marker='o', color='blue')
                    ax1.axhline(m_v, color='green', ls='--', label='Mean')
                    ax1.axhline(U, color='red', ls='--', label='UCL')
                    ax1.axhline(L, color='red', ls='--', label='LCL')
                    ax1.set_title(f"Individuals Chart for {feat}")
                    ax1.legend()
                    MR = np.abs(np.diff(v))
                    ax2.plot(MR, marker='o', color='orange')
                    ax2.axhline(np.mean(MR), color='green', ls='--')
                    ax2.set_title("Moving Range Chart")
                    st.pyplot(fig)

            # --- Success Probability Curves ---
            target_grade = 'A-B+數'
            pairs = [("YS","TS"), ("EL","YPE")]
            for f1, f2 in pairs:
                if f1 in df_t.columns and f2 in df_t.columns:
                    st.markdown(f"### Success Probability Curves: {f1} & {f2}")
                    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
                    for ax, f in zip(axes, [f1, f2]):
                        temp = df_t[[f, target_grade, 'Total_Count']].dropna()
                        if len(temp) > 0:
                            temp['bin'] = pd.qcut(temp[f], q=12, duplicates='drop')
                            bin_r = temp.groupby('bin', observed=True).agg({target_grade:'sum', 'Total_Count':'sum'})
                            bin_r['SR'] = (bin_r[target_grade]/bin_r['Total_Count']*100).fillna(0).round(0).astype(int)
                            bin_r['L'] = bin_r.index.map(lambda x: f"{x.left:.0f}-{x.right:.0f}")
                            x_pos = np.arange(len(bin_r))
                            ax.bar(x_pos, bin_r['Total_Count'], color='lightgray', alpha=0.5)
                            ax2 = ax.twinx()
                            ax2.plot(x_pos, bin_r['SR'], marker='o', color='green', lw=2)
                            ax.set_xticks(x_pos)
                            ax.set_xticklabels(bin_r['L'], rotation=45, ha='right')
                            ax.set_xlabel(f)
                            ax2.set_ylim(0, 105)
                    st.pyplot(fig)
                    st.markdown("---")

        # --- EXPORT FINAL ---
        if all_export_data:
            st.markdown("### 📥 Export Final Proposed Control Limits")
            towrite = io.BytesIO()
            pd.DataFrame(all_export_data).to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button(label="📥 Tải xuống File Tổng Hợp (Excel)", data=towrite, 
                               file_name="QC_Proposed_Control_Limits.xlsx")

else:
    st.info("Please upload an Excel file.")
