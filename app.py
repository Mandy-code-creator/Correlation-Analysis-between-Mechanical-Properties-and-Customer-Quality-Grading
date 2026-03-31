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

    # --- TAB 1: SUMMARY (Keep 2 decimals for %) ---
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
                                 stat='count', alpha=0.45, ax=ax, edgecolor='white', linewidth=0.5)
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
                    y_p = y_max_l * (0.95 - (idx_m % 4) * 0.08)
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
            st.markdown("---")

    # --- TAB 3: OPTIMIZATION (Data-Driven Executive View) ---
    with tab3:
        st.header("3. Production Control Limits & Goals (A-B & Above Focused)")
        sigma_factor = st.radio("Select Sigma Factor for Mill Safety Zone", [2.0, 2.5, 3.0], index=0)

        spec_limits = {
            "YS": (405, 500), "TS": (415, 550), "EL": (25, None), "YPE": (4, None)
        }

        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)
        all_export_data = []
        plot_data_dict = {}

        for thick in thickness_list:
            st.subheader(f"Thickness Category: {thick}")
            df_t = df[df['厚度歸類'] == thick]
            plot_data_dict[thick] = {}
            status_list = []
            
            good_grades = [c for c in ['A+B+數', 'A-B+數', 'A-B數'] if c in df_t.columns]

            for feat in mech_features:
                temp_calc = df_t[[feat] + good_grades].dropna(subset=[feat])
                if len(temp_calc) == 0: continue

                temp_calc['Good_Count'] = temp_calc[good_grades].sum(axis=1)
                temp_calc_good = temp_calc[temp_calc['Good_Count'] > 0]

                low, high = spec_limits.get(feat, (None, None))
                spec_str = f"{int(low)}–{int(high)}" if low and high else (f">={int(low)}" if low else "N/A")

                # --- CALCULATE DATA-DRIVEN LIMITS ---
                if not temp_calc_good.empty:
                    vals_good = temp_calc_good[feat].values
                    wgts_good = temp_calc_good['Good_Count'].values
                    
                    # 1. Target Goal (Mean)
                    mean_val = np.average(vals_good, weights=wgts_good)
                    std_val = np.sqrt(np.average((vals_good - mean_val)**2, weights=wgts_good))
                    
                    # Lọc nhiễu Outliers (Giữ lại phổ dữ liệu thuần khiết)
                    mask = (vals_good >= mean_val - 3*std_val) & (vals_good <= mean_val + 3*std_val)
                    if mask.sum() > 0:
                        vals_good = vals_good[mask]
                        wgts_good = wgts_good[mask]
                        mean_val = np.average(vals_good, weights=wgts_good)
                        std_val = np.sqrt(np.average((vals_good - mean_val)**2, weights=wgts_good))
                    
                    plot_data_dict[thick][feat] = vals_good
                    target_goal = int(round(mean_val))
                    
                    # 2. Data-Driven Release Range (3 Sigma: 99.73% natural spread of GOOD coils)
                    rel_low_raw = mean_val - 3 * std_val
                    rel_high_raw = mean_val + 3 * std_val
                    
                    # Đảm bảo Release Range thực tế không được phép lấn ra ngoài Customer Spec
                    rel_low = max(rel_low_raw, low) if low is not None else rel_low_raw
                    rel_high = min(rel_high_raw, high) if high is not None else rel_high_raw
                    
                    release_range = f"{int(round(rel_low))}–{int(round(rel_high))}" if high is not None else f">={int(round(rel_low))}"
                    
                    # 3. Mill Range (Tighter operational control based on selected sigma_factor)
                    mill_low_raw = mean_val - sigma_factor * std_val
                    mill_high_raw = mean_val + sigma_factor * std_val
                    
                    mill_low = max(mill_low_raw, rel_low)
                    mill_high = min(mill_high_raw, rel_high)
                    
                    mill_range = f"{int(round(mill_low))}–{int(round(mill_high))}" if high is not None else f">={int(round(mill_low))}"
                else:
                    target_goal = "N/A"
                    release_range = "N/A"
                    mill_range = "N/A"
                    mean_val = 0
                    std_val = 0

                # --- SEGMENT DISTRIBUTION ---
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

                row_data = {
                    "Thickness": thick,
                    "Feature": feat,
                    "Customer Spec Limit": spec_str, # Trưng ra để Sếp so sánh
                    "Data-Driven Release Range": release_range, # Tính từ thực tế dải hàng tốt
                    "Target Goal": target_goal,
                    "Mill Range (Proposed)": mill_range,
                    "Segment Distribution": seg_dist,
                    "Status": "✅ Safe" if (low is None or (mean_val - sigma_factor*std_val) >= low) else "⚠ Risk"
                }
                
                status_list.append(row_data)
                all_export_data.append(row_data)

            display_df_3 = pd.DataFrame(status_list)
            if not display_df_3.empty:
                display_df_3 = display_df_3.drop(columns=['Thickness'])
            st.dataframe(display_df_3, use_container_width=True, hide_index=True)

            # --- I-MR Charts ---
            for feat in mech_features:
                if feat in plot_data_dict[thick]:
                    v = plot_data_dict[thick][feat] 
                    if len(v) > 1:
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
                        
                        m_v = np.mean(v)
                        s_v = np.std(v, ddof=1)
                        U, L = m_v + sigma_factor*s_v, m_v - sigma_factor*s_v
                        
                        ax1.plot(v, marker='o', color='blue', markersize=4)
                        ax1.axhline(m_v, color='green', ls='--', label=f'Mean: {int(round(m_v))}')
                        ax1.axhline(U, color='red', ls='--', label=f'UCL: {int(round(U))}')
                        ax1.axhline(L, color='red', ls='--', label=f'LCL: {int(round(L))}')
                        ax1.set_title(f"Individuals Chart (A-B & Above Only): {feat}")
                        ax1.legend(loc='upper right', fontsize=8)
                        
                        MR = np.abs(np.diff(v))
                        ax2.plot(MR, marker='o', color='orange', markersize=4)
                        ax2.axhline(np.mean(MR), color='green', ls='--', label=f'MR Mean: {int(round(np.mean(MR)))}')
                        ax2.set_title("Moving Range Chart")
                        ax2.legend(loc='upper right', fontsize=8)
                        
                        fig.tight_layout(pad=3.0)
                        st.pyplot(fig)
            st.markdown("---")

        # --- EXPORT FINAL ---
        if all_export_data:
            st.markdown("### 📥 Download Final QC Report")
            towrite = io.BytesIO()
            pd.DataFrame(all_export_data).to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button(label="📥 Download Executive Report (Excel)", data=towrite, 
                               file_name="QC_Mill_Range_Report.xlsx")
else:
    st.info("Please upload an Excel file.")
