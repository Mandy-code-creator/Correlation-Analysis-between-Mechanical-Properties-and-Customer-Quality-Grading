import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import math
import io
from fpdf import FPDF
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QC Mechanical Properties Optimizer", layout="wide")

st.title("📊 Mechanical Properties & Quality Yield Optimizer (Grade: A-B+ Focused)")
st.markdown("---")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your Excel data (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip() 

    # --- 2. DATA PREPROCESSING (UPDATED GRADES) ---
    # Cấu hình lại các cột cấp độ theo thực tế của bạn
    count_cols = ['A-B+', 'A-B-', 'B+', 'B']
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
        "2. Distribution Analysis (A-B+ Focus)",
        "3. PRODUCTION CONTROL LIMITS (EXECUTIVE VIEW)"
    ])

    # --- TAB 1: SUMMARY (Khớp 100% với tính tay) ---
    with tab1:
        st.header("1. Quality Summary by Thickness")
        # Gom nhóm và cộng tổng theo đúng các cột Mandy có
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Total Coils'] = summary_df[count_cols].sum(axis=1)
        
        # Tính tỷ lệ % cho từng loại
        for col in count_cols:
            summary_df[f"% {col}"] = (summary_df[col] / summary_df['Total Coils'] * 100).fillna(0).round(2)
            
        display_df = summary_df.copy()
        display_df.rename(columns={'厚度歸類': 'Thickness'}, inplace=True)
        display_df.insert(0, 'No.', range(1, len(display_df) + 1))
        
        # Ép kiểu số nguyên để mất cái đuôi .0
        int_cols = [c for c in (count_cols + ['Total Coils', 'No.']) if c in display_df.columns]
        for c in int_cols:
            display_df[c] = display_df[c].astype(int)
                
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- TAB 2: DISTRIBUTION (2x2 VIEW) ---
    with tab2:
        st.header("2. Distribution Analysis (A-B+ vs Others)")
        # Mapping màu sắc cho các cấp độ mới
        color_map = {'A-B+': '#2ca02c', 'A-B-': '#ff7f0e', 'B+': '#d62728', 'B': '#9467bd'}
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)

        def plot_feature_dist(ax, data, feat, thick, is_right_col=False):
            N_t = data['Total_Count'].sum()
            k_b = max(int(1 + 3.322 * math.log10(N_t)) if N_t > 0 else 10, 5)
            
            for col_n in count_cols:
                temp_d = data[[feat, col_n]].dropna()
                temp_d = temp_d[temp_d[col_n] > 0]
                if len(temp_d) >= 1:
                    vals_d, wgts_d = temp_d[feat].values, temp_d[col_n].values
                    color = color_map.get(col_n, '#1f77b4')
                    
                    sns.histplot(x=vals_d, weights=wgts_d, label=col_n, color=color, bins=k_b, 
                                 stat='count', alpha=0.4, ax=ax, edgecolor='white')
                    
                    if len(vals_d) > 2: # Vẽ đường cong cho dữ liệu đủ lớn
                        m_d = np.average(vals_d, weights=wgts_d)
                        s_d = np.sqrt(np.average((vals_d - m_d)**2, weights=wgts_d))
                        if s_d > 0:
                            x_range = np.linspace(m_d - 4*s_d, m_d + 4*s_d, 100)
                            bin_w = (vals_d.max() - vals_d.min()) / k_b if vals_d.max() != vals_d.min() else 1
                            ax.plot(x_range, stats.norm.pdf(x_range, m_d, s_d) * wgts_d.sum() * bin_w, color=color, lw=2)

            ax.set_title(f"{feat} - Thick: {thick}")
            if is_right_col: ax.legend(title="Grade", bbox_to_anchor=(1.05, 1), loc='upper left')

        for thickness in thickness_list:
            df_thickness = df[df['厚度歸類'] == thickness]
            st.markdown(f"### 📏 Analysis for Thickness: **{thickness}**")
            cols_dist = st.columns(2)
            for idx, feat in enumerate(['YS', 'TS', 'EL', 'YPE']):
                if feat in mech_features:
                    with cols_dist[idx % 2]:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        plot_feature_dist(ax, df_thickness, feat, thickness, is_right_col=(idx % 2 != 0))
                        st.pyplot(fig)
                        fig.savefig(f"dist_{feat}_{thickness}.png", bbox_inches='tight')
            st.markdown("---")

    # --- TAB 3: OPTIMIZATION (EXECUTIVE 2x2 VIEW) ---
    with tab3:
        st.header("3. Production Control Limits & Goals (Based on A-B+ Data)")
        sigma_choice = st.radio("Select Sigma Factor", [2.0, 2.5, 3.0], index=0)
        spec_limits = {"YS": (405, 500), "TS": (415, 550), "EL": (25, None), "YPE": (4, None)}
        all_export_data = []
        plot_data_dict = {}

        for thick in thickness_list:
            st.subheader(f"Thickness Category: {thick}")
            df_t = df[df['厚度歸類'] == thick]
            plot_data_dict[thick] = {}
            status_list = []
            
            # CHỈ LẤY GRADE A-B+ ĐỂ TÍNH TOÁN TARGET
            target_grade = 'A-B+' if 'A-B+' in df_t.columns else None

            for feat in mech_features:
                if target_grade:
                    temp_calc = df_t[[feat, target_grade]].dropna(subset=[feat])
                    temp_calc = temp_calc[temp_calc[target_grade] > 0]
                else:
                    temp_calc = pd.DataFrame()

                low, high = spec_limits.get(feat, (None, None))
                spec_str = f"{int(low)}–{int(high)}" if low and high else (f">={int(low)}" if low else "N/A")

                if not temp_calc.empty:
                    vals_raw = temp_calc[feat].values
                    wgts_raw = temp_calc[target_grade].values
                    
                    # IQR FILTERING
                    q1, q3 = np.percentile(vals_raw, 25), np.percentile(vals_raw, 75)
                    iqr = q3 - q1
                    mask = (vals_raw >= q1 - 1.5*iqr) & (vals_raw <= q3 + 1.5*iqr)
                    
                    v_f = vals_raw[mask] if mask.sum() > 0 else vals_raw
                    w_f = wgts_raw[mask] if mask.sum() > 0 else wgts_raw
                    
                    m_val = np.average(v_f, weights=w_f)
                    s_val = np.sqrt(np.average((v_f - m_val)**2, weights=w_f))
                    
                    plot_data_dict[thick][feat] = {'values': v_f, 'mean': m_val, 'std': s_val}
                    t_goal = int(round(m_val))
                    rel_range = f"{int(round(m_val - 3*s_val))}–{int(round(m_val + 3*s_val))}"
                    m_range = f"{int(round(m_val - sigma_choice*s_val))}–{int(round(m_val + sigma_choice*s_val))}"
                    tol_val = int(round(sigma_choice * s_val))
                else:
                    t_goal, rel_range, m_range, tol_val, m_val, s_val = "N/A", "N/A", "N/A", "N/A", 0, 0

                total_n = df_t[count_cols].sum().sum()
                seg_dist = ", ".join([f"{k}:{int(round(df_t[k].sum()/total_n*100))}%" for k in count_cols]) if total_n > 0 else "N/A"

                row = {"Feature": feat, "Current Control Limit": spec_str, "Segment Distribution": seg_dist,
                       "Data-Driven Release Range": rel_range, "Target Goal": t_goal,
                       f"Tolerance (±{sigma_choice}σ)": tol_val, "Mill Range (Proposed)": m_range}
                status_list.append(row)
                export_row = row.copy(); export_row['Thickness'] = thick
                all_export_data.append(export_row)

            st.dataframe(pd.DataFrame(status_list), use_container_width=True, hide_index=True)

            # I-MR 2x2 View
            top_4 = [f for f in ['YS', 'TS', 'EL', 'YPE'] if f in plot_data_dict[thick]]
            cols_imr = st.columns(2)
            for idx, feat in enumerate(top_4):
                with cols_imr[idx % 2]:
                    d = plot_data_dict[thick][feat]
                    v, mv, sv = d['values'], d['mean'], d['std']
                    if len(v) > 1:
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                        ucl, lcl = mv + sigma_choice*sv, mv - sigma_choice*sv
                        ax1.plot(v, marker='o', color='blue', ms=3, lw=1)
                        outs = np.where((v > ucl) | (v < lcl))[0]
                        ax1.scatter(outs, v[outs], color='red', s=30, zorder=3)
                        ax1.axhline(mv, color='green', ls='--'); ax1.axhline(ucl, color='red', ls='--'); ax1.axhline(lcl, color='red', ls='--')
                        ax1.set_title(f"I-Chart: {feat}")
                        
                        mr = np.abs(np.diff(v))
                        mrm, mru = np.mean(mr), 3.267 * np.mean(mr)
                        ax2.plot(mr, marker='o', color='orange', ms=3, lw=1)
                        mrouts = np.where(mr > mru)[0]
                        ax2.scatter(mrouts, mr[mrouts], color='red', s=30, zorder=3)
                        ax2.axhline(mrm, color='green', ls='--'); ax2.axhline(mru, color='red', ls='--')
                        ax2.set_title("MR-Chart")
                        fig.tight_layout(); st.pyplot(fig)
                        fig.savefig(f"imr_{feat}_{thick}.png", bbox_inches='tight')
            st.markdown("---")

    # --- PDF EXPORT ---
    st.markdown("### 🖨️ Export PDF Executive Report")
    def clean(t): return str(t).replace('±', '+/-').replace('–', '-').encode('latin-1', 'ignore').decode('latin-1')

    if st.button("Generate & Download PDF"):
        pdf = FPDF(orientation='L')
        # Page 1: Summary
        pdf.add_page(); pdf.set_font('Arial', 'B', 16); pdf.cell(0, 10, "QC MECHANICAL PROPERTIES REPORT", ln=True, align="C"); pdf.ln(5)
        pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, "1. Quality Summary", ln=True)
        pdf.set_font('Arial', 'B', 8); cw = [10, 20] + [15]*len(count_cols) + [20] + [15]*len(count_cols)
        for i, col in enumerate(display_df.columns): pdf.cell(cw[i] if i < len(cw) else 20, 8, clean(col), border=1, align='C')
        pdf.ln(); pdf.set_font('Arial', '', 8)
        for _, r in display_df.iterrows():
            for i, v in enumerate(r):
                if isinstance(v, (int, float)) and v == int(v): v = int(v)
                pdf.cell(cw[i] if i < len(cw) else 20, 8, clean(v), border=1, align='C')
            pdf.ln()

        # Grid Pages
        for thick in thickness_list:
            # Distribution Page
            pdf.add_page(); pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, f"2. Distribution - Thickness: {thick}", ln=True); ys = pdf.get_y()
            for idx, f in enumerate(['YS', 'TS', 'EL', 'YPE']):
                path = f"dist_{f}_{thick}.png"
                if os.path.exists(path): pdf.image(path, x=(10 if idx%2==0 else 150), y=(ys if idx<2 else ys+75), w=135)
            
            # Table & Trending Page
            pdf.add_page(); pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, f"3. Control Limits & Trending - Thickness: {thick}", ln=True)
            pdf.set_font('Arial', 'B', 8); heads = ["Feature", "Current Limit", "Segment Dist", "Release Range", "Target", "Tol", "Mill Range"]; c_w3 = [25, 25, 80, 35, 15, 20, 40]
            for i, h in enumerate(heads): pdf.cell(c_w3[i], 7, clean(h), border=1, align='C')
            pdf.ln(); pdf.set_font('Arial', '', 7)
            for row in all_export_data:
                if row['Thickness'] == thick:
                    v_list = [row["Feature"], row["Current Control Limit"], row["Segment Distribution"], row["Data-Driven Release Range"], row["Target Goal"], row[f"Tolerance (±{sigma_choice}σ)"], row["Mill Range (Proposed)"]]
                    for i, v in enumerate(v_list): pdf.cell(c_w3[i], 7, clean(v), border=1, align='C')
                    pdf.ln()
            yt = pdf.get_y() + 5
            for idx, f in enumerate(['YS', 'TS', 'EL', 'YPE']):
                path = f"imr_{f}_{thick}.png"
                if os.path.exists(path): pdf.image(path, x=(10 if idx%2==0 else 150), y=(yt if idx<2 else yt+75), w=135)

        pdf.output("QC_Report.pdf")
        with open("QC_Report.pdf", "rb") as f:
            st.download_button("📥 Download Final PDF", f.read(), "QC_Report.pdf", "application/pdf")
