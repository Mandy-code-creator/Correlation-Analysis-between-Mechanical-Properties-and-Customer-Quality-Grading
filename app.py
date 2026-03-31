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

st.title("📊 Mechanical Properties & Quality Yield Optimizer (Standard QC View)")
st.markdown("---")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your Excel data (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip() 

    # --- 2. DATA PREPROCESSING (LẤY ĐÚNG CÁC CỘT CÓ CHỮ 數) ---
    count_cols = ['A-B+數', 'A-B數', 'A-B-數', 'B+數', 'B數']
    count_cols = [col for col in count_cols if col in df.columns]
    
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Tính tổng số lượng cuộn thực tế
    df['Total_Count'] = df[count_cols].sum(axis=1)

    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
    mech_features = [feat for feat in mech_features if feat in df.columns]
    for feat in mech_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')

    # 1. TÌM GIÁ TRỊ CAO NHẤT TOÀN CỤC ĐỂ ĐỒNG NHẤT THANG ĐO
    max_counts = []
    for f in ['YS', 'TS', 'EL', 'YPE']:
        if f in df.columns:
            temp_total = df[count_cols].sum(axis=1)
            max_counts.append(temp_total.max())
    
    global_y_limit = max(max_counts) * 1.25 if max_counts else 600

    # --- 3. CREATE TABS ---
    tab1, tab2, tab3 = st.tabs([
        "1. Summary & Yields", 
        "2. Distribution Analysis (Standard View)",
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
        int_cols = [c for c in (count_cols + ['Total Coils', 'No.']) if c in display_df.columns]
        for c in int_cols:
            display_df[c] = display_df[c].astype(int)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- TAB 2: DISTRIBUTION (CÂN ĐỐI + NHÃN SO LE) ---
    with tab2:
        st.header("2. Mechanical Properties Distribution Analysis")
        
        def plot_standard_dist(ax, data, feat, title_suffix, is_right_col=False):
            N_t = data['Total_Count'].sum()
            k_b = 15 
            color_map = {'A-B+數': '#2ca02c', 'A-B-數': '#ff7f0e', 'B+數': '#d62728', 'B數': '#9467bd', 'A-B數': '#1f77b4'}
            mean_inf = []
            
            for col_n in count_cols:
                temp_d = data[[feat, col_n]].dropna()
                temp_d = temp_d[temp_d[col_n] > 0]
                if len(temp_d) >= 1:
                    vals_d, wgts_d = temp_d[feat].values, temp_d[col_n].values
                    color = color_map.get(col_n, '#7f7f7f')
                    sns.histplot(x=vals_d, weights=wgts_d, label=col_n.replace('數',''), 
                                 color=color, bins=k_b, stat='count', alpha=0.4, ax=ax, edgecolor='white')
                    
                    m_d = np.average(vals_d, weights=wgts_d)
                    s_d = np.sqrt(np.average((vals_d - m_d)**2, weights=wgts_d))
                    ax.axvline(m_d, color=color, ls='--', lw=2)
                    mean_inf.append({'val': m_d, 'color': color})

                    if len(vals_d) > 2 and s_d > 0:
                        x_range = np.linspace(m_d - 4*s_d, m_d + 4*s_d, 100)
                        bin_w = (vals_d.max() - vals_d.min()) / k_b if vals_d.max() != vals_d.min() else 1
                        ax.plot(x_range, stats.norm.pdf(x_range, m_d, s_d) * wgts_d.sum() * bin_w, color=color, lw=2)

            ax.set_ylim(0, global_y_limit)

            if mean_inf:
                mean_inf.sort(key=lambda x: x['val'])
                levels = [0.92, 0.82, 0.72, 0.62] 
                for i_m, info in enumerate(mean_inf):
                    y_p = global_y_limit * levels[i_m % len(levels)]
                    ax.text(info['val'], y_p, f"{info['val']:.1f}", color=info['color'], 
                            fontsize=8, fontweight='bold', ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.9, edgecolor=info['color'], boxstyle='round,pad=0.2'))

            ax.set_title(f"{feat} - {title_suffix}", fontsize=12, fontweight='bold')
            if is_right_col: 
                ax.legend(title="Grade", bbox_to_anchor=(1.05, 1), loc='upper left')

        st.subheader("🌐 Overall Factory Distribution")
        ov_cols = st.columns(2)
        for idx, feat in enumerate(['YS', 'TS', 'EL', 'YPE']):
            if feat in mech_features:
                with ov_cols[idx % 2]:
                    fig_ov, ax_ov = plt.subplots(figsize=(10, 5))
                    plot_standard_dist(ax_ov, df, feat, "Overall", is_right_col=(idx % 2 != 0))
                    st.pyplot(fig_ov)
                    fig_ov.savefig(f"overall_{feat}.png", bbox_inches='tight')

        st.markdown("---")
        st.subheader("🔍 Detailed Distribution per Thickness")
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=lambda x: float(x))
        for thick in thickness_list:
            df_thick = df[df['厚度歸類'] == thick]
            st.markdown(f"### 📏 Thickness: **{thick}**")
            cols_dist = st.columns(2)
            for idx, feat in enumerate(['YS', 'TS', 'EL', 'YPE']):
                if feat in mech_features:
                    with cols_dist[idx % 2]:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        plot_standard_dist(ax, df_thick, feat, f"Thick {thick}", is_right_col=(idx % 2 != 0))
                        st.pyplot(fig)
                        fig.savefig(f"dist_{feat}_{thick}.png", bbox_inches='tight')

# --- TAB 3: OPTIMIZATION (GỘP A-B TRỞ LÊN & THÊM GIỚI HẠN TỔNG THỂ) ---
    with tab3:
        st.header("3. Production Control Limits & Goals (A-B & Above Focused)")
        
        # 1. Cấu hình chung
        sigma_choice = st.radio("Select Sigma Factor for Mill Safety Zone", [2.0, 2.5, 3.0], index=0)
        spec_limits = {"YS": (405, 500), "TS": (415, 550), "EL": (25, None), "YPE": (4, None)}
        good_cols = [c for c in ['A-B+數', 'A-B數'] if c in df.columns]

        # ---------------------------------------------------------
        # PHẦN MỚI: TÍNH TOÁN GIỚI HẠN KIỂM SOÁT TỔNG THỂ (OVERALL)
        # ---------------------------------------------------------
        st.subheader("🌐 Overall Factory Control Limits (0.5 ~ 0.8 Combined)")
        st.info("Bảng này tính toán mục tiêu dựa trên toàn bộ dữ liệu hàng đạt (A-B trở lên) của nhà máy.")
        
        overall_status = []
        for feat in mech_features:
            if good_cols:
                # Lọc dữ liệu hàng tốt trên toàn bộ dataframe
                df_overall = df[[feat] + good_cols].dropna(subset=[feat])
                df_overall['Good_Qty'] = df_overall[good_cols].sum(axis=1)
                df_overall = df_overall[df_overall['Good_Qty'] > 0]
                
                if not df_overall.empty:
                    v_o, w_o = df_overall[feat].values, df_overall['Good_Qty'].values
                    # Lọc nhiễu IQR tổng thể
                    q1_o, q3_o = np.percentile(v_o, 25), np.percentile(v_o, 75)
                    iqr_o = q3_o - q1_o
                    mask_o = (v_o >= q1_o - 1.5*iqr_o) & (v_o <= q3_o + 1.5*iqr_o)
                    vf_o, wf_o = v_o[mask_o], w_o[mask_o]
                    
                    m_ov = np.average(vf_o, weights=wf_o)
                    s_ov = np.sqrt(np.average((vf_o - m_ov)**2, weights=wf_o))
                    
                    low, high = spec_limits.get(feat, (None, None))
                    spec_s = f"{int(low)}–{int(high)}" if low and high else (f">={int(low)}" if low else "N/A")
                    
                    overall_status.append({
                        "Feature": feat,
                        "Standard Limit": spec_s,
                        "Overall Target": int(round(m_ov)),
                        f"Overall Tol (±{sigma_choice}σ)": int(round(sigma_choice * s_ov)),
                        "Overall Mill Range": f"{int(round(m_ov - sigma_choice*s_ov))}–{int(round(m_ov + sigma_choice*s_ov))}"
                    })

        if overall_status:
            st.dataframe(pd.DataFrame(overall_status), use_container_width=True, hide_index=True)

        st.markdown("---")

        # ---------------------------------------------------------
        # PHẦN CHI TIẾT: GIỚI HẠN THEO TỪNG ĐỘ DÀY (GIỮ NGUYÊN NHƯ CŨ)
        # ---------------------------------------------------------
        st.subheader("🔍 Detailed Limits per Thickness Category")
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=lambda x: float(x))
        all_export_data = []
        plot_data_dict = {}

        for thick in thickness_list:
            st.markdown(f"#### 📏 Thickness Category: **{thick}**")
            df_t = df[df['厚度歸類'] == thick]
            plot_data_dict[thick] = {}
            thick_status = []
            
            for feat in mech_features:
                temp_calc = df_t[[feat] + good_cols].dropna(subset=[feat]) if good_cols else pd.DataFrame()
                if not temp_calc.empty:
                    temp_calc['Good_Qty'] = temp_calc[good_cols].sum(axis=1)
                    temp_calc = temp_calc[temp_calc['Good_Qty'] > 0]

                low, high = spec_limits.get(feat, (None, None))
                spec_str = f"{int(low)}–{int(high)}" if low and high else (f">={int(low)}" if low else "N/A")

                if not temp_calc.empty:
                    v_f, w_f = temp_calc[feat].values, temp_calc['Good_Qty'].values
                    q1, q3 = np.percentile(v_f, 25), np.percentile(v_f, 75)
                    iqr = q3 - q1
                    mask = (v_f >= q1 - 1.5*iqr) & (v_f <= q3 + 1.5*iqr)
                    vf, wf = v_f[mask] if mask.sum() > 0 else v_f, w_f[mask] if mask.sum() > 0 else w_f
                    
                    m_v = np.average(vf, weights=wf)
                    s_v = np.sqrt(np.average((vf - m_v)**2, weights=wf))
                    plot_data_dict[thick][feat] = {'values': vf, 'mean': m_v, 'std': s_v}
                    
                    row = {
                        "Feature": feat, "Current Limit": spec_str,
                        "Target Goal": int(round(m_v)),
                        f"Tol (±{sigma_choice}σ)": int(round(sigma_choice * s_v)),
                        "Proposed Mill Range": f"{int(round(m_v - sigma_choice*s_v))}–{int(round(m_v + sigma_choice*s_v))}"
                    }
                    thick_status.append(row)
                    exp_row = row.copy(); exp_row['Thickness'] = thick; all_export_data.append(exp_row)

            st.dataframe(pd.DataFrame(thick_status), use_container_width=True, hide_index=True)
            
            # Biểu đồ I-MR cho từng độ dày
            c_imr = st.columns(2)
            top4 = [f for f in ['YS', 'TS', 'EL', 'YPE'] if f in plot_data_dict[thick]]
            for idx, f in enumerate(top4):
                with c_imr[idx % 2]:
                    d = plot_data_dict[thick][f]
                    v, mv, sv = d['values'], d['mean'], d['std']
                    if len(v) > 1:
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
                        U, L = mv + sigma_choice*sv, mv - sigma_choice*sv
                        ax1.plot(v, marker='o', color='blue', ms=3, lw=1)
                        ax1.axhline(mv, color='green', ls='--')
                        ax1.axhline(U, color='red', ls='--'); ax1.axhline(L, color='red', ls='--')
                        ax1.set_title(f"I-Chart: {f} (Thick {thick})")
                        
                        mr = np.abs(np.diff(v)); mrm = np.mean(mr)
                        ax2.plot(mr, marker='o', color='orange', ms=3, lw=1)
                        ax2.axhline(mrm, color='green', ls='--')
                        ax2.set_title("Moving Range")
                        fig.tight_layout(); st.pyplot(fig)
    # --- PDF EXPORT ---
    st.markdown("### 🖨️ Export PDF Executive Report")
    def clean(t): return str(t).replace('±', '+/-').replace('–', '-').encode('latin-1', 'ignore').decode('latin-1')

    if st.button("Generate & Download PDF"):
        pdf = FPDF(orientation='L')
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16); pdf.cell(0, 10, "QC MECHANICAL PROPERTIES REPORT", ln=True, align="C"); pdf.ln(5)
        pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, "1. Quality Summary", ln=True)
        pdf.set_font('Arial', 'B', 8); cw = [10, 20] + [15]*len(count_cols) + [20] + [15]*len(count_cols)
        for i, col in enumerate(display_df.columns): pdf.cell(cw[i] if i < len(cw) else 20, 8, clean(col), border=1, align='C')
        pdf.ln(); pdf.set_font('Arial', '', 8)
        for _, r in display_df.iterrows():
            for i, v in enumerate(r): pdf.cell(cw[i] if i < len(cw) else 20, 8, clean(v), border=1, align='C')
            pdf.ln()

        for thick in thickness_list:
            pdf.add_page(); pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, f"2. Distribution - Thickness: {thick}", ln=True); ys = pdf.get_y()
            for idx, f in enumerate(['YS', 'TS', 'EL', 'YPE']):
                path = f"dist_{f}_{thick}.png"
                if os.path.exists(path): pdf.image(path, x=(10 if idx%2==0 else 150), y=(ys if idx<2 else ys+75), w=135)
            
            pdf.add_page(); pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, f"3. Control Limits & Trending - Thickness: {thick}", ln=True)
            heads = ["Feature", "Current Limit", "Segment Dist", "Release Range", "Target", "Tol", "Mill Range"]; c_w3 = [25, 25, 80, 35, 15, 20, 40]
            pdf.set_font('Arial', 'B', 8)
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
