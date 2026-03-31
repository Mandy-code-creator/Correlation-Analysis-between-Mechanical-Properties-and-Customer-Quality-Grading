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
from matplotlib.patches import Patch

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QC Mechanical Properties Optimizer", layout="wide")

st.title("📊 Mechanical Properties & Quality Yield Optimizer (Standard QC View)")
st.markdown("---")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your Excel data (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip() 

    # Khởi tạo biến export để tránh NameError
    all_export_data = []
    
    # --- 2. DATA PREPROCESSING ---
    count_cols = ['A-B+數', 'A-B數', 'A-B-數', 'B+數', 'B數']
    count_cols = [col for col in count_cols if col in df.columns]
    
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['Total_Count'] = df[count_cols].sum(axis=1)

    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
    mech_features = [feat for feat in mech_features if feat in df.columns]
    for feat in mech_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')

    thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=lambda x: float(x))

    # --- 3. CREATE TABS ---
    tab1, tab2, tab3 = st.tabs([
        "1. Summary & Yields", 
        "2. Distribution Analysis",
        "3. Control Limits & Goals"
    ])

    # --- TAB 1: SUMMARY ---
    with tab1:
        st.header("1. Quality Summary by Thickness")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Total Data'] = summary_df[count_cols].sum(axis=1)
        for col in count_cols:
            summary_df[f"% {col.replace('數','')}"] = (summary_df[col] / summary_df['Total Data'] * 100).fillna(0).round(2)
        display_df = summary_df.copy()
        display_df.rename(columns={'厚度歸類': 'Thickness'}, inplace=True)
        for col in count_cols:
            display_df.rename(columns={col: col.replace('數','')}, inplace=True)
        display_df.insert(0, 'No.', range(1, len(display_df) + 1))
        
        int_cols = [c.replace('數','') for c in count_cols] + ['Total Data', 'No.']
        for c in int_cols:
            if c in display_df.columns:
                display_df[c] = display_df[c].astype(int)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- TAB 2: DISTRIBUTION (SMART AUTO SCALING FIX) ---
    with tab2:
        st.header("2. Mechanical Properties Distribution Analysis")

        # Calculate Global Max for Overall section only
        max_counts_ov = []
        for f in ['YS', 'TS', 'EL', 'YPE']:
            if f in df.columns:
                temp_total = df[count_cols].sum(axis=1)
                max_counts_ov.append(temp_total.max())
        global_y_ov = max(max_counts_ov) * 1.3 if max_counts_ov else 300

        def plot_standard_dist(ax, data, feat, title_suffix, is_overall=False, is_right_col=False):
            N_t = data['Total_Count'].sum()
            k_b = 15 
            color_map = {'A-B+數': '#1f77b4', 'A-B-數': '#ff7f0e', 'B+數': '#d62728', 'B數': '#7f7f7f', 'A-B數': '#9467bd'}
            mean_inf = []
            
            # Add subtle dotted grid
            ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)
            
            # Vẽ từng Grade
            for col_n in count_cols:
                temp_d = data[[feat, col_n]].dropna()
                temp_d = temp_d[temp_d[col_n] > 0]
                if not temp_d.empty:
                    vals_d, wgts_d = temp_d[feat].values, temp_d[col_n].values
                    color = color_map.get(col_n, '#7f7f7f')
                    
                    # Vẽ Histogram
                    sns.histplot(x=vals_d, weights=wgts_d, label=col_n.replace('數',''), 
                                 color=color, bins=k_b, stat='count', alpha=0.4, ax=ax, edgecolor='white', zorder=2)
                    
                    # Tính Weighted Mean
                    m_d = np.average(vals_d, weights=wgts_d)
                    s_d = np.sqrt(np.average((vals_d - m_d)**2, weights=wgts_d))
                    
                    # Vẽ đường kẻ Mean và lưu thông tin nhãn
                    ax.axvline(m_d, color=color, ls='--', lw=1.5, zorder=3)
                    mean_inf.append({'val': m_d, 'color': color})

                    # Vẽ đường cong chuẩn
                    if len(vals_d) > 2 and s_d > 0:
                        x_range = np.linspace(vals_d.min() - 5, vals_d.max() + 5, 100)
                        bin_w = (vals_d.max() - vals_d.min()) / k_b if vals_d.max() != vals_d.min() else 1
                        ax.plot(x_range, stats.norm.pdf(x_range, m_d, s_d) * wgts_d.sum() * bin_w, color=color, lw=2, zorder=4)

            # --- SMART SCALING FIX: Ép thang Y tự động cho từng biểu đồ ---
            if is_overall:
                current_limit = global_y_ov # Giữ thang đo lớn cho Overall
            else:
                # Tự tính local max cho độ dày và feature này, nới hờ 35% cho nhãn số
                local_max = data[count_cols].sum(axis=1).max()
                current_limit = local_max * 1.35 if local_max > 0 else 100
            
            ax.set_ylim(0, current_limit)

            # Hiển thị nhãn giá trị Mean so le
            if mean_inf:
                mean_inf.sort(key=lambda x: x['val'])
                y_max = ax.get_ylim()[1]
                levels = [0.90, 0.82, 0.74, 0.66]
                for i_m, info in enumerate(mean_inf):
                    y_p = y_max * levels[i_m % len(levels)] # Tránh đè nhãn
                    ax.text(info['val'], y_p, f"{int(round(info['val']))}", color=info['color'], 
                            fontsize=9, fontweight='bold', ha='center', va='center', zorder=5,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor=info['color'], boxstyle='round,pad=0.2'))

            # FORMAT LEGEND ĐỂ KHÔNG MẤT CẤP CHẤT LƯỢNG
            ax.set_title(f"{feat} (Thick: {title_suffix})", fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel("Count", fontsize=10)
            if is_right_col:
                legend_elements = [Patch(facecolor=color_map[k], edgecolor='white', label=k.replace('數',''), alpha=0.5) for k in color_map]
                ax.legend(handles=legend_elements, title="Grade", title_fontsize='10', fontsize='9', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

        # --- PHẦN VẼ ---
        st.subheader("🌐 Factory Overall Distribution (Balanced Combined View)")
        ov_cols = st.columns(2)
        for idx, feat in enumerate(['YS', 'TS', 'EL', 'YPE']):
            if feat in mech_features:
                with ov_cols[idx % 2]:
                    fig_ov, ax_ov = plt.subplots(figsize=(10, 5))
                    # Pass True for Overall scaling
                    plot_standard_dist(ax_ov, df, feat, "Overall", is_overall=True, is_right_col=(idx % 2 != 0))
                    st.pyplot(fig_ov)
                    fig_ov.savefig(f"overall_{feat}.png", bbox_inches='tight')

        st.markdown("---")
        st.subheader("🔍 Detailed Distribution per Thickness Category")
        for thick in thickness_list:
            df_thick = df[df['厚度歸類'] == thick]
            st.markdown(f"### 📏 Category: **{thick}**")
            cols_dist = st.columns(2)
            for idx, feat in enumerate(['YS', 'TS', 'EL', 'YPE']):
                if feat in mech_features:
                    with cols_dist[idx % 2]:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        # Pass False to enable Smart Auto-scaling for each chart
                        plot_standard_dist(ax, df_thick, feat, thick, is_overall=False, is_right_col=(idx % 2 != 0))
                        st.pyplot(fig)
                        fig.savefig(f"dist_{feat}_{thick}.png", bbox_inches='tight')
            st.markdown("---")

    # --- TAB 3: OPTIMIZATION & I-MR CHARTS ---
    with tab3:
        st.header("3. Production Control Limits & Goals (A-B & Above Focused)")
        
        sigma_choice = st.radio("Select Confidence Interval (Sigma Factor)", [2.0, 2.5, 3.0], index=0)
        spec_limits = {"YS": (405, 500), "TS": (415, 550), "EL": (25, None), "YPE": (4, None)}
        good_cols = [c for c in ['A-B+數', 'A-B數'] if c in df.columns]

        # 1. OVERALL FACTORY (CẬP NHẬT ĐẦY ĐỦ THÔNG TIN)
        st.subheader("🌐 Overall Factory Performance Goals")
        overall_status = []
        
        # Tính tỷ lệ phần trăm phân bổ tổng thể (Segment Distribution)
        total_n_overall = df[count_cols].sum().sum()
        seg_dist_overall = "N/A" if total_n_overall == 0 else ", ".join([f"{k.replace('數','')}:{int(round(df[k].sum()/total_n_overall*100))}%" for k in count_cols])

        for feat in mech_features:
            if good_cols:
                df_ov = df[[feat] + good_cols].dropna(subset=[feat])
                df_ov['Good_Qty'] = df_ov[good_cols].sum(axis=1)
                df_ov = df_ov[df_ov['Good_Qty'] > 0]
                
                low, high = spec_limits.get(feat, (None, None))
                spec_str_ov = f"{int(low)}-{int(high)}" if low and high else (f">={int(low)}" if low else "N/A")

                if not df_ov.empty:
                    v, w = df_ov[feat].values, df_ov['Good_Qty'].values
                    m_ov = np.average(v, weights=w)
                    s_ov = np.sqrt(np.average((v - m_ov)**2, weights=w))
                    
                    overall_status.append({
                        "Feature": feat, 
                        "Current Control Limit": spec_str_ov,
                        "Segment Distribution": seg_dist_overall,
                        "Data-Driven Release Range": f"{int(round(m_ov - 3*s_ov))}-{int(round(m_ov + 3*s_ov))}",
                        "Target Goal": int(round(m_ov)),
                        f"Tolerance (±{sigma_choice}σ)": int(round(sigma_choice*s_ov)),
                        "Mill Range (Proposed)": f"{int(round(m_ov - sigma_choice*s_ov))}-{int(round(m_ov + sigma_choice*s_ov))}"
                    })
        st.dataframe(pd.DataFrame(overall_status), use_container_width=True, hide_index=True)

        st.markdown("---")
        
        # 2. LOCAL THICKNESS
        st.subheader("🔍 Local Control Limits & I-MR Trending")
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
                spec_str = f"{int(low)}-{int(high)}" if low and high else (f">={int(low)}" if low else "N/A")

                if not temp_calc.empty:
                    v, w = temp_calc[feat].values, temp_calc['Good_Qty'].values
                    mv = np.average(v, weights=w)
                    sv = np.sqrt(np.average((v - mv)**2, weights=w))
                    plot_data_dict[thick][feat] = {'values': v, 'mean': mv, 'std': sv}
                    
                    total_n = df_t[count_cols].sum().sum()
                    seg_dist = "N/A" if total_n == 0 else ", ".join([f"{k.replace('數','')}:{int(round(df_t[k].sum()/total_n*100))}%" for k in count_cols])
                    
                    row = {
                        "Feature": feat, 
                        "Current Control Limit": spec_str,
                        "Segment Distribution": seg_dist,
                        "Data-Driven Release Range": f"{int(round(mv - 3*sv))}-{int(round(mv + 3*sv))}",
                        "Target Goal": int(round(mv)),
                        f"Tolerance (±{sigma_choice}σ)": int(round(sigma_choice*sv)),
                        "Mill Range (Proposed)": f"{int(round(mv - sigma_choice*sv))}-{int(round(mv + sigma_choice*sv))}"
                    }
                    thick_status.append(row)
                    
                    exp_row = row.copy()
                    exp_row['Thickness'] = thick
                    all_export_data.append(exp_row)

            st.dataframe(pd.DataFrame(thick_status), use_container_width=True, hide_index=True)
            
            cols_imr = st.columns(2)
            top4 = [f for f in ['YS', 'TS', 'EL', 'YPE'] if f in plot_data_dict[thick]]
            for idx, f in enumerate(top4):
                with cols_imr[idx % 2]:
                    d = plot_data_dict[thick][f]
                    v, mv, sv = d['values'], d['mean'], d['std']
                    if len(v) > 1:
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})
                        ucl, lcl = mv + sigma_choice*sv, mv - sigma_choice*sv
                        
                        ax1.plot(v, marker='o', color='#1f77b4', ms=4, lw=1)
                        ax1.axhline(mv, color='green', ls='--')
                        ax1.axhline(ucl, color='red', ls='--')
                        ax1.axhline(lcl, color='red', ls='--')
                        ax1.set_title(f"I-Chart: {f} (Thick: {thick})", fontsize=11, fontweight='bold')
                        ax1.set_ylabel("Value")
                        
                        mr = np.abs(np.diff(v)); mrm = np.mean(mr); mru = 3.267 * mrm
                        ax2.plot(mr, marker='o', color='orange', ms=4, lw=1)
                        ax2.axhline(mrm, color='green', ls='--')
                        ax2.axhline(mru, color='red', ls='--')
                        ax2.set_title("Moving Range", fontsize=10)
                        ax2.set_ylabel("Range")
                        fig.tight_layout(); st.pyplot(fig)
                        fig.savefig(f"imr_{f}_{thick}.png", bbox_inches='tight') # LƯU ẢNH I-MR
            st.markdown("---")

    # --- EXPORT SECTION ---
    st.sidebar.header("📥 Export Options")
    if st.sidebar.button("Download Excel Report"):
        towrite = io.BytesIO()
        pd.DataFrame(all_export_data).to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        st.sidebar.download_button(label="Click to Download Excel", data=towrite, file_name="QC_Optimization_Report.xlsx")

    # --- PDF EXPORT (PHỤC HỒI CODE IN BIỂU ĐỒ) ---
    st.markdown("### 🖨️ Export PDF Executive Report")
    def clean(t): return str(t).replace('±', '+/-').replace('–', '-').encode('latin-1', 'ignore').decode('latin-1')

    if st.button("Generate & Download PDF"):
        pdf = FPDF(orientation='L')
        
        # Trang 1: Summary Data
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16); pdf.cell(0, 10, "QC MECHANICAL PROPERTIES REPORT", ln=True, align="C"); pdf.ln(5)
        pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, "1. Quality Summary", ln=True)
        pdf.set_font('Arial', 'B', 8); cw = [10, 20] + [15]*len(count_cols) + [20] + [15]*len(count_cols)
        for i, col in enumerate(display_df.columns): pdf.cell(cw[i] if i < len(cw) else 20, 8, clean(col), border=1, align='C')
        pdf.ln(); pdf.set_font('Arial', '', 8)
        for _, r in display_df.iterrows():
            for i, v in enumerate(r): pdf.cell(cw[i] if i < len(cw) else 20, 8, clean(v), border=1, align='C')
            pdf.ln()

        # Trang 2: Overall Distribution Charts
        pdf.add_page(); pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, "2. Factory Overall Distribution", ln=True); ys = pdf.get_y()
        for idx, f in enumerate(['YS', 'TS', 'EL', 'YPE']):
            path = f"overall_{f}.png"
            if os.path.exists(path): pdf.image(path, x=(10 if idx%2==0 else 150), y=(ys if idx<2 else ys+75), w=135)

        # Các trang chi tiết: Distribution và I-MR cho từng loại độ dày
        for thick in thickness_list:
            # Distribution PDF
            pdf.add_page(); pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, f"3. Distribution Analysis - Thickness: {thick}", ln=True); ys = pdf.get_y()
            for idx, f in enumerate(['YS', 'TS', 'EL', 'YPE']):
                path = f"dist_{f}_{thick}.png"
                if os.path.exists(path): pdf.image(path, x=(10 if idx%2==0 else 150), y=(ys if idx<2 else ys+75), w=135)
            
            # Control Limits & I-MR PDF
            pdf.add_page(); pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, f"4. Control Limits & Trending - Thickness: {thick}", ln=True)
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
