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

   # --- TAB 2: DISTRIBUTION (PROFESSIONAL QC STYLE) ---
    with tab2:
        st.header("2. Mechanical Properties Distribution Analysis")
        
        # Calculate Global Max for Overall section scaling
        max_counts_ov = []
        for f in ['YS', 'TS', 'EL', 'YPE']:
            if f in df.columns:
                max_counts_ov.append(df[count_cols].sum(axis=1).max())
        global_y_ov = max(max_counts_ov) * 1.3 if max_counts_ov else 300

        def plot_qc_dist(ax, data, feat, title, is_overall=False, is_right=False):
            # 1. Setup
            k_b = 15 
            color_map = {'A-B+數': '#1f77b4', 'A-B數': '#ff7f0e', 'A-B-數': '#9467bd', 'B+數': '#d62728', 'B數': '#7f7f7f'}
            mean_inf = []
            
            # Add subtle dotted grid (as seen in your image)
            ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)
            
            # 2. Plotting Histogram & Normal Curve
            for col_n in count_cols:
                temp_d = data[[feat, col_n]].dropna()
                temp_d = temp_d[temp_d[col_n] > 0]
                if not temp_d.empty:
                    vals, wgts = temp_d[feat].values, temp_d[col_n].values
                    color = color_map.get(col_n, '#7f7f7f')
                    
                    # Histogram
                    sns.histplot(x=vals, weights=wgts, label=col_n.replace('數',''), 
                                 color=color, bins=k_b, stat='count', alpha=0.4, ax=ax, zorder=2)
                    
                    # Stats
                    m = np.average(vals, weights=wgts)
                    s = np.sqrt(np.average((vals - m)**2, weights=wgts))
                    
                    # Vertical Mean Line
                    ax.axvline(m, color=color, ls='--', lw=1.5, zorder=3)
                    mean_inf.append({'val': m, 'color': color})

                    # Normal Distribution Curve
                    if len(vals) > 2 and s > 0:
                        x_r = np.linspace(vals.min() - 5, vals.max() + 5, 100)
                        bin_w = (vals.max() - vals.min()) / k_b if vals.max() != vals_d.min() else 1
                        ax.plot(x_r, stats.norm.pdf(x_r, m, s) * wgts.sum() * bin_w, color=color, lw=2, zorder=4)

            # 3. Smart Scaling
            y_limit = global_y_ov if is_overall else data[count_cols].sum(axis=1).max() * 1.3
            ax.set_ylim(0, y_limit if y_limit > 0 else 100)

            # 4. Precise Staggered Labels (Matches your Image)
            if mean_inf:
                mean_inf.sort(key=lambda x: x['val'])
                # Start labels near the top and step down
                y_max = ax.get_ylim()[1]
                levels = [0.90, 0.82, 0.74, 0.66, 0.58]
                
                for i, info in enumerate(mean_inf):
                    y_pos = y_max * levels[i % len(levels)]
                    ax.text(info['val'], y_pos, f"{int(round(info['val']))}", 
                            color=info['color'], fontsize=9, fontweight='bold',
                            ha='center', va='center', zorder=5,
                            bbox=dict(facecolor='white', alpha=0.9, edgecolor=info['color'], boxstyle='round,pad=0.3'))

            # 5. Title & Labels Formatting
            ax.set_title(f"{feat} (Thick: {title})", fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_xlabel("") # Keep X-axis clean
            
            # Professional Legend on the right
            if is_right:
                ax.legend(title="Grade", title_fontsize='10', fontsize='9', 
                          bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

        # --- Tab Drawing Logic (Overall & Detailed) ---
        st.subheader("🌐 Factory Overall Distribution")
        ov_cols = st.columns(2)
        for idx, feat in enumerate(['YS', 'TS', 'EL', 'YPE']):
            if feat in mech_features:
                with ov_cols[idx % 2]:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    plot_qc_dist(ax, df, feat, "Overall", is_overall=True, is_right=(idx % 2 != 0))
                    st.pyplot(fig)

        st.markdown("---")
        st.subheader("🔍 Thickness Detailed Analysis")
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=lambda x: float(x))
        for thick in thickness_list:
            df_thick = df[df['厚度歸類'] == thick]
            st.markdown(f"### 📏 Category: **{thick}**")
            cols_dist = st.columns(2)
            for idx, feat in enumerate(['YS', 'TS', 'EL', 'YPE']):
                if feat in mech_features:
                    with cols_dist[idx % 2]:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        plot_qc_dist(ax, df_thick, feat, thick, is_overall=False, is_right=(idx % 2 != 0))
                        st.pyplot(fig)
            st.markdown("---")

# --- TAB 3: OPTIMIZATION ---
    with tab3:
        st.header("3. Production Control Limits & Targets (A-B & Above)")
        sigma_choice = st.radio("Select Confidence Interval (Sigma Factor)", [2.0, 2.5, 3.0], index=0)
        spec_limits = {"YS": (405, 500), "TS": (415, 550), "EL": (25, None), "YPE": (4, None)}
        good_cols = [c for c in ['A-B+數', 'A-B數'] if c in df.columns]

        # OVERALL SECTION
        st.subheader("🌐 Overall Factory Performance Goals")
        overall_status = []
        for feat in mech_features:
            if good_cols:
                df_ov = df[[feat] + good_cols].dropna(subset=[feat])
                df_ov['Good_Qty'] = df_ov[good_cols].sum(axis=1)
                df_ov = df_ov[df_ov['Good_Qty'] > 0]
                if not df_ov.empty:
                    v, w = df_ov[feat].values, df_ov['Good_Qty'].values
                    q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
                    iqr = q3 - q1
                    mask = (v >= q1 - 1.5*iqr) & (v <= q3 + 1.5*iqr)
                    vf, wf = v[mask], w[mask]
                    m_ov = np.average(vf, weights=wf)
                    s_ov = np.sqrt(np.average((vf - m_ov)**2, weights=wf))
                    low, high = spec_limits.get(feat, (None, None))
                    spec_s = f"{int(low)}-{int(high)}" if low and high else (f">={int(low)}" if low else "N/A")
                    overall_status.append({
                        "Feature": feat, "Current Spec": spec_s,
                        "Global Target": int(round(m_ov)),
                        f"Global Tol (±{sigma_choice}σ)": int(round(sigma_choice * s_ov)),
                        "Factory Mill Range": f"{int(round(m_ov - sigma_choice*s_ov))}-{int(round(m_ov + sigma_choice*s_ov))}"
                    })
        st.dataframe(pd.DataFrame(overall_status), use_container_width=True, hide_index=True)

        st.markdown("---")
        # DETAILED SECTION
        st.subheader("🔍 Local Control Limits by Thickness")
        all_export_data = []
        for thick in thickness_list:
            st.markdown(f"#### Thickness Category: **{thick}**")
            df_t = df[df['厚度歸類'] == thick]
            thick_status = []
            for feat in mech_features:
                temp_calc = df_t[[feat] + good_cols].dropna(subset=[feat]) if good_cols else pd.DataFrame()
                if not temp_calc.empty:
                    temp_calc['Good_Qty'] = temp_calc[good_cols].sum(axis=1)
                    temp_calc = temp_calc[temp_calc['Good_Qty'] > 0]
                if not temp_calc.empty:
                    v, w = temp_calc[feat].values, temp_calc['Good_Qty'].values
                    m_v = np.average(v, weights=w)
                    s_v = np.sqrt(np.average((v - m_v)**2, weights=w))
                    low, high = spec_limits.get(feat, (None, None))
                    spec_str = f"{int(low)}-{int(high)}" if low and high else (f">={int(low)}" if low else "N/A")
                    row = {
                        "Feature": feat, "Spec Limit": spec_str,
                        "Local Target": int(round(m_v)),
                        f"Tol (±{sigma_choice}σ)": int(round(sigma_choice * s_v)),
                        "Proposed Mill Range": f"{int(round(m_v - sigma_choice*s_v))}-{int(round(m_v + sigma_choice*s_v))}"
                    }
                    thick_status.append(row)
                    exp_row = row.copy(); exp_row['Thickness'] = thick; all_export_data.append(exp_row)
            st.dataframe(pd.DataFrame(thick_status), use_container_width=True, hide_index=True)

    # --- EXPORT SECTION ---
    st.sidebar.header("📥 Export Options")
    if st.sidebar.button("Download Excel Report"):
        towrite = io.BytesIO()
        pd.DataFrame(all_export_data).to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        st.sidebar.download_button(label="Click to Download Excel", data=towrite, file_name="QC_Optimization_Report.xlsx")
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
