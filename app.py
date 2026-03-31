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

st.title("📊 Mechanical Properties & Quality Yield Optimizer (A-B+ Focus)")
st.markdown("---")

# --- 1. FILE UPLOAD ---
# Dòng này phải nằm sát lề trái
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

    # --- 3. CREATE TABS ---
    tab1, tab2, tab3 = st.tabs([
        "1. Summary & Yields", 
        "2. Distribution Analysis (A-B+ Focus)",
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

    # --- TAB 2: DISTRIBUTION (CẬP NHẬT: SO SÁNH TỔNG HỢP + CHI TIẾT) ---
    with tab2:
        st.header("2. Mechanical Properties Distribution Analysis")
        
        # Lấy danh sách độ dày và sắp xếp theo số để vẽ biểu đồ so sánh
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=lambda x: float(x))
        
        # ---------------------------------------------------------
        # PHẦN 1: BIỂU ĐỒ SO SÁNH TỔNG HỢP (0.5 ~ 0.8 TRÊN CÙNG 1 HÌNH)
        # ---------------------------------------------------------
        st.subheader("📌 Cross-Thickness Comparison (Combined View)")
        st.info("Biểu đồ dưới đây giúp so sánh xu hướng cơ lý giữa các độ dày khác nhau.")
        
        comp_cols = st.columns(2)
        # Vẽ 4 đặc tính quan trọng nhất
        for idx, feat in enumerate(['YS', 'TS', 'EL', 'YPE']):
            if feat in mech_features:
                with comp_cols[idx % 2]:
                    fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
                    
                    # Vẽ đường cong phân phối (KDE) cho từng độ dày
                    for thick in thickness_list:
                        df_thick = df[df['厚 độ 歸類'] == thick].dropna(subset=[feat])
                        if not df_thick.empty:
                            vals = df_thick[feat].values
                            # Trọng số là tổng tất cả các cuộn (A-B+, A-B, A-B-, B+, B)
                            weights = df_thick[count_cols].sum(axis=1).values
                            
                            if weights.sum() > 0:
                                sns.kdeplot(x=vals, weights=weights, label=f"Thick {thick}", 
                                            ax=ax_comp, fill=True, alpha=0.1, linewidth=2)
                    
                    ax_comp.set_title(f"Comparison: {feat} across Thicknesses", fontsize=12, fontweight='bold')
                    ax_comp.set_xlabel(feat)
                    ax_comp.set_ylabel("Density (Mật độ)")
                    ax_comp.legend(title="Thickness")
                    ax_comp.grid(axis='y', linestyle='--', alpha=0.3)
                    
                    st.pyplot(fig_comp)
                    # Lưu ảnh để xuất PDF (Trang so sánh tổng hợp)
                    fig_comp.savefig(f"compare_{feat}.png", bbox_inches='tight')

        st.markdown("---")

        # ---------------------------------------------------------
        # PHẦN 2: CHI TIẾT TỪNG ĐỘ DÀY (GIỮ NGUYÊN LOGIC CŨ CỦA MANDY)
        # ---------------------------------------------------------
        st.subheader("🔍 Detailed Distribution per Thickness Category")
        color_map = {'A-B+數': '#2ca02c', 'A-B-數': '#ff7f0e', 'B+數': '#d62728', 'B數': '#9467bd', 'A-B數': '#1f77b4'}

        def plot_feature_dist(ax, data, feat, thick, is_right_col=False):
            # Tính toán số bin dựa trên tổng số lượng cuộn thực tế
            N_t = data['Total_Count'].sum()
            k_b = max(int(1 + 3.322 * math.log10(N_t)) if N_t > 0 else 10, 5)
            
            for col_n in count_cols:
                temp_d = data[[feat, col_n]].dropna()
                temp_d = temp_d[temp_d[col_n] > 0]
                if len(temp_d) >= 1:
                    vals_d, wgts_d = temp_d[feat].values, temp_d[col_n].values
                    color = color_map.get(col_n, '#7f7f7f')
                    
                    sns.histplot(x=vals_d, weights=wgts_d, label=col_n.replace('數',''), color=color, bins=k_b, 
                                 stat='count', alpha=0.4, ax=ax, edgecolor='white')
                    
                    if len(vals_d) > 2:
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
            with st.expander(f"📏 Click to expand Analysis for Thickness: {thickness}"):
                cols_dist = st.columns(2)
                for idx, feat in enumerate(['YS', 'TS', 'EL', 'YPE']):
                    if feat in mech_features:
                        with cols_dist[idx % 2]:
                            fig, ax = plt.subplots(figsize=(10, 5))
                            plot_feature_dist(ax, df_thickness, feat, thickness, is_right_col=(idx % 2 != 0))
                            st.pyplot(fig)
                            fig.savefig(f"dist_{feat}_{thickness}.png", bbox_inches='tight')
    # --- TAB 3: OPTIMIZATION (Executive View - Based on A-B & Above) ---
    with tab3:
        st.header("3. Production Control Limits & Goals (A-B & Above Focused)")
        
        # Lựa chọn hệ số Sigma để tính dải an toàn (Mill Range)
        sigma_choice = st.radio("Select Sigma Factor for Mill Safety Zone", [2.0, 2.5, 3.0], index=0)

        # Thiết lập giới hạn kiểm soát hiện tại (Current Control Limit)
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
            
            # XÁC ĐỊNH NHÓM HÀNG TỐT (A-B TRỞ LÊN)
            good_grade_cols = [c for c in ['A-B+數', 'A-B數'] if c in df_t.columns]

            for feat in mech_features:
                # 1. Lọc dữ liệu: Chỉ lấy những hàng có đặc tính cơ lý VÀ có số lượng hàng tốt > 0
                if good_grade_cols:
                    temp_calc = df_t[[feat] + good_grade_cols].dropna(subset=[feat])
                    temp_calc['Total_Good_Qty'] = temp_calc[good_grade_cols].sum(axis=1)
                    temp_calc = temp_calc[temp_calc['Total_Good_Qty'] > 0]
                else:
                    temp_calc = pd.DataFrame()

                # Lấy chuỗi hiển thị giới hạn (ví dụ: 405-500)
                low, high = spec_limits.get(feat, (None, None))
                spec_str = f"{int(low)}–{int(high)}" if low and high else (f">={int(low)}" if low else "N/A")

                if not temp_calc.empty:
                    vals_raw = temp_calc[feat].values
                    wgts_raw = temp_calc['Total_Good_Qty'].values
                    
                    # 2. THUẬT TOÁN IQR (Gọt nhiễu để dải số hợp lý hơn)
                    q1, q3 = np.percentile(vals_raw, 25), np.percentile(vals_raw, 75)
                    iqr = q3 - q1
                    mask = (vals_raw >= q1 - 1.5*iqr) & (vals_raw <= q3 + 1.5*iqr)
                    
                    v_f = vals_raw[mask] if mask.sum() > 0 else vals_raw
                    w_f = wgts_raw[mask] if mask.sum() > 0 else wgts_raw
                    
                    # 3. TÍNH TOÁN CÁC CHỈ SỐ THỐNG KÊ (CÓ TRỌNG SỐ)
                    mean_val = np.average(v_f, weights=w_f)
                    std_val = np.sqrt(np.average((v_f - mean_val)**2, weights=w_f))
                    
                    plot_data_dict[thick][feat] = {'values': v_f, 'mean': mean_val, 'std': std_val}
                    
                    target_goal = int(round(mean_val))
                    release_range = f"{int(round(mean_val - 3*std_val))}–{int(round(mean_val + 3*std_val))}"
                    mill_range = f"{int(round(mean_val - sigma_choice*std_val))}–{int(round(mean_val + sigma_choice*std_val))}"
                    tolerance_val = int(round(sigma_choice * std_val))
                else:
                    target_goal, release_range, mill_range, tolerance_val = "N/A", "N/A", "N/A", "N/A"
                    mean_val, std_val = 0, 0 

                # Tính toán phân bổ phần trăm các cấp độ (Segment Distribution)
                seg_total = df_t[count_cols].sum().sum()
                seg_dist = ", ".join([f"{k.replace('數','')}:{int(round(df_t[k].sum()/seg_total*100))}%" for k in count_cols]) if seg_total > 0 else "N/A"

                # 4. TỔNG HỢP DỮ LIỆU DÒNG (ĐÃ BỎ STATUS)
                row_data = {
                    "Feature": feat,
                    "Current Control Limit": spec_str,
                    "Segment Distribution": seg_dist,
                    "Data-Driven Release Range": release_range,
                    "Target Goal": target_goal,
                    f"Tolerance (±{sigma_choice}σ)": tolerance_val, 
                    "Mill Range (Proposed)": mill_range
                }
                
                status_list.append(row_data)
                export_row = row_data.copy()
                export_row['Thickness'] = thick
                all_export_data.append(export_row)

            # Hiển thị bảng số liệu tối ưu
            if status_list:
                st.dataframe(pd.DataFrame(status_list), use_container_width=True, hide_index=True)

            # --- 5. VẼ BIỂU ĐỒ I-MR (DÀN TRANG 2x2 TRÊN WEB) ---
            top_4_feats = [f for f in ['YS', 'TS', 'EL', 'YPE'] if f in plot_data_dict[thick]]
            cols_imr = st.columns(2)
            
            for idx, feat in enumerate(top_4_feats):
                with cols_imr[idx % 2]:
                    d = plot_data_dict[thick][feat]
                    v, m_v, s_v = d['values'], d['mean'], d['std']
                    
                    if len(v) > 1:
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
                        U, L = m_v + sigma_choice*s_v, m_v - sigma_choice*s_v
                        
                        # Individuals Chart + Tô đỏ Outliers
                        ax1.plot(v, marker='o', color='blue', markersize=3, lw=1, zorder=1)
                        out_idx = np.where((v > U) | (v < L))[0]
                        ax1.scatter(out_idx, v[out_idx], color='red', s=30, zorder=2)
                        ax1.axhline(m_v, color='green', ls='--', label=f'Mean: {target_goal}')
                        ax1.axhline(U, color='red', ls='--', label=f'UCL: {int(round(U))}')
                        ax1.axhline(L, color='red', ls='--', label=f'LCL: {int(round(L))}')
                        ax1.set_title(f"I-Chart: {feat} (Thick: {thick})")
                        ax1.legend(loc='upper right', fontsize=7)
                        
                        # Moving Range Chart + Tô đỏ Outliers
                        MR = np.abs(np.diff(v))
                        MR_m = np.mean(MR)
                        MR_U = 3.267 * MR_m
                        ax2.plot(MR, marker='o', color='orange', markersize=3, lw=1, zorder=1)
                        mr_out = np.where(MR > MR_U)[0]
                        ax2.scatter(mr_out, MR[mr_out], color='red', s=30, zorder=2)
                        ax2.axhline(MR_m, color='green', ls='--')
                        ax2.axhline(MR_U, color='red', ls='--')
                        ax2.set_title("Moving Range Chart")
                        
                        fig.tight_layout()
                        st.pyplot(fig)
                        fig.savefig(f"imr_{feat}_{thick}.png", bbox_inches='tight')
            st.markdown("---")

        # Nút tải Excel (Dữ liệu đã gộp A-B trở lên)
        if all_export_data:
            st.markdown("### 📥 Download Final QC Report (Excel)")
            towrite = io.BytesIO()
            pd.DataFrame(all_export_data).to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button(label="📥 Download Excel Report", data=towrite, file_name="QC_Mill_Range_Report.xlsx")
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
            for i, v in enumerate(r):
                if isinstance(v, (int, float)) and v == int(v): v = int(v)
                pdf.cell(cw[i] if i < len(cw) else 20, 8, clean(v), border=1, align='C')
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
