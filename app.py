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

    # --- TAB 2: DISTRIBUTION (PARALLEL VIEW) ---
    with tab2:
        st.header("2. Distribution Analysis (Parallel Clear View)")
        grade_mapping = {'A+B+': 'A+B+數', 'A-B+': 'A-B+數', 'A-B': 'A-B數', 'A-B-': 'A-B-數', 'B+': 'B+數'}
        # Đã setup màu sắc tương phản rõ ràng
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
                    
                    # CẢI TIẾN MÀU SẮC BIỂU ĐỒ CỘT Ở ĐÂY (Tăng alpha lên 0.45 và thêm viền trắng)
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

# --- TAB 3: OPTIMIZATION (Data-Driven Executive View) ---
    with tab3:
        st.header("3. Production Control Limits & Goals (A-B & Above Focused)")
        
        # 1. Khai báo biến sigma_choice
        sigma_choice = st.radio("Select Sigma Factor for Mill Safety Zone", [2.0, 2.5, 3.0], index=0)

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
                if len(temp_calc) == 0: 
                    continue

                temp_calc['Good_Count'] = temp_calc[good_grades].sum(axis=1)
                temp_calc_good = temp_calc[temp_calc['Good_Count'] > 0]

                low, high = spec_limits.get(feat, (None, None))
                if low is not None and high is not None:
                    spec_str = f"{int(low)}–{int(high)}"
                elif low is not None:
                    spec_str = f">={int(low)}"
                else:
                    spec_str = "N/A"

                # --- CALCULATE DATA-DRIVEN LIMITS ---
                if not temp_calc_good.empty:
                    vals_good = temp_calc_good[feat].values
                    wgts_good = temp_calc_good['Good_Count'].values
                    
                    # Bước 1: Tính toán sơ bộ (để tìm ngưỡng lọc nhiễu)
                    m_raw = np.average(vals_good, weights=wgts_good)
                    s_raw = np.sqrt(np.average((vals_good - m_raw)**2, weights=wgts_good))
                    
                    # Bước 2: Lọc nhiễu Outliers (3-Sigma)
                    mask = (vals_good >= m_raw - 3*s_raw) & (vals_good <= m_raw + 3*s_raw)
                    
                    if mask.sum() > 0:
                        vals_final = vals_good[mask]
                        wgts_final = wgts_good[mask]
                        # Tính toán các giá trị chính thức từ tập dữ liệu "sạch"
                        mean_val = np.average(vals_final, weights=wgts_final)
                        std_val = np.sqrt(np.average((vals_final - mean_val)**2, weights=wgts_final))
                    else:
                        mean_val, std_val = m_raw, s_raw
                        vals_final = vals_good
                    
                    # QUAN TRỌNG: Lưu thành dạng Dictionary để bên dưới gọi d['values'] không bị lỗi
                    plot_data_dict[thick][feat] = {
                        'values': vals_final,
                        'mean': mean_val,
                        'std': std_val
                    }
                    
                    target_goal = int(round(mean_val))
                    
                    # 2. INTERNAL RELEASE RANGE (Dải 3-sigma sạch)
                    rel_low = mean_val - 3 * std_val
                    rel_high = mean_val + 3 * std_val
                    release_range = f"{int(round(rel_low))}–{int(round(rel_high))}"
                    
                    # 3. MILL RANGE (PROPOSED) 
                    mill_low = mean_val - sigma_choice * std_val
                    mill_high = mean_val + sigma_choice * std_val
                    mill_range = f"{int(round(mill_low))}–{int(round(mill_high))}"

                    # 4. TOLERANCE
                    tolerance_val = int(round(sigma_choice * std_val))
                else:
                    target_goal, release_range, mill_range, tolerance_val = "N/A", "N/A", "N/A", "N/A"
                    mean_val, std_val = 0, 0  # Bổ sung std_val = 0 để tránh lỗi UnboundLocalError

                # --- SEGMENT DISTRIBUTION ---
                seg_total = df_t[count_cols].sum().sum()
                if seg_total > 0:
                    seg_dist = ", ".join([f"{k.replace('數','')}: {int(round(df_t[k].sum()/seg_total*100))}%" for k in count_cols])
                else:
                    seg_dist = "N/A"

                # --- ROW DATA ---
                row_data = {
                    "Feature": feat,
                    "Customer Spec Limit": spec_str,
                    "Segment Distribution": seg_dist,
                    "Data-Driven Release Range": release_range,
                    "Target Goal": target_goal,
                    f"Tolerance (±{sigma_choice}σ)": tolerance_val, 
                    "Mill Range (Proposed)": mill_range,
                    "Status": "✅ Safe" if (low is None or (mean_val - sigma_choice*std_val) >= low) else "⚠ Risk"
                }
                
                status_list.append(row_data)
                
                # Thêm vào danh sách xuất Excel
                export_row = row_data.copy()
                export_row['Thickness'] = thick
                all_export_data.append(export_row)

            # Hiển thị bảng sau khi kết thúc vòng lặp feat cho từng độ dày
            if status_list:
                st.dataframe(pd.DataFrame(status_list), use_container_width=True, hide_index=True)

            # --- VẼ BIỂU ĐỒ I-MR ---
            for feat in mech_features:
                if feat in plot_data_dict[thick]:
                    d = plot_data_dict[thick][feat]
                    
                    # CHỐT CHẶN AN TOÀN: Kiểm tra d có phải Dictionary không
                    if isinstance(d, dict) and 'values' in d:
                        v = d['values']
                        m_v = d['mean']
                        s_v = d['std']
                        
                        if len(v) > 1:
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
                            U, L = m_v + sigma_choice*s_v, m_v - sigma_choice*s_v
                            
                            # Individuals Chart
                            ax1.plot(v, marker='o', color='blue', markersize=4)
                            ax1.axhline(m_v, color='green', ls='--', label=f'Mean: {int(round(m_v))}')
                            ax1.axhline(U, color='red', ls='--', label=f'UCL: {int(round(U))}')
                            ax1.axhline(L, color='red', ls='--', label=f'LCL: {int(round(L))}')
                            ax1.set_title(f"Individuals Chart: {feat} (Unified Data) - Thick {thick}")
                            ax1.legend(loc='upper right', fontsize=8)
                            
                            # Moving Range Chart
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
# =========================================================================
    # --- XUẤT BÁO CÁO PDF (EXECUTIVE PDF REPORT) ---
    # =========================================================================
    st.markdown("---")
    st.header("🖨️ Export PDF Executive Report")

    # Hàm lọc ký tự Unicode (Giúp FPDF không bị lỗi khi gặp icon hoặc tiếng Trung)
    def clean_text(text):
        text = str(text)
        text = text.replace('✅ Safe', 'SAFE').replace('⚠ Risk', 'RISK')
        text = text.replace('±', '+/-').replace('–', '-')
        # Bỏ qua các ký tự tiếng Trung/Unicode không tương thích để tránh sập app
        return text.encode('latin-1', 'ignore').decode('latin-1')

    if st.button("Generate & Download PDF Report"):
        with st.spinner("Đang tổng hợp dữ liệu và kết xuất PDF..."):
            
            # Khởi tạo bản in khổ Ngang (Landscape) để chứa đủ các cột dữ liệu
            pdf = FPDF(orientation='L') 
            
            # ---------------------------------------------------------
            # PHẦN 1: QUALITY SUMMARY BY THICKNESS
            # ---------------------------------------------------------
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, "QC MECHANICAL PROPERTIES & YIELD REPORT", ln=True, align="C")
            pdf.ln(5)

            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "1. Quality Summary by Thickness", ln=True)
            
            # Thiết lập độ rộng 14 cột cho bảng Summary (Khớp với màn hình dọc)
            pdf.set_font('Arial', 'B', 8)
            col_widths_1 = [10, 20] + [15]*5 + [20] + [15]*5 
            
            # In Header
            for i, col in enumerate(display_df.columns):
                pdf.cell(col_widths_1[i], 8, clean_text(col), border=1, align='C')
            pdf.ln()
            
            # In dữ liệu từng dòng
            pdf.set_font('Arial', '', 8)
            for _, row in display_df.iterrows():
                for i, val in enumerate(row):
                    pdf.cell(col_widths_1[i], 8, clean_text(val), border=1, align='C')
                pdf.ln()

            # ---------------------------------------------------------
            # PHẦN 2: DISTRIBUTION ANALYSIS CHARTS
            # ---------------------------------------------------------
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "2. Distribution Analysis (Parallel Clear View)", ln=True)
            
            for thick in thickness_list:
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 10, f"Thickness Category: {thick}", ln=True)
                
                # Căn tọa độ để xếp 2 biểu đồ / 1 hàng ngang
                x_start = 10
                y_start = pdf.get_y()
                x_offset = x_start
                
                for feat in mech_features:
                    img_path = f"dist_{feat}_{thick}.png"
                    if os.path.exists(img_path):
                        pdf.image(img_path, x=x_offset, y=y_start, w=130)
                        x_offset += 135
                        if x_offset > 150: # Đã vẽ xong 2 hình, xuống hàng
                            y_start += 70  # Dời tọa độ Y xuống
                            x_offset = x_start
                            # Nếu sắp hết trang thì qua trang mới
                            if y_start > 140: 
                                pdf.add_page()
                                y_start = pdf.get_y()
                pdf.add_page()

            # ---------------------------------------------------------
            # PHẦN 3: PRODUCTION CONTROL LIMITS & I-MR CHARTS
            # ---------------------------------------------------------
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "3. Production Control Limits & Goals (A-B & Above Focused)", ln=True)

            for thick in thickness_list:
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 10, f"Thickness: {thick}", ln=True)
                
                # Vẽ bảng Status
                pdf.set_font('Arial', 'B', 8)
                headers = ["Feature", "Standard", "Segment Distribution", "Release Range", "Target", f"Tol(+/-{sigma_choice})", "Mill Range", "Status"]
                col_widths_3 = [15, 25, 75, 35, 15, 20, 35, 20] # Tổng = 240 (Vừa vặn trang ngang)
                
                for i, head in enumerate(headers):
                    pdf.cell(col_widths_3[i], 8, head, border=1, align='C')
                pdf.ln()
                
                pdf.set_font('Arial', '', 8)
                for row in all_export_data:
                    if row['Thickness'] == thick:
                        pdf.cell(col_widths_3[0], 8, clean_text(row.get("Feature", "")), border=1, align='C')
                        pdf.cell(col_widths_3[1], 8, clean_text(row.get("Internal Standard", "")), border=1, align='C')
                        pdf.cell(col_widths_3[2], 8, clean_text(row.get("Segment Distribution", "")), border=1, align='C')
                        pdf.cell(col_widths_3[3], 8, clean_text(row.get("Data-Driven Release Range", "")), border=1, align='C')
                        pdf.cell(col_widths_3[4], 8, clean_text(row.get("Target Goal", "")), border=1, align='C')
                        tol_key = f"Tolerance (±{sigma_choice}σ)"
                        pdf.cell(col_widths_3[5], 8, clean_text(row.get(tol_key, "")), border=1, align='C')
                        pdf.cell(col_widths_3[6], 8, clean_text(row.get("Mill Range (Proposed)", "")), border=1, align='C')
                        pdf.cell(col_widths_3[7], 8, clean_text(row.get("Status", "")), border=1, align='C')
                        pdf.ln()
                
                pdf.ln(5)
                # Chèn ảnh biểu đồ I-MR
                for feat in mech_features:
                    img_path = f"imr_{feat}_{thick}.png"
                    if os.path.exists(img_path):
                        # Biểu đồ I-MR lớn nên để dàn trải hết bề ngang
                        pdf.image(img_path, w=260)
                        pdf.add_page()

            # ---------------------------------------------------------
            # LƯU VÀ TẢI PDF
            # ---------------------------------------------------------
            pdf.output("QC_Report_Temp.pdf")
            with open("QC_Report_Temp.pdf", "rb") as f:
                pdf_bytes = f.read()
                
            st.download_button(
                label="📥 Bấm vào đây để tải PDF Report",
                data=pdf_bytes,
                file_name="QC_Executive_Report.pdf",
                mime="application/pdf"
            )
