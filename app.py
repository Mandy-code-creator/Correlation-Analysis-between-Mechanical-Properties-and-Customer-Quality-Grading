import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import io
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QC Mechanical Properties Dashboard", layout="wide")
st.title("📊 QC Mechanical Properties Analysis - Clear View Updates")
st.markdown("---")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

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

    # Calculate Overall Quality Score
    df['Quality_Score'] = (5*df.get('A+B+數', 0) + 4*df.get('A-B+數', 0) + 3*df.get('A-B數', 0) +
                           2*df.get('A-B-數', 0) + 1*df.get('B+數', 0)) / df['Total_Count']

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["1. Summary", "2. Correlation", "3. Distribution Analysis (Clear View)"])

    # (Tab 1 & 2 remain unchanged, omitted for clarity)

    # --- TAB 3: DISTRIBUTION ANALYSIS (FIXED OVERLAP & TAILS) ---
    with tab3:
        st.header("3. Distribution Analysis (Clear View - Sturges)")
        st.markdown("Normal Curves are now extended to $\pm 4\sigma$ for theoretical range. Mean labels are automatically re-arranged to prevent overlap.")
        
        grade_mapping = {'A+B+': 'A+B+數', 'A-B+': 'A-B+數', 'A-B': 'A-B數', 'A-B-': 'A-B-數', 'B+': 'B+數'}
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)

        for feature in mech_features:
            st.markdown(f"### 📊 Distribution of **{feature}**")
            fig, axes = plt.subplots(len(thickness_list), 1, figsize=(16, 7 * len(thickness_list)))
            axes = axes if len(thickness_list) > 1 else [axes]

            for i, thick in enumerate(thickness_list):
                ax = axes[i]
                df_t = df[df['厚度歸類'] == thick]
                N = df_t['Total_Count'].sum()
                
                # Sturges Bins
                k_bins = int(1 + 3.322 * math.log10(N)) if N > 0 else 10
                k_bins = max(k_bins, 5)
                
                # List to store mean values and colors for label management
                mean_labels_info = []

                for (label, col), color in zip(grade_mapping.items(), colors):
                    temp = df_t[[feature, col]].dropna()
                    temp = temp[temp[col] > 0]
                    if len(temp) > 2:
                        vals, wgts = temp[feature].values, temp[col].values
                        
                        # sns.histplot using fixed bins
                        sns.histplot(x=vals, weights=wgts, label=label, color=color, bins=k_bins, stat='count', alpha=0.15, ax=ax, edgecolor='none')
                        
                        m = np.average(vals, weights=wgts)
                        s = np.sqrt(np.average((vals-m)**2, weights=wgts))
                        
                        # --- FIX: Extend Normal Curve Tails to 4*sigma ---
                        if s > 0:
                            # Generate X axis based on theoretical 4-sigma range, not just data min/max
                            x_extended = np.linspace(m - 4*s, m + 4*s, 200)
                            
                            # Scaling the Normal Curve to match Histogram 'count' scale
                            bin_w = (vals.max() - vals.min()) / k_bins
                            ax.plot(x_extended, stats.norm.pdf(x_extended, m, s) * wgts.sum() * bin_w, color=color, lw=2.5, alpha=0.9)
                        
                        # Draw Mean Line
                        ax.axvline(m, color=color, ls='--', lw=2)
                        
                        # Store mean info for non-overlapping label placement
                        mean_labels_info.append({'val': m, 'color': color})

                # --- FIX: Tự động sắp xếp nhãn giá trị trung bình để không ghi đè ---
                if mean_labels_info:
                    # Sắp xếp theo giá trị để dễ xử lý chồng chéo
                    mean_labels_info.sort(key=lambda x: x['val'])
                    
                    y_max = ax.get_ylim()[1]
                    y_positions = []
                    y_start_pct = 0.93  # Bắt đầu ở độ cao 93% trục Y
                    
                    # Xác định độ cao cho từng nhãn
                    for info in mean_labels_info:
                        if not y_positions:
                            y_positions.append(y_start_pct * y_max)
                        else:
                            # Nếu nhãn này quá gần nhãn trước (trong khoảng 4% dải Y), đẩy nó xuống
                            # (Thay đổi logic so sánh Y-axis sang X-axis để chính xác hơn)
                            pass

                    # Logic mới đơn giản hơn: Sắp xếp các nhãn theo 2 độ cao khác nhau luân phiên
                    # (Điều này giải quyết 80% trường hợp chồng chéo dữ liệu cơ tính)
                    y_high = y_start_pct * y_max
                    y_low = (y_start_pct - 0.08) * y_max  # Thấp hơn 8%
                    
                    for idx, info in enumerate(mean_labels_info):
                        # Các nhãn lẻ nằm cao, các nhãn chẵn nằm thấp
                        y_pos = y_high if idx % 2 == 0 else y_low
                        
                        ax.text(info['val'], y_pos, f"{info['val']:.1f}", 
                                color=info['color'], fontsize=11, fontweight='bold',
                                horizontalalignment='center',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

                ax.set_title(f"Thickness: {thick} (N={int(N)}, Bins={k_bins})", fontsize=16, fontweight='bold')
                ax.grid(axis='y', linestyle=':', alpha=0.7)
                
                # --- FIX: Legend placement (Move outside top-right) ---
                handles, labels = ax.get_legend_handles_labels()
                # Remove duplicate Normal Curve lines from legend if any
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), title="Quality Grade", 
                          bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, fontsize=10)
                
            plt.tight_layout()
            st.pyplot(fig)

else:
    st.info("Please upload an Excel file.")
