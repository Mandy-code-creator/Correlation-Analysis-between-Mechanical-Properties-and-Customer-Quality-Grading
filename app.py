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

st.title("📊 QC Mechanical Properties Analysis - Full Comprehensive View")
st.markdown("---")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Read data
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip() 

    # --- 2. DATA PREPROCESSING ---
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    count_cols = [col for col in count_cols if col in df.columns]
    
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Mechanical properties
    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
    mech_features = [feat for feat in mech_features if feat in df.columns]
    
    for feat in mech_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
        # Filter out 0 or negative to avoid noise in distribution
        df.loc[df[feat] <= 0, feat] = np.nan

    df['Total_Count'] = df[count_cols].sum(axis=1)
    df = df[df['Total_Count'] > 0].copy()

    # Calculate Overall Quality Score for Correlation
    df['Quality_Score'] = (5*df.get('A+B+數', 0) + 4*df.get('A-B+數', 0) + 3*df.get('A-B數', 0) +
                           2*df.get('A-B-數', 0) + 1*df.get('B+數', 0)) / df['Total_Count']

    # --- 3. CREATE TABS ---
    tab1, tab2, tab3 = st.tabs([
        "1. Summary & Pie Charts", 
        "2. Correlation Analysis", 
        "3. Distribution Analysis (Sturges & Extended Tails)"
    ])

    # --- TAB 1: SUMMARY & PIE CHARTS ---
    with tab1:
        st.header("1. Summary by Thickness")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Total Coils'] = summary_df[count_cols].sum(axis=1)
        
        for col in count_cols:
            summary_df[f"% {col}"] = (summary_df[col]/summary_df['Total Coils']*100).round(2)
            
        col_t1, col_t2 = st.columns([1.5, 1])
        with col_t1:
            st.subheader("Data Summary Table")
            display_df = summary_df.copy()
            display_df.rename(columns={'厚度歸類': 'Thickness'}, inplace=True)
            st.dataframe(display_df, use_container_width=True)
            
            # Export simple summary to Excel
            towrite = io.BytesIO()
            display_df.to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button("📥 Download Summary Excel", data=towrite, file_name="QC_Summary.xlsx")

        with col_t2:
            st.subheader("Quality Proportion Pie Charts")
            plot_df = summary_df.set_index('厚度歸類')[count_cols]
            plot_df.columns = ['A+B+', 'A-B+', 'A-B', 'A-B-', 'B+']
            pie_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']

            n_pies = len(plot_df)
            if n_pies > 0:
                fig_pie, axes_pie = plt.subplots((n_pies+1)//2, min(2, n_pies), figsize=(14, 6 * ((n_pies+1)//2)))
                axes_pie_flat = np.array(axes_pie).flatten() if n_pies > 1 else [axes_pie]
                for i, thick in enumerate(plot_df.index):
                    ax_p = axes_pie_flat[i]
                    data_p = plot_df.loc[thick]
                    mask_p = data_p > 0
                    ax_p.pie(data_p[mask_p], autopct=lambda p: f'{p:.1f}%' if p>3 else '', startangle=90, 
                             colors=[c for c,m in zip(pie_colors, mask_p) if m], 
                             wedgeprops={'edgecolor':'white','linewidth':2}, 
                             textprops={'fontsize':12,'fontweight':'bold'})
                    ax_p.set_title(f"Thickness: {thick}", fontsize=16, fontweight='bold')
                for j in range(i+1, len(axes_pie_flat)): axes_pie_flat[j].axis('off')
                fig_pie.legend(plot_df.columns, title="Quality Grade", loc="center right")
                st.pyplot(fig_pie)

    # --- TAB 2: CORRELATION MATRIX ---
    with tab2:
        st.header("2. Correlation Analysis")
        corr_matrix = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        corr_matrix.columns = ['Quality Correlation Index']
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=0), use_container_width=True)
        st.info("💡 High positive values mean the property helps quality. Negative values mean it might hurt quality.")

    # --- TAB 3: DISTRIBUTION (CLEAR VIEW & EXTENDED TAILS) ---
    with tab3:
        st.header("3. Distribution Analysis (Sturges Rule & Extended Tails)")
        st.markdown("Normal curves now cover $\pm 4\sigma$ theoretical range. Labels are auto-positioned.")
        
        grade_mapping = {'A+B+': 'A+B+數', 'A-B+': 'A-B+數', 'A-B': 'A-B數', 'A-B-': 'A-B-數', 'B+': 'B+數'}
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)

        for feature in mech_features:
            st.markdown(f"### 📊 Distribution of **{feature}**")
            fig, axes = plt.subplots(len(thickness_list), 1, figsize=(16, 8 * len(thickness_list)))
            axes = axes if len(thickness_list) > 1 else [axes]

            for i, thick in enumerate(thickness_list):
                ax = axes[i]
                df_t = df[df['厚度歸類'] == thick]
                N_total = df_t['Total_Count'].sum()
                
                # Sturges Binning
                k_bins = int(1 + 3.322 * math.log10(N_total)) if N_total > 0 else 10
                k_bins = max(k_bins, 5)
                
                mean_info_list = []

                for (label, col_name), color in zip(grade_mapping.items(), colors):
                    temp_data = df_t[[feature, col_name]].dropna()
                    temp_data = temp_data[temp_data[col_name] > 0]
                    
                    if len(temp_data) > 2:
                        vals = temp_data[feature].values
                        wgts = temp_data[col_name].values
                        
                        # Histogram
                        sns.histplot(x=vals, weights=wgts, label=label, color=color, bins=k_bins, 
                                     stat='count', alpha=0.15, ax=ax, edgecolor='none')
                        
                        # Statistics
                        m_val = np.average(vals, weights=wgts)
                        s_val = np.sqrt(np.average((vals - m_val)**2, weights=wgts))
                        
                        # --- EXTENDED NORMAL CURVE TAILS ---
                        if s_val > 0:
                            # 4-sigma range for theoretical tail visualization
                            x_range = np.linspace(m_val - 4*s_val, m_val + 4*s_val, 200)
                            bin_w = (vals.max() - vals.min()) / k_bins if vals.max() != vals.min() else 1
                            ax.plot(x_range, stats.norm.pdf(x_range, m_val, s_val) * wgts.sum() * bin_w, 
                                    color=color, lw=3, alpha=0.85)
                        
                        # Mean Line
                        ax.axvline(m_val, color=color, ls='--', lw=2)
                        mean_info_list.append({'val': m_val, 'color': color})

                # --- ANTI-OVERLAP LABEL LOGIC ---
                if mean_info_list:
                    mean_info_list.sort(key=lambda x: x['val'])
                    y_max_limit = ax.get_ylim()[1]
                    
                    for idx, info in enumerate(mean_info_list):
                        # Alternating heights to avoid text overlap
                        y_pos = (0.94 if idx % 2 == 0 else 0.85) * y_max_limit
                        ax.text(info['val'], y_pos, f"{info['val']:.1f}", 
                                color=info['color'], fontsize=12, fontweight='bold',
                                ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

                ax.set_title(f"Thickness: {thick} (N={int(N_total)}, Sturges Bins={k_bins})", fontsize=18, fontweight='bold')
                ax.set_ylabel("Coil Count")
                ax.grid(axis='y', linestyle=':', alpha=0.6)
                
                # Legend placement outside
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), title="Quality Grade", 
                          bbox_to_anchor=(1.01, 1), loc='upper left')
                
            plt.tight_layout()
            st.pyplot(fig)

else:
    st.info("Please upload an Excel file to see all views.")
