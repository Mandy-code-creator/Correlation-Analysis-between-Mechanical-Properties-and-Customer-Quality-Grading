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
st.title("📊 Quality Grade & Mechanical Properties Analysis")
st.markdown("---")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()  # remove extra spaces

    # Store figures for Excel export
    exported_figures = {}

    # --- 2. DATA PREPROCESSING ---
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    count_cols = [col for col in count_cols if col in df.columns]
    
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']
    mech_features = [feat for feat in mech_features if feat in df.columns]
    
    for feat in mech_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
        # Clean data: remove 0 or negative values to avoid noise
        df.loc[df[feat] <= 0, feat] = np.nan

    df['Total_Count'] = df[count_cols].sum(axis=1)
    df = df[df['Total_Count'] > 0].copy()

    # Calculate Overall Quality Score
    df['Quality_Score'] = (5*df.get('A+B+數', 0) + 4*df.get('A-B+數', 0) + 3*df.get('A-B數', 0) +
                           2*df.get('A-B-數', 0) + 1*df.get('B+數', 0)) / df['Total_Count']

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Summary & Pie Charts",
        "2. Correlation Matrix",
        "3. Distribution Analysis",
        "4. Optimal Target Limits"
    ])

    # --- TAB 1: SUMMARY ---
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

        with col_t2:
            st.subheader("Quality Proportions")
            plot_df = summary_df.set_index('厚度歸類')[count_cols]
            plot_df.columns = ['A+B+', 'A-B+', 'A-B', 'A-B-', 'B+']
            pie_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']

            n_pies = len(plot_df)
            if n_pies > 0:
                fig1, axes1 = plt.subplots((n_pies+1)//2, min(2, n_pies), figsize=(14, 6 * ((n_pies+1)//2)))
                axes_flat = np.array(axes1).flatten() if n_pies > 1 else [axes1]
                for i, thick in enumerate(plot_df.index):
                    ax = axes_flat[i]
                    data = plot_df.loc[thick]
                    mask = data > 0
                    ax.pie(data[mask], autopct=lambda p: f'{p:.1f}%' if p>3 else '', startangle=90, 
                           colors=[c for c,m in zip(pie_colors, mask) if m], 
                           wedgeprops={'edgecolor':'white','linewidth':2}, 
                           textprops={'fontsize':12,'fontweight':'bold'})
                    ax.set_title(f"Thickness: {thick}", fontsize=16, fontweight='bold')
                for j in range(i+1, len(axes_flat)): axes_flat[j].axis('off')
                fig1.legend(plot_df.columns, title="Quality Grade", loc="center right")
                st.pyplot(fig1)
                exported_figures['Quality_Pie'] = fig1

    # --- TAB 2: CORRELATION ---
    with tab2:
        st.header("2. Correlation Analysis")
        corr_matrix = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        corr_matrix.columns = ['Quality Correlation Index']
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=0), use_container_width=True)

    # --- TAB 3: DISTRIBUTION (STURGES RULE & MEAN LABELS) ---
    with tab3:
        st.header("3. Distribution Analysis (Sturges Rule)")
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
                
                # Sturges Rule for Bins
                k_bins = int(1 + 3.322 * math.log10(N)) if N > 0 else 10
                k_bins = max(k_bins, 5)

                for (label, col), color in zip(grade_mapping.items(), colors):
                    temp = df_t[[feature, col]].dropna()
                    temp = temp[temp[col] > 0]
                    if len(temp) > 2:
                        vals, wgts = temp[feature].values, temp[col].values
                        sns.histplot(x=vals, weights=wgts, label=label, color=color, bins=k_bins, stat='count', alpha=0.2, ax=ax)
                        
                        m = np.average(vals, weights=wgts)
                        s = np.sqrt(np.average((vals-m)**2, weights=wgts))
                        
                        # Normal Curve
                        if s > 0:
                            x = np.linspace(vals.min(), vals.max(), 100)
                            bin_w = (vals.max() - vals.min()) / k_bins
                            ax.plot(x, stats.norm.pdf(x, m, s) * wgts.sum() * bin_w, color=color, lw=3)
                        
                        # Mean Line & Text Label
                        ax.axvline(m, color=color, ls='--', lw=2)
                        ax.text(m, ax.get_ylim()[1] * 0.9, f'{m:.1f}', color=color, fontsize=11, fontweight='bold',
                                horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

                ax.set_title(f"Thickness: {thick} (N={int(N)}, Bins={k_bins})", fontsize=16, fontweight='bold')
                ax.legend(title="Quality Grade")
            st.pyplot(fig)
            exported_figures[f'{feature}_Dist'] = fig

    # --- TAB 4: OPTIMAL LIMITS ---
    with tab4:
        st.header("4. Target Optimization (90% Coverage)")
        opt_results = []
        for feat in mech_features:
            all_good_vals = []
            for col in ['A+B+數', 'A-B+數', 'A-B數']:
                if col in df.columns:
                    temp = df[[feat, col]].dropna()
                    temp = temp[temp[col] > 0]
                    all_good_vals.extend(np.repeat(temp[feat].values, temp[col].values.astype(int)))
            
            if len(all_good_vals) > 10:
                arr = np.array(all_good_vals)
                l_limit, u_limit = np.percentile(arr, 5), np.percentile(arr, 95)
                opt_results.append({'Parameter': feat, 'Optimal Lower': round(l_limit, 2), 'Optimal Upper': round(u_limit, 2)})
                st.success(f"🎯 **{feat} Target Window:** {l_limit:.2f} ~ {u_limit:.2f}")

        if opt_results:
            st.table(pd.DataFrame(opt_results))

    # --- EXCEL EXPORT ---
    st.markdown("---")
    if st.button("📥 Generate Full Excel Report"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            if opt_results: pd.DataFrame(opt_results).to_excel(writer, sheet_name='Target_Limits', index=False)
            
            workbook = writer.book
            sheet = workbook.add_worksheet('Charts')
            row = 0
            for name, fig in exported_figures.items():
                imgdata = io.BytesIO()
                fig.savefig(imgdata, format='png', bbox_inches='tight')
                sheet.insert_image(row, 0, name, {'image_data': imgdata, 'x_scale': 0.5, 'y_scale': 0.5})
                row += 40
        st.download_button("Download Report", output.getvalue(), "QC_Full_Analysis.xlsx")

else:
    st.info("Please upload an Excel file to start.")
