import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# --- Page Config ---
st.set_page_config(page_title="QC Mechanical Properties Dashboard", layout="wide")
st.title("📊 QC Mechanical Properties Analysis")
st.markdown("---")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()  # remove extra spaces

    # --- Data Preprocessing ---
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']

    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    for feat in mech_features:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors='coerce')
            df.loc[df[feat] <= 0, feat] = np.nan

    df['Total_Count'] = df[count_cols].sum(axis=1)
    df = df[df['Total_Count'] > 0].copy()

    # --- Quality Score ---
    df['Quality_Score'] = (5*df['A+B+數'] + 4*df['A-B+數'] + 3*df['A-B數'] +
                           2*df['A-B-數'] + 1*df['B+數']) / df['Total_Count']

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Summary & Percentages",
        "2. Correlation Matrix",
        "3. Weighted Distribution",
        "4. Optimal & Safe Limits"
    ])

    # --- Tab 1: Summary ---
    with tab1:
        st.header("1. Summary by Thickness")
        summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
        summary_df['Total Coils'] = summary_df[count_cols].sum(axis=1)
        for col in count_cols:
            summary_df[f"% {col}"] = (summary_df[col]/summary_df['Total Coils']*100).round(2)
        st.dataframe(summary_df, use_container_width=True)

    # --- Tab 2: Correlation ---
    with tab2:
        st.header("2. Correlation Matrix")
        corr_matrix = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        corr_matrix.columns = ['Correlation with Quality Score']
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=0), use_container_width=True)

    # --- Tab 4: Optimal & Safe Limits ---
    with tab4:
        st.header("4. Optimal & Safe Limits")
        limits = []
        for feat in mech_features:
            # --- Optimal Limit by percentile ---
            all_vals = []
            for grade_col in count_cols:
                temp = df[[feat, grade_col]].dropna()
                temp = temp[temp[grade_col]>0]
                if len(temp)>0:
                    vals = temp[feat].values
                    wgts = temp[grade_col].values
                    all_vals.extend(np.repeat(vals, wgts.astype(int)))
            if len(all_vals) > 10:
                lower_opt = np.percentile(all_vals, 2.5)
                upper_opt = np.percentile(all_vals, 97.5)
            else:
                lower_opt, upper_opt = np.nan, np.nan

            # --- Safe Limit from weighted mean/std of good grades only ---
            all_good_vals = []
            for grade_col in ['A+B+數','A-B+數']:
                temp = df[[feat, grade_col]].dropna()
                temp = temp[temp[grade_col]>0]
                if len(temp)>0:
                    vals = temp[feat].values
                    wgts = temp[grade_col].values
                    all_good_vals.extend(np.repeat(vals, wgts.astype(int)))
            if len(all_good_vals) > 5:
                w_mean = np.average(all_good_vals)
                w_std = np.std(all_good_vals)
                lower_safe = w_mean - 2*w_std
                upper_safe = w_mean + 2*w_std
            else:
                lower_safe, upper_safe = np.nan, np.nan

            limits.append([feat, lower_opt, upper_opt, lower_safe, upper_safe])

        limits_df = pd.DataFrame(limits, columns=['Parameter',
                                                  'Optimal Lower','Optimal Upper',
                                                  'Safe Lower','Safe Upper'])
        st.dataframe(limits_df, use_container_width=True)

    # --- Tab 3: Weighted Distribution with limits ---
    with tab3:
        st.header("3. Weighted Distribution by Thickness with Limits")
        grade_mapping = {
            'A+B+ (Excellent)': 'A+B+數',
            'A-B+ (Good)': 'A-B+數',
            'A-B (Average)': 'A-B數',
            'A-B- (Poor)': 'A-B-數',
            'B+ (Reject)': 'B+數'
        }
        colors = ['#2ca02c','#1f77b4','#ff7f0e','#9467bd','#d62728']
        thickness_list = sorted(df['厚度歸類'].dropna().unique(), key=str)

        # Dictionary of limits
        limits_dict = {row['Parameter']:(row['Optimal Lower'],row['Optimal Upper'],row['Safe Lower'],row['Safe Upper'])
                       for idx,row in limits_df.iterrows()}

        for feature in mech_features:
            with st.expander(f"📊 Distribution: {feature}", expanded=True):
                fig, axes = plt.subplots(len(thickness_list),1,figsize=(16,6*len(thickness_list)))
                axes = axes if len(thickness_list)>1 else [axes]

                for i, thickness in enumerate(thickness_list):
                    ax = axes[i]
                    df_thick = df[df['厚度歸類']==thickness]
                    has_data = False

                    for (grade_label, grade_col), color in zip(grade_mapping.items(), colors):
                        if grade_col not in df_thick.columns:
                            continue
                        temp_df = df_thick[[feature, grade_col]].dropna()
                        temp_df = temp_df[temp_df[grade_col]>0]
                        if len(temp_df) > 3:
                            has_data = True
                            values = temp_df[feature].values
                            weights = temp_df[grade_col].values
                            total_weight = weights.sum()
                            sns.histplot(temp_df, x=feature, weights=grade_col, label=grade_label,
                                         color=color, bins=20, kde=False, stat='count', alpha=0.3, ax=ax)

                            # Weighted distribution line
                            weighted_mean = np.average(values, weights=weights)
                            weighted_std = np.sqrt(np.average((values-weighted_mean)**2, weights=weights))
                            if weighted_std>0:
                                x_axis = np.linspace(values.min(), values.max(),150)
                                pdf = stats.norm.pdf(x_axis, weighted_mean, weighted_std)
                                scaled_pdf = pdf * total_weight * ((values.max()-values.min())/20)
                                ax.plot(x_axis, scaled_pdf, color=color, linewidth=2, alpha=0.8)
                            ax.axvline(weighted_mean, color=color, linestyle='--', linewidth=1, alpha=0.8)

                    # Draw limits
                    if feature in limits_dict:
                        opt_lower,opt_upper,safe_lower,safe_upper = limits_dict[feature]
                        if not np.isnan(opt_lower):
                            ax.axvline(opt_lower,color='blue',linestyle='-.',linewidth=2,label='Optimal Lower')
                        if not np.isnan(opt_upper):
                            ax.axvline(opt_upper,color='blue',linestyle='-.',linewidth=2,label='Optimal Upper')
                        if not np.isnan(safe_lower):
                            ax.axvline(safe_lower,color='green',linestyle='-',linewidth=2,label='Safe Lower')
                        if not np.isnan(safe_upper):
                            ax.axvline(safe_upper,color='green',linestyle='-',linewidth=2,label='Safe Upper')

                    if not has_data:
                        ax.set_title(f"Thickness: {thickness} - Not enough data", color='gray')
                        ax.axis('off')
                    else:
                        ax.set_title(f"Thickness: {thickness}")
                        ax.set_xlabel(feature)
                        ax.set_ylabel("Number of Coils")
                        ax.grid(axis='y', linestyle=':', alpha=0.7)
                        ax.legend(title="Grade / Limit")

                plt.tight_layout()
                st.pyplot(fig,use_container_width=True)

else:
    st.info("Please upload an Excel file to start analysis.")
