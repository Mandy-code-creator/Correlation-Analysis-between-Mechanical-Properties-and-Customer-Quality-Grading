import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# ================= PAGE CONFIG =================
st.set_page_config(page_title="QC Mechanical Analysis Dashboard", layout="wide")
st.title("📊 QC Dashboard: Mechanical Properties vs Quality Grading")
st.markdown("---")

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("Upload your Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    # ================= DATA PREPROCESSING =================
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']

    for col in count_cols + mech_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for feat in mech_features:
        df.loc[df[feat] <= 0, feat] = np.nan

    df['Total_Count'] = df[count_cols].sum(axis=1)
    df = df[df['Total_Count'] > 0].copy()

    # ================= QUALITY SCORE =================
    df['Quality_Score'] = (5*df['A+B+數'] + 4*df['A-B+數'] + 3*df['A-B數'] +
                           2*df['A-B-數'] + 1*df['B+數']) / df['Total_Count']

    # ================= SPEC LIMITS =================
    spec_limits = {
        'YS': (405, 500),
        'TS': (415, 550),
        'EL': (25, np.nan),
        'YPE': (4, np.nan)
    }

    # ================= OPTIMAL LIMITS =================
    optimal_limits = {}
    df_good = df.copy()
    df_good['Good_Count'] = df_good['A+B+數'].fillna(0) + df_good['A-B+數'].fillna(0)
    df_good = df_good[df_good['Good_Count'] > 0]

    for feat in mech_features:
        if feat in df.columns:
            weighted_vals = []
            temp_df = df_good[[feat,'Good_Count']].dropna()
            for idx, row in temp_df.iterrows():
                val = row[feat]
                w = row['Good_Count']
                try:
                    w_int = int(round(w))
                    if w_int>0:
                        weighted_vals.extend([val]*w_int)
                except:
                    continue
            if len(weighted_vals)>0:
                lower = np.percentile(weighted_vals,2.5)
                upper = np.percentile(weighted_vals,97.5)
                if feat in spec_limits:
                    min_spec,max_spec = spec_limits[feat]
                    lower = max(lower,min_spec) if not np.isnan(min_spec) else lower
                    upper = min(upper,max_spec) if not np.isnan(max_spec) else upper
                optimal_limits[feat] = (round(lower,2),round(upper,2))
            else:
                optimal_limits[feat] = (None,None)

    # ================= TABS =================
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Summary & Percentages",
        "2. Correlation Matrix",
        "3. Full Distribution",
        "4. Optimal Mechanical Limits"
    ])

    # --- TAB 1: SUMMARY ---
    with tab1:
        st.header("1. Summary of Coil Grades by Thickness")
        if '厚度歸類' in df.columns:
            summary_df = df.groupby('厚度歸類')[count_cols].sum().reset_index()
            summary_df['Total Coils'] = summary_df[count_cols].sum(axis=1)
            for col in count_cols:
                summary_df[f"% {col}"] = (summary_df[col]/summary_df['Total Coils']*100).round(2)
            
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("Column '厚度歸類' not found for grouping.")

    # --- TAB 2: CORRELATION ---
    with tab2:
        st.header("2. Correlation with Quality Score")
        corr_df = df[['Quality_Score'] + mech_features].corr()[['Quality_Score']].drop('Quality_Score')
        corr_df.columns = ['Correlation with Quality Score']
        st.dataframe(corr_df.style.background_gradient(cmap='coolwarm'), use_container_width=True)

    # --- TAB 3: FULL DISTRIBUTION ---
    with tab3:
        st.header("3. Full Distribution of Mechanical Properties")
        for feat in mech_features:
            st.markdown(f"### {feat}")
            weighted_vals = []
            for idx,row in df_good.iterrows():
                val = row[feat]
                w = row['Good_Count']
                if pd.notna(val) and w>0:
                    weighted_vals.extend([val]*int(round(w)))
            if len(weighted_vals)==0:
                st.info(f"No valid data for {feat}")
                continue
            fig,ax = plt.subplots(figsize=(12,5))
            sns.histplot(weighted_vals,bins=20,kde=True,ax=ax,color="#1f77b4",alpha=0.5)
            st.pyplot(fig,use_container_width=True)

    # --- TAB 4: OPTIMAL LIMITS ---
    with tab4:
        st.header("4. Optimal Mechanical Limits Based on Good Coil %")
        table_data=[]
        for feat in mech_features:
            lower,upper = optimal_limits.get(feat,(None,None))
            min_spec,max_spec = spec_limits.get(feat,(None,None))
            table_data.append({
                'Feature': feat,
                'Optimal Lower Limit': lower,
                'Optimal Upper Limit': upper,
                'Current Spec Min': min_spec,
                'Current Spec Max': max_spec
            })
        table_df = pd.DataFrame(table_data)
        st.dataframe(table_df,use_container_width=True)
        
        # Plot with limits
        for feat in mech_features:
            st.markdown(f"### {feat} Distribution with Limits")
            weighted_vals = []
            for idx,row in df_good.iterrows():
                val=row[feat]
                w=row['Good_Count']
                if pd.notna(val) and w>0:
                    weighted_vals.extend([val]*int(round(w)))
            if len(weighted_vals)==0:
                st.info(f"No valid data for {feat}")
                continue
            fig,ax=plt.subplots(figsize=(12,5))
            sns.histplot(weighted_vals,bins=20,kde=True,ax=ax,color="#1f77b4",alpha=0.5)
            lower,upper = optimal_limits[feat]
            min_spec,max_spec = spec_limits.get(feat,(None,None))
            if lower is not None: ax.axvline(lower,color='green',linestyle='--',label='Optimal Lower',linewidth=2)
            if upper is not None: ax.axvline(upper,color='green',linestyle='-',label='Optimal Upper',linewidth=2)
            if min_spec is not None: ax.axvline(min_spec,color='red',linestyle='--',label='Current Spec Min',linewidth=2)
            if max_spec is not None: ax.axvline(max_spec,color='red',linestyle='-',label='Current Spec Max',linewidth=2)
            ax.set_xlabel(feat)
            ax.set_ylabel("Weighted Count of Good Coils")
            ax.legend()
            st.pyplot(fig,use_container_width=True)

else:
    st.info("Please upload your Excel file (.xlsx) to start analysis.")
