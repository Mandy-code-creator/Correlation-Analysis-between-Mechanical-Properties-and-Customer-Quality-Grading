import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Optimal Mechanical Limits Dashboard", layout="wide")
st.title("📊 QC Dashboard: Optimal Mechanical Limits from Grade Distribution")
st.markdown("---")

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader(
    "Upload your Excel file (.xlsx) with coil data",
    type=["xlsx"]
)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    # ================= DATA PREPROCESSING =================
    count_cols = ['A+B+數', 'A-B+數', 'A-B數', 'A-B-數', 'B+數']
    mech_features = ['YS', 'TS', 'EL', 'YPE', 'HARDNESS']

    # Convert numeric
    for col in count_cols + mech_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove zero or negative values for mech features
    for feat in mech_features:
        df.loc[df[feat] <= 0, feat] = np.nan

    # Total count
    df['Total_Count'] = df[count_cols].sum(axis=1)
    df = df[df['Total_Count'] > 0].copy()

    # ================= COMPUTE QUALITY SCORE =================
    df['Quality_Score'] = (5 * df['A+B+數'] + 4 * df['A-B+數'] +
                           3 * df['A-B數'] + 2 * df['A-B-數'] + 1 * df['B+數']) / df['Total_Count']

    # ================= CALCULATE OPTIMAL LIMITS =================
    # Current Spec Limits
    spec_limits = {
        'YS': (405, 500),
        'TS': (415, 550),
        'EL': (25, np.nan),
        'YPE': (4, np.nan)
    }

    optimal_limits = {}

    # Weighted data for all "good grade" (A+B+ + A-B+)
    df_good = df.copy()
    df_good['Good_Count'] = df_good['A+B+數'] + df_good['A-B+數']

    for feat in mech_features:
        if feat in df.columns:
            weighted_vals = []
            temp_df = df_good[[feat, 'Good_Count']].dropna()
            for idx, row in temp_df.iterrows():
                val = row[feat]
                w = int(row['Good_Count'])
                weighted_vals.extend([val]*w)
            if len(weighted_vals) > 0:
                lower = np.percentile(weighted_vals, 2.5)
                upper = np.percentile(weighted_vals, 97.5)
                # Adjust with spec
                if feat in spec_limits:
                    min_spec, max_spec = spec_limits[feat]
                    lower = max(lower, min_spec) if not np.isnan(min_spec) else lower
                    upper = min(upper, max_spec) if not np.isnan(max_spec) else upper
                optimal_limits[feat] = (round(lower,2), round(upper,2))
            else:
                optimal_limits[feat] = (None, None)

    # ================= DISPLAY TABLE =================
    st.header("📋 Optimal Mechanical Limits (Based on % Good Grade Coils)")
    table_data = []
    for feat in mech_features:
        lower, upper = optimal_limits.get(feat, (None, None))
        min_spec, max_spec = spec_limits.get(feat, (None, None))
        table_data.append({
            'Feature': feat,
            'Optimal Lower Limit': lower,
            'Optimal Upper Limit': upper,
            'Current Spec Min': min_spec,
            'Current Spec Max': max_spec
        })
    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True)

    # ================= HISTOGRAM VISUAL =================
    st.header("📊 Mechanical Feature Distribution with Limits")
    for feat in mech_features:
        if feat not in df.columns:
            continue
        st.markdown(f"### {feat}")
        # Weighted distribution
        weighted_vals = []
        for idx, row in df_good.iterrows():
            val = row[feat]
            w = int(row['Good_Count'])
            if pd.notna(val) and w>0:
                weighted_vals.extend([val]*w)
        if len(weighted_vals) == 0:
            st.info(f"No valid data for {feat}")
            continue

        fig, ax = plt.subplots(figsize=(12,6))
        sns.histplot(weighted_vals, bins=20, kde=True, color="#1f77b4", alpha=0.5, ax=ax)

        # Plot limits
        lower, upper = optimal_limits[feat]
        min_spec, max_spec = spec_limits.get(feat, (None, None))
        if lower is not None:
            ax.axvline(lower, color='green', linestyle='--', label='Optimal Lower Limit', linewidth=2)
        if upper is not None:
            ax.axvline(upper, color='green', linestyle='-', label='Optimal Upper Limit', linewidth=2)
        if min_spec is not None:
            ax.axvline(min_spec, color='red', linestyle='--', label='Current Spec Min', linewidth=2)
        if max_spec is not None:
            ax.axvline(max_spec, color='red', linestyle='-', label='Current Spec Max', linewidth=2)

        ax.set_xlabel(f"{feat}")
        ax.set_ylabel("Weighted Count of Good Grade Coils")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

else:
    st.info("Please upload your Excel data file (.xlsx) to start analysis.")
