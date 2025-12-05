import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# 1. Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv("hustler_marketing_final_age_104weeks_7channels.csv")
    # For optimization
    agg_opt = df.groupby(['week', 'channel'], as_index=False).agg(
        spend_inr=('spend_inr', 'sum'),
        conversions=('conversions', 'sum')
    )
    # Channel level seasonality for RF simulator
    seasonality = df.groupby('channel').agg(
        seasonality_base=('seasonality_base', 'mean'),
        seasonality_factor_channel=('seasonality_factor_channel', 'mean')
    )
    avg_aov = df['avg_order_value_inr'].mean()
    return df, agg_opt, seasonality, avg_aov

# 2. Fit per channel response curves for optimization
@st.cache_data
def fit_channel_models(agg_opt):
    channels = agg_opt['channel'].unique().tolist()
    coeffs = {}
    slopes = []
    for ch in channels:
        sub = agg_opt[agg_opt['channel'] == ch]
        X = sub[['spend_inr']].values
        y = sub['conversions'].values
        m = LinearRegression().fit(X, y)
        coeffs[ch] = (float(m.intercept_), float(m.coef_[0]))
        slopes.append(float(m.coef_[0]))
    return channels, coeffs, np.array(slopes)

# 3. Load RF model and encoder for simulator
@st.cache_resource
def load_rf_model():
    model = joblib.load("hustler_conversion_model_rf.pkl")
    ohe = joblib.load("hustler_ohe_channel.pkl")
    return model, ohe

# 4. Optimization logic
def optimize_budget(total_budget, channels, coeffs, slopes):
    ch_to_idx = {ch: i for i, ch in enumerate(channels)}

    min_shares = np.zeros(len(channels))
    max_shares = np.zeros(len(channels))

    for ch in channels:
        i = ch_to_idx[ch]
        if ch == "Display Ads":
            min_shares[i] = 0.00
            max_shares[i] = 0.15
        else:
            min_shares[i] = 0.05
            if ch in ["Influencer Marketing", "YouTube Ads"]:
                max_shares[i] = 0.40
            else:
                max_shares[i] = 0.30

    min_bounds = min_shares * total_budget
    max_bounds = max_shares * total_budget

    alloc = min_bounds.copy()
    budget_left = total_budget - min_bounds.sum()

    order = np.argsort(-slopes)

    for i in order:
        if budget_left <= 1e-6:
            break
        room = max_bounds[i] - alloc[i]
        if room <= 0:
            continue
        add = min(room, budget_left)
        alloc[i] += add
        budget_left -= add

    equal_spend = np.full(len(channels), total_budget / len(channels))
    total_conv_opt = 0.0
    total_conv_eq = 0.0

    for i, ch in enumerate(channels):
        a, b = coeffs[ch]
        total_conv_opt += a + b * alloc[i]
        total_conv_eq += a + b * equal_spend[i]

    lift = (total_conv_opt - total_conv_eq) / total_conv_eq * 100 if total_conv_eq > 0 else 0.0

    df_alloc = pd.DataFrame({
        "Channel": channels,
        "Optimal Spend (₹)": np.round(alloc, 2),
        "% of Budget": np.round(alloc / total_budget * 100, 2)
    }).sort_values(by="Optimal Spend (₹)", ascending=False)

    return df_alloc, total_conv_eq, total_conv_opt, lift

# 5. RF based scenario simulator
def predict_conversions_rf(spend_dict, total_budget, model, ohe, seasonality, future_week=105, avg_aov=1000.0):
    channels = list(spend_dict.keys())
    spends = np.array([spend_dict[ch] for ch in channels])

    log_spend = np.log1p(spends)
    weekly_budget_vec = np.full(len(channels), total_budget)
    week_vec = np.full(len(channels), future_week)

    seasonality_base_vec = []
    seasonality_factor_vec = []
    for ch in channels:
        seasonality_base_vec.append(seasonality.loc[ch, 'seasonality_base'])
        seasonality_factor_vec.append(seasonality.loc[ch, 'seasonality_factor_channel'])

    seasonality_base_vec = np.array(seasonality_base_vec)
    seasonality_factor_vec = np.array(seasonality_factor_vec)

    X_num = np.column_stack([
        spends,
        log_spend,
        seasonality_base_vec,
        seasonality_factor_vec,
        weekly_budget_vec,
        week_vec
    ])

    ch_df = pd.DataFrame({'channel': channels})
    ch_encoded = ohe.transform(ch_df[['channel']])

    X_full = np.hstack([X_num, ch_encoded])

    preds = model.predict(X_full)
    total_conversions = preds.sum()
    total_revenue = total_conversions * avg_aov

    per_channel = pd.DataFrame({
        "Channel": channels,
        "Spend (₹)": spends,
        "Predicted conversions": np.round(preds, 1)
    }).sort_values(by="Spend (₹)", ascending=False)

    return total_conversions, total_revenue, per_channel

# 6. Streamlit UI
def main():
    st.set_page_config(page_title="Hustler AI Marketing Optimizer", layout="wide")
    st.title("Hustler AI Agent for Cross Channel Marketing Optimization")

    df_raw, agg_opt, seasonality, avg_aov = load_data()
    channels, coeffs, slopes = fit_channel_models(agg_opt)
    model_rf, ohe = load_rf_model()

    tab1, tab2, tab3 = st.tabs(["Optimization engine", "What-if simulator", "Data explorer"])

    # ---------- TAB 1: OPTIMIZATION ----------
    with tab1:
        st.subheader("AI optimized budget allocation")

        st.markdown("""
        This tool uses 104 weeks of Hustler performance data to:
        - Learn response curves (conversions vs spend) for each channel
        - Recommend an optimal marketing mix for a given weekly budget
        - Compare equal split vs optimized allocation in terms of predicted conversions
        """)

        st.sidebar.header("Optimization controls")
        total_budget_opt = st.sidebar.number_input(
            "Total weekly budget for optimization (₹)",
            min_value=10000,
            max_value=500000,
            value=100000,
            step=5000,
            key="budget_opt"
        )

        if st.sidebar.button("Run optimization", key="run_opt"):
            df_alloc, conv_eq, conv_opt, lift = optimize_budget(total_budget_opt, channels, coeffs, slopes)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total budget", f"₹{total_budget_opt:,.0f}")
            with col2:
                st.metric("Conversions (equal split)", f"{conv_eq:.1f}")
            with col3:
                st.metric("Conversions (optimized)", f"{conv_opt:.1f}", f"{lift:.1f}% lift")

            st.subheader("Optimal budget allocation by channel")
            st.dataframe(df_alloc.reset_index(drop=True), use_container_width=True)

            st.subheader("Budget allocation share")
            st.bar_chart(df_alloc.set_index("Channel")["% of Budget"])

            st.caption("Channels with higher marginal conversions per rupee receive more budget. Display Ads can rationally get zero if its incremental efficiency is low.")
        else:
            st.info("Set a budget in the sidebar and click 'Run optimization' to see recommendations.")

    # ---------- TAB 2: RF WHAT-IF SIMULATOR ----------
    with tab2:
        st.subheader("What-if scenario using Random Forest predictor")
        st.markdown("Set manual spends per channel and see predicted conversions for that plan.")

        col_inputs = st.columns(3)
        spend_inputs = {}

        with col_inputs[0]:
            spend_inputs["Influencer Marketing"] = st.number_input("Influencer Marketing (₹)", 0, 200000, 40000, 5000)
            spend_inputs["YouTube Ads"] = st.number_input("YouTube Ads (₹)", 0, 200000, 35000, 5000)
        with col_inputs[1]:
            spend_inputs["Instagram Ads"] = st.number_input("Instagram Ads (₹)", 0, 200000, 10000, 5000)
            spend_inputs["Email"] = st.number_input("Email (₹)", 0, 200000, 5000, 5000)
        with col_inputs[2]:
            spend_inputs["Facebook Ads"] = st.number_input("Facebook Ads (₹)", 0, 200000, 5000, 5000)
            spend_inputs["Google Search Ads"] = st.number_input("Google Search Ads (₹)", 0, 200000, 5000, 5000)
            spend_inputs["Display Ads"] = st.number_input("Display Ads (₹)", 0, 200000, 0, 5000)

        total_manual_budget = sum(spend_inputs.values())
        st.markdown(f"**Total manual budget:** ₹{total_manual_budget:,.0f}")

        if st.button("Predict performance for this plan"):
            total_conv, total_rev, df_per_channel = predict_conversions_rf(
                spend_inputs, total_manual_budget, model_rf, ohe, seasonality, future_week=105, avg_aov=avg_aov
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Predicted conversions", f"{total_conv:.1f}")
            with col_b:
                st.metric("Predicted revenue (approx)", f"₹{total_rev:,.0f}")

            st.subheader("Channel level prediction")
            st.dataframe(df_per_channel.reset_index(drop=True), use_container_width=True)

            st.caption("This uses the Random Forest model trained on 104 weeks of data. Compare this manual plan with the optimized mix from Tab 1.")
        else:
            st.info("Adjust spends and click 'Predict performance for this plan'.")

    # ---------- TAB 3: DATA EXPLORER ----------
    with tab3:
        st.subheader("Data explorer by channel and age group")
        st.markdown("Slice the 104-week dataset by channel, age group, and time to see how performance actually behaved.")

        channels_all = sorted(df_raw['channel'].unique().tolist())
        age_groups_all = sorted(df_raw['age_group'].unique().tolist())
        week_min = int(df_raw['week'].min())
        week_max = int(df_raw['week'].max())

        f1, f2, f3 = st.columns(3)
        with f1:
            sel_channels = st.multiselect("Channels", channels_all, default=channels_all)
        with f2:
            sel_ages = st.multiselect("Age groups", age_groups_all, default=age_groups_all)
        with f3:
            week_range = st.slider("Week range", min_value=week_min, max_value=week_max, value=(week_min, week_max))

        mask = (
            df_raw['channel'].isin(sel_channels)
            & df_raw['age_group'].isin(sel_ages)
            & df_raw['week'].between(week_range[0], week_range[1])
        )
        df_filt = df_raw[mask].copy()

        if df_filt.empty:
            st.warning("No data for this combination of filters.")
        else:
            # KPIs for this slice
            total_spend = df_filt['spend_inr'].sum()
            total_conv = df_filt['conversions'].sum()
            total_rev = df_filt['revenue_inr'].sum()
            avg_roi_slice = df_filt['roi'].mean()

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Total spend", f"₹{total_spend:,.0f}")
            with k2:
                st.metric("Total conversions", f"{total_conv:,.0f}")
            with k3:
                st.metric("Total revenue", f"₹{total_rev:,.0f}")
            with k4:
                st.metric("Average ROI", f"{avg_roi_slice:.2f}")

            group_choice = st.radio(
                "Group by",
                ["Channel", "Age group", "Channel × Age group"],
                horizontal=True
            )

            if group_choice == "Channel":
                agg = df_filt.groupby('channel').agg(
                    spend_inr=('spend_inr', 'sum'),
                    conversions=('conversions', 'sum'),
                    revenue_inr=('revenue_inr', 'sum'),
                    avg_roi=('roi', 'mean')
                ).round(2)
            elif group_choice == "Age group":
                agg = df_filt.groupby('age_group').agg(
                    spend_inr=('spend_inr', 'sum'),
                    conversions=('conversions', 'sum'),
                    revenue_inr=('revenue_inr', 'sum'),
                    avg_roi=('roi', 'mean')
                ).round(2)
            else:
                agg = df_filt.groupby(['channel', 'age_group']).agg(
                    spend_inr=('spend_inr', 'sum'),
                    conversions=('conversions', 'sum'),
                    revenue_inr=('revenue_inr', 'sum'),
                    avg_roi=('roi', 'mean')
                ).round(2)

            st.subheader("Aggregated performance for selected slice")
            st.dataframe(agg, use_container_width=True)

            st.subheader("Conversions by week for this slice")
            conv_ts = df_filt.groupby('week')['conversions'].sum()
            st.line_chart(conv_ts)

            st.caption("Example: select only Instagram + age group 18–34 to see how that audience performed on Instagram over time.")

    st.markdown("---")
    st.markdown("Data source: simulated weekly performance for Hustler across 7 channels and 3 age groups.")

if __name__ == "__main__":
    main()
