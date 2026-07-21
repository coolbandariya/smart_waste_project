import folium
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

from person1_preprocessing import preprocess_data
from person2_ml_prediction import train_ml_model
from person3_fuzzy_logic import calculate_priority, create_fuzzy_system
from person4_optimization import solve_vrp

st.set_page_config(
    page_title="Smart Waste Collection Dashboard",
    page_icon="🗑️",
    layout="wide",
)

st.title("🗑️ Smart Waste Collection & Route Optimization System")
st.markdown(
    "AI-Powered IoT Waste Monitoring, Dynamic Priority Scoring, and Route Optimization"
)


# Load or train model
@st.cache_resource
def load_resources():
    fuzzy_sim = create_fuzzy_system()
    try:
        model = train_ml_model()
    except Exception:
        model = None
    return fuzzy_sim, model


fuzzy_sim, ml_model = load_resources()

# Load Data (Generates automatically if missing)
import os
from person1_preprocessing import generate_synthetic_bin_data

csv_path = "cleaned_waste_data.csv"

if not os.path.exists(csv_path):
    st.info("Generating waste bin data for first-time setup...")
    raw_df = generate_synthetic_bin_data()
    clean_df = preprocess_data(raw_df)
    clean_df.to_csv(csv_path, index=False)

df = pd.read_csv(csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
latest_df = df.groupby("bin_id").last().reset_index()

# Calculate priority scores for current state
latest_df["priority_score"] = latest_df.apply(
    lambda row: calculate_priority(
        row["fill_level_pct"],
        12,  # sample hours elapsed
        row["criticality"],
        fuzzy_sim,
    ),
    axis=1,
)

# Sidebar Filters & Controls
st.sidebar.header("🕹️ Collection Controls")
min_priority = st.sidebar.slider(
    "Minimum Priority Score for Pickup", 0, 100, 50
)
num_trucks = st.sidebar.number_input(
    "Number of Collection Trucks", min_value=1, max_value=5, value=2
)
truck_capacity = st.sidebar.slider("Truck Capacity (Liters)", 500, 3000, 1000)

# Filter bins requiring collection
pickup_bins = latest_df[
    latest_df["priority_score"] >= min_priority
].to_dict("records")

# Top Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Bins Monitored", len(latest_df))
col2.metric("Bins Requiring Pickup", len(pickup_bins))
col3.metric("Avg Fill Level", f"{latest_df['fill_level_pct'].mean():.1f}%")
col4.metric("High Priority Bins", len(latest_df[latest_df["priority_score"] > 70]))

st.divider()

# Layout: Map + Optimization Results
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📍 Live Bin Locations & Priority Map")

    # Depot coordinates
    depot_lat, depot_lon = 28.5355, 77.3910
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12)

    # Add Depot Marker
    folium.Marker(
        [depot_lat, depot_lon],
        popup="Central Waste Depot",
        icon=folium.Icon(color="black", icon="home"),
    ).add_to(m)

    # Add Bin Markers
    for _, bin_row in latest_df.iterrows():
        p_score = bin_row["priority_score"]
        color = "red" if p_score >= 70 else "orange" if p_score >= 40 else "green"

        folium.CircleMarker(
            location=[bin_row["latitude"], bin_row["longitude"]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"<b>Bin:</b> {bin_row['bin_id']}<br>"
            f"<b>Fill:</b> {bin_row['fill_level_pct']}%<br>"
            f"<b>Priority:</b> {p_score}/100<br>"
            f"<b>Criticality:</b> {bin_row['criticality']}",
        ).add_to(m)

    # Fixed: Using folium_static instead of st_folium
    folium_static(m, width=700, height=450)

with col_right:
    st.subheader("🚚 Optimized Truck Routes")

    if st.button("Generate Optimal Routes", type="primary"):
        if not pickup_bins:
            st.info("No bins meet the selected priority threshold for pickup.")
        else:
            routes = solve_vrp(
                pickup_bins,
                num_vehicles=num_trucks,
                vehicle_capacity=truck_capacity,
            )

            for vehicle, details in routes.items():
                visited_indices = details["route_bin_indices"]
                bin_names = [pickup_bins[idx]["bin_id"] for idx in visited_indices]

                st.markdown(f"### {vehicle}")
                st.write(
                    f"**Route:** {' ➔ '.join(['Depot'] + bin_names + ['Depot'])}"
                )
                st.write(f"**Total Load:** {details['total_load']} Liters")
                st.progress(
                    min(details["total_load"] / truck_capacity, 1.0)
                )

# Data Table View
with st.expander("📊 View Detailed Bin Status Data"):
    st.dataframe(
        latest_df[
            [
                "bin_id",
                "fill_level_pct",
                "criticality",
                "priority_score",
                "capacity_liters",
            ]
        ].sort_values("priority_score", ascending=False)
    )