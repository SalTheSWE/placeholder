# streamlit_app.py
# Pure Streamlit interface skeleton (no processing logic)

import streamlit as st
import os 
st.set_page_config(page_title="First Class Cabin Monitor", layout="wide")
st.title("First Class Cabin Monitor")
st.markdown("""
<style>
.live-feed video {
    width: 100%;
    max-width: 100px;   /* change this */
}
</style>
""", unsafe_allow_html=True)
VIDEO_PATH = "resources/input_video/IMG_3433.MOV"
# ---------------------------
# Tabs
# ---------------------------
tab_live, tab_users, tab_db = st.tabs(
    ["Live Monitor", "First Class Users", "Search Database"]
)

# ===========================
# TAB 1 — LIVE MONITOR
# ===========================
with tab_live:
    # Outer layout
    col_feed_area, col_controls, col_alerts = st.columns([3, 1, 1])

    # ---- Live Feed ----
    with col_feed_area:
        st.subheader("Live Feed")

        # Inner layout to control video width
        video_col, spacer_col = st.columns([1, 2])  # <-- video takes 1/3 of feed area

        with video_col:
            if os.path.exists(VIDEO_PATH):
                st.video(VIDEO_PATH)
            else:
                st.info("Video file not found.")


    # ---- CSV output ----
    st.subheader("Flagged events (CSV sent to employees)")
    st.info("Flagged event table will appear here.")
    table_placeholder = st.empty()

    st.download_button(
        "Download flagged events CSV",
        data=b"",
        file_name="flagged_events.csv",
        mime="text/csv",
    )

# ===========================
# TAB 2 — FIRST CLASS USERS
# ===========================
with tab_users:

    st.subheader("First Class Passenger List")
    st.info("First class passenger CSV table will appear here.")

    users_table_placeholder = st.empty()

    st.markdown("### Passenger Image Viewer")

    col_select, col_action = st.columns([3, 1])

    with col_select:
        user_selector = st.selectbox(
            "Select passenger",
            options=["Passenger list will load here"],
        )

    with col_action:
        show_image_btn = st.button("Show Image")

    st.info("Passenger image will appear here.")
    user_image_placeholder = st.empty()

    st.download_button(
        "Download passenger CSV",
        data=b"",
        file_name="first_class_passengers.csv",
        mime="text/csv",
    )

# ===========================
# TAB 3 — SEARCH DATABASE
# ===========================
with tab_db:

    st.subheader("Search Database")
    st.info("Search database records using a query.")

    col_query, col_action = st.columns([4, 1])

    with col_query:
        query_input = st.text_input(
            "Enter search query",
            placeholder="Example: passenger_id=1234 OR identity='John Doe'",
        )

    with col_action:
        search_btn = st.button("Search")

    st.markdown("### Search Results")
    st.info("Query results will appear here.")
    db_results_placeholder = st.empty()

    st.download_button(
        "Download search results",
        data=b"",
        file_name="search_results.csv",
        mime="text/csv",
    )

# ---------------------------
# Button feedback (UI only)
# ---------------------------
if 'start_btn' in locals() and start_btn:
    st.success("Live feed started (interface only).")

if 'stop_btn' in locals() and stop_btn:
    st.warning("Live feed stopped.")

if 'search_btn' in locals() and search_btn:
    st.success("Search executed (interface only).")

if 'show_image_btn' in locals() and show_image_btn:
    st.success("Passenger image requested (interface only).")
