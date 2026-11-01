import streamlit as st

st.set_page_config(
    page_title="NSE Stock Assistant",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NSE Stock Assistant")
st.success("Your app is running successfully!")

st.write(
    "Setup test complete. The stock scanner, news checks, "
    "and backtesting modules will be added next."
)

st.warning(
    "Project testing only. No trading signals are active."
)
