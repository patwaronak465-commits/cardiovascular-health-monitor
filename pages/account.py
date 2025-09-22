import streamlit as st

st.set_page_config(page_title="Account", page_icon="👤")

if not st.session_state.get("password_correct", False):
    st.error("You must be logged in to view this page.")
    st.stop()

st.title("👤 Account Information")
st.write(f"**Username:** `{st.session_state.get('username', 'N/A')}`")
st.write("**Subscription:** Pro Plan")
st.write("**Member Since:** 2025")

if st.sidebar.button("Logout"):
    st.session_state["password_correct"] = False
    st.rerun()