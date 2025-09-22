import streamlit as st

st.set_page_config(page_title="Settings", page_icon="⚙️")

st.title("⚙️ Settings")

if st.session_state.get('password_correct', False):
    st.header("App Configuration")
    
    current_theme = st.session_state.get('theme', 'Dark')
    new_theme = st.radio("Choose Theme", ["Dark", "Light"], index=0 if current_theme == "Dark" else 1)
    
    if new_theme != current_theme:
        st.session_state['theme'] = new_theme
        st.success(f"Theme changed to {new_theme}!")
        st.rerun()

else:
    st.error("You must be logged in to view this page.")