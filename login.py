import streamlit as st

def login_page():
    st.set_page_config(page_title="Login", page_icon="🔐")
    st.title("🔐 Login")

    password = st.text_input("Enter password", type="password")
    
    # Placeholder password for demonstration
    if st.button("Login"):
        if password == "12345":  # Replace with a secure password
            st.session_state['password_correct'] = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Incorrect password.")

# Main app logic
if 'password_correct' not in st.session_state or not st.session_state['password_correct']:
    login_page()
else:
    st.title("You are logged in.")
    st.write("Please select a page from the sidebar.")