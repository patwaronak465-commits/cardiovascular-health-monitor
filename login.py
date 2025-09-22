import streamlit as st

# Function to handle the login page UI
def login_page():
    st.set_page_config(page_title="Login", page_icon="🔐")
    st.title("🔐 Login")

    password = st.text_input("Enter password", type="password")

    if st.button("Login"):
        # Placeholder password for demonstration
        if password == "12345":  # You should use a more secure method in a real app
            st.session_state['password_correct'] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

# Function to display content after a successful login
def main_app():
    st.title("You are logged in")
    st.write("Please select a page from the sidebar")
    
    # Add a logout button in the sidebar
    # This button, when clicked, will set the session state to False and rerun the app
    with st.sidebar:
        st.button("Logout", on_click=lambda: st.session_state.update(password_correct=False))

# Main logic to check for login status
if 'password_correct' not in st.session_state or not st.session_state['password_correct']:
    login_page()
else:
    main_app()