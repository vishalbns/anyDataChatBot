import streamlit as st
import requests

# Set the API endpoints
API_BASE_URL = "http://127.0.0.1:8000"  # Change this to your FastAPI server URL

# Title of the app
st.title("Chat with your data!")

# Radio button to select input type
input_type = st.radio("Select Input Type", ("File Upload", "Text Input"))

# Handling File Upload
if input_type == "File Upload":
    uploaded_file = st.file_uploader("Choose a file (CSV, TXT, JSON)", type=["csv", "txt", "json"])
    
    if uploaded_file is not None:
        # Show file details
        st.write(f"Filename: {uploaded_file.name}")
        st.write(f"File type: {uploaded_file.type}")

        # Button to upload the file
        if st.button("Upload File"):
            response = requests.post(f"{API_BASE_URL}/upload/", files={"file": uploaded_file})
            st.write(response.json())

# Handling Text Input
elif input_type == "Text Input":
    text_input = st.text_area("Enter plain text or URL")
    
    if text_input:
        # Button to upload the text
        if st.button("Upload Text"):
            response = requests.post(f"{API_BASE_URL}/upload/text/", data={'text': text_input})
            st.write(response.json())

