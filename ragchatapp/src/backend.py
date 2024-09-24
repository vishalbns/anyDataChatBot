import os
import pickle
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse
from google.cloud import storage
from urllib.parse import urlparse
import validators
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the path for Google Cloud service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../myraglangchainapp-fb8241de750a.json"

# Function to determine content type and folder based on file extension
def content_type_finder(file_name):
    if file_name.endswith('.csv'):
        content_type = 'text/csv'
        folder_name = 'csvfiles'
    elif file_name.endswith('.txt'):
        content_type = 'text/plain'
        folder_name = 'txtfiles'
    elif file_name.endswith('.json'):
        content_type = 'application/json'
        folder_name = 'jsonfiles'
    else:
        content_type = 'application/octet-stream'
        folder_name = 'others'

    return content_type, folder_name

# Create FastAPI application instance
app = FastAPI()

# Google Cloud Storage client
client = storage.Client()
bucket_name = 'userdatafiles'

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload files to Google Cloud Storage.
    Supports CSV, TXT, JSON, and other file types.
    """
    try:
        # Get the file's content type and folder name based on its extension
        content_type, folder_name = content_type_finder(file.filename)

        # Create a blob object for the specified bucket
        bucket = client.bucket(bucket_name)    
        blob = bucket.blob(f"{folder_name}/{file.filename}")
    
        # Upload the file to the Google Cloud Storage bucket
        blob.upload_from_file(file.file, content_type=content_type)

        logger.info(f"Uploaded file '{file.filename}' to '{bucket_name}/{folder_name}' with content type '{content_type}'.")
        return JSONResponse(content={"message": f"File '{file.filename}' uploaded to '{bucket_name}', /{folder_name} with content type '{content_type}'"})
    
    except Exception as e:
        logger.error(f"Failed to upload file: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload/text/")
async def upload_text(content: str = Body(...)):
    """
    Endpoint to upload text content to Google Cloud Storage.
    Text can be plain text or a URL. URLs are saved in a different folder.
    """
    try:
        bucket = client.bucket(bucket_name)

        # Check if the content is a valid URL
        # Check if the content is a valid URL
        print(content)
        if content.startswith(("text=http")):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'urls/url_input_{timestamp}.txt'  # Save to "urls" folder
        else:
            # Generate a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'plaininputtexts/uploaded_text_{timestamp}.txt'  # Filename with timestamp

        # Create a blob object for the specified filename
        blob = bucket.blob(filename)

        # Upload the text content to the Google Cloud Storage bucket
        blob.upload_from_string(content)

        logger.info(f"Uploaded text successfully to {filename}.")
        return JSONResponse(content={"message": f"Uploaded text successfully to {filename}."})
    
    except Exception as e:
        logger.error(f"Failed to upload text: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
