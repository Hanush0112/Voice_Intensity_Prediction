import os
import librosa
import soundfile as sf
import dropbox
from io import BytesIO
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Dropbox API credentials
APP_KEY = os.getenv('APP_KEY')
APP_SECRET = os.getenv('APP_SECRET')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
REFRESH_TOKEN = os.getenv('REFRESH_TOKEN')

# Function to refresh access token using the refresh token
def refresh_access_token():
    url = "https://api.dropbox.com/oauth2/token"
    
    data = {
        'grant_type': 'refresh_token',
        'refresh_token': REFRESH_TOKEN,
        'client_id': APP_KEY,
        'client_secret': APP_SECRET
    }
    
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        tokens = response.json()
        new_access_token = tokens['access_token']
        
        # Update the access token in the .env file
        with open('.env', 'r') as file:
            env_vars = file.readlines()

        with open('.env', 'w') as file:
            for line in env_vars:
                if 'ACCESS_TOKEN' in line:
                    file.write(f'ACCESS_TOKEN={new_access_token}\n')
                else:
                    file.write(line)

        print("Access token refreshed.")
        return new_access_token
    else:
        print("Failed to refresh access token:", response.text)
        return None

# Function to get the valid access token (refresh if needed)
def get_valid_access_token():
    dbx = dropbox.Dropbox(ACCESS_TOKEN)

    try:
        # Check if the access token is valid
        dbx.users_get_current_account()
        return ACCESS_TOKEN
    except dropbox.exceptions.AuthError:
        # Refresh access token if it's expired
        print("Access token expired, refreshing...")
        return refresh_access_token()

# Function to split WAV files and extract timestamps in Dropbox
def split_wav_and_get_timestamps_dropbox(input_dir, output_dir, segment_duration=3):
    """
    Splits WAV files from a Dropbox folder into segments and extracts timestamps.

    Args:
        input_dir: Path to the input folder in Dropbox.
        output_dir: Path to the output folder in Dropbox.
        segment_duration: Duration of each segment in seconds.
    """
    # Get a valid Dropbox access token
    access_token = get_valid_access_token()

    # Connect to Dropbox
    dbx = dropbox.Dropbox(access_token)

    # Check if the output folder exists, if not, create it
    try:
        dbx.files_get_metadata(output_dir)
    except dropbox.exceptions.ApiError:
        dbx.files_create_folder(output_dir)

    timestamps = []
    
    # List files in the input folder
    for entry in dbx.files_list_folder(input_dir).entries:
        if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith(".wav"):
            filepath = f"{input_dir}/{entry.name}"
            
            # Download the file content
            _, res = dbx.files_download(filepath)
            audio_data = BytesIO(res.content)

            # Load the audio using librosa
            y, sr = librosa.load(audio_data, sr=None)

            for i in range(0, len(y), int(sr * segment_duration)):
                segment = y[i:i + int(sr * segment_duration)]
                
                # Create output filename
                output_filename = os.path.splitext(entry.name)[0] + f"_{i // (sr * segment_duration)}.wav"
                output_filepath = f"{output_dir}/{output_filename}"
                
                # Save the segment to a buffer
                buffer = BytesIO()
                sf.write(buffer, segment, sr, format='wav')
                buffer.seek(0)  # Rewind the buffer for reading

                # Upload the segment to Dropbox
                dbx.files_upload(buffer.read(), output_filepath, mode=dropbox.files.WriteMode.overwrite)

                # Calculate timestamps
                start_time = i / sr
                end_time = (i + len(segment)) / sr
                timestamps.append((output_filename, start_time, end_time))

            # Check if there's remaining audio (less than segment duration)
            remaining_audio = y[i:]
            if len(remaining_audio) > 0 and i + len(remaining_audio) < len(y):
                output_filename = os.path.splitext(entry.name)[0] + f"_{i // (sr * segment_duration)}_remaining.wav"
                output_filepath = f"{output_dir}/{output_filename}"

                # Save and upload the remaining segment
                buffer = BytesIO()
                sf.write(buffer, remaining_audio, sr, format='wav')
                buffer.seek(0)
                dbx.files_upload(buffer.read(), output_filepath, mode=dropbox.files.WriteMode.overwrite)

                start_time = i / sr
                end_time = (i + len(remaining_audio)) / sr
                timestamps.append((output_filename, start_time, end_time))

    return timestamps


# Example usage
input_directory = "/input_wav"  # Replace with your input folder in Dropbox
output_directory = "/output_wav"  # Replace with your output folder in Dropbox

timestamps = split_wav_and_get_timestamps_dropbox(input_directory, output_directory)

print("Timestamps:")
for filename, start_time, end_time in timestamps:
    print(f"File: {filename}, Start Time: {start_time:.2f}s, End Time: {end_time:.2f}s")
    
