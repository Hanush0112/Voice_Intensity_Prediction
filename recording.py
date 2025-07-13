from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from split import split_wav_and_get_timestamps_local  # Local version for splitting
from m1 import process_all_wav_files_in_local_directory as model_1_process  # Local version for model 1 processing
from m3 import process_all_wav_files_in_local_directory as model_2_process  # Local version for model 2 processing
from sugges2 import process_model_outputs  # Local version for nervousness analysis
from dotenv import load_dotenv
from m3 import a
from m1 import b
# Load environment variables
load_dotenv()

app = Flask(__name__)

# Path to store uploaded files locally
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if the file has the allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert audio file to WAV format if necessary
def convert_to_wav(filepath):
    audio = AudioSegment.from_file(filepath)
    wav_filename = os.path.splitext(filepath)[0] + ".wav"
    audio.export(wav_filename, format="wav")
    return wav_filename

@app.route('/')
def index():
    return render_template('index.html',Intensity=a, Nervousness=b)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Convert to .wav if necessary
        if not filename.endswith('.wav'):
            filepath = convert_to_wav(filepath)
            filename = os.path.basename(filepath)

        # Split the WAV file into chunks locally
        split_wav_and_get_timestamps_local(app.config['UPLOAD_FOLDER'], 'output_wav')

        # Run the models on the local WAV file
        model_1_output = model_1_process(os.path.join(app.config['UPLOAD_FOLDER'], 'output_wav'))
        model_2_output = model_2_process(os.path.join(app.config['UPLOAD_FOLDER'], 'output_wav'))

        # Combine model outputs into a 2D array
        model_outputs = [model_1_output, model_2_output]

        # Determine majority opinion
        nervous_count = sum(1 for result in model_outputs if result[1] == "nervous")
        not_nervous_count = len(model_outputs) - nervous_count
        majority_opinion = "nervous" if nervous_count > not_nervous_count else "not nervous"

        # Get timestamps for nervousness
        nervous_timestamps = process_model_outputs(
            app.config['UPLOAD_FOLDER'], 'output_wav', None, None, None, target_emotion="nervous"
        )

        return render_template('index.html', results=majority_opinion, timestamps=nervous_timestamps)

    return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)
