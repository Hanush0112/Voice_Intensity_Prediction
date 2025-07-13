import os
import librosa
import soundfile as sf

# Function to split WAV files and extract timestamps from a local directory
def split_wav_and_get_timestamps_local(input_dir, output_dir, segment_duration=3):
    """
    Splits WAV files from a local folder into segments and extracts timestamps.

    Args:
        input_dir: Path to the input folder on the local system.
        output_dir: Path to the output folder on the local system.
        segment_duration: Duration of each segment in seconds.
    """
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamps = []

    # List files in the input folder
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_dir, filename)

            # Load the audio using librosa
            y, sr = librosa.load(filepath, sr=None)

            for i in range(0, len(y), int(sr * segment_duration)):
                segment = y[i:i + int(sr * segment_duration)]
                
                # Create output filename
                output_filename = os.path.splitext(filename)[0] + f"_{i // (sr * segment_duration)}.wav"
                output_filepath = os.path.join(output_dir, output_filename)

                # Save the segment to the output folder
                sf.write(output_filepath, segment, sr, format='wav')

                # Calculate timestamps
                start_time = i / sr
                end_time = (i + len(segment)) / sr
                timestamps.append((output_filename, start_time, end_time))

            # Check if there's remaining audio (less than segment duration)
            remaining_audio = y[i:]
            if len(remaining_audio) > 0 and i + len(remaining_audio) < len(y):
                output_filename = os.path.splitext(filename)[0] + f"_{i // (sr * segment_duration)}_remaining.wav"
                output_filepath = os.path.join(output_dir, output_filename)

                # Save the remaining segment
                sf.write(output_filepath, remaining_audio, sr, format='wav')

                start_time = i / sr
                end_time = (i + len(remaining_audio)) / sr
                timestamps.append((output_filename, start_time, end_time))

    return timestamps


# Example usage
input_directory = "uploads"  # Replace with your local input folder path
output_directory = "output"  # Replace with your local output folder path

timestamps = split_wav_and_get_timestamps_local(input_directory, output_directory)

print("Timestamps:")
for filename, start_time, end_time in timestamps:
    print(f"File: {filename}, Start Time: {start_time:.2f}s, End Time: {end_time:.2f}s")
