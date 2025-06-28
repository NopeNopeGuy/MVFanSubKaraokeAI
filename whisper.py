import whisperx
import gc
import json
import os
# The 'google.genai' import was in your original code but not used.
# It's kept here in case you plan to use it later.
from google import genai

device = "cpu"
audio_file = "vocals.wav"
batch_size = 4 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

# Check if the audio file exists before proceeding
if not os.path.exists(audio_file):
    print(f"Error: Audio file not found at '{audio_file}'")
    print("Please make sure the audio file is in the same directory as the script, or provide the full path.")
    exit()

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("kotoba-tech/kotoba-whisper-v2.0-faster", device, compute_type=compute_type)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size, language="ja")

print("--- Initial Transcription Result ---")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("-" * 30)


# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# --- MODIFICATION START ---

# Define the name for your output file
output_filename = "aligned_transcription.json"

# Write the final aligned result to the specified file
# Using 'w' mode to write, and 'utf-8' encoding to support Japanese characters
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Successfully aligned transcription and saved the result to '{output_filename}'")

# --- MODIFICATION END ---
