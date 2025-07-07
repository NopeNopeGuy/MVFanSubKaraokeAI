# Karaoke Subtitle Creator

This project is a comprehensive suite of tools designed to automate the creation of high-quality, syllable-timed karaoke subtitles. It fetches a song from YouTube, separates the vocals, transcribes the lyrics with precise timing, and generates a final `.ass` subtitle file, which can be combined with the instrumental track to create a karaoke video.

## Features

-   **Automated Song Acquisition**: Downloads audio from YouTube and fetches lyrics from Genius.
-   **Vocal-Instrumental Separation**: Uses Demucs via the Sieve platform to isolate vocals for accurate transcription.
-   **Advanced Transcription**:
    -   Multiple transcription backends: Google Gemini, local Whisper, or a Sieve/Whisper+Gemini enhancement pipeline.
    -   Generates detailed JSON output with word and syllable-level timestamps.
    -   Supports romanization and translation for non-English songs.
-   **Karaoke File Generation**:
    -   Creates `.ass` subtitle files with syllable-level karaoke effects (`\k`).
    -   Supports standard and multi-line "instrumental" karaoke formats.
    -   Optionally generates a final karaoke video with the subtitles burned in using FFmpeg. (Currently super broken)
-   **Modular & Configurable**: Each step is a separate script, allowing for flexibility and manual intervention.

## Workflow

The process is broken down into three main stages, handled by three distinct Python scripts:

### 1. Prepare Song Assets (`start.py`)

This script takes a YouTube URL, downloads the audio, fetches the lyrics, and separates the vocal and instrumental tracks.

**Usage:**
```bash
python start.py "YOUTUBE_URL" --name "Song Name"
```

This will create a new directory (e.g., `Song_Name/`) containing:
-   `song.mp3`: The original audio.
-   `lyrics.txt`: The fetched lyrics.
-   `vocals.mp3`: The isolated vocal track.
-   `instrumental.mp3`: The isolated instrumental track.

### 2. Transcribe Vocals (`transcription.py`)

This script transcribes the isolated vocal track (`vocals.mp3`) into a structured JSON file with precise syllable timings. The `whisper-gemini` mode is recommended for the highest quality, as it uses the fetched lyrics to correct the transcription.

**Usage:**
```bash
python transcription.py "Song_Name/vocals.mp3" --model whisper-gemini --lyrics "Song_Name/lyrics.txt" --output "Song_Name/transcription.json"
```

This will produce `transcription.json` inside the song's directory.

### 3. Create Subtitles (`creator.py`)

This script takes the `transcription.json` file and generates the final karaoke `.ass` subtitle file. It can also generate a video if the instrumental track is present.

**Usage:**
```bash
# Generate a standard .ass file
python creator.py "Song_Name/transcription.json"

# Generate an instrumental-style .ass file and a karaoke video
python creator.py "Song_Name/transcription.json" --instrumental-karaoke --generate-video
```

This creates:
-   `transcription.ass`: The standard karaoke subtitle file.
-   `transcription_instrumental.ass`: The instrumental-style subtitle file.
-   `transcription_karaoke_video.mp4`: The final karaoke video.

## Setup

### Dependencies

First, install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### External Tools

You must have **FFmpeg** installed and available in your system's PATH. This is required for audio processing by `pydub`, `yt-dlp`, and for video creation.

-   **On Ubuntu/Debian**: `sudo apt-get install ffmpeg`
-   **On macOS (Homebrew)**: `brew install ffmpeg`
-   **On Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` directory to your PATH.

### API Keys and Authentication

1.  **Google Gemini API Key**: Required for transcription, translation, and enhancement.
    -   Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    -   Set it as an environment variable:
        ```bash
        export GEMINI_API_KEY="YOUR_API_KEY"
        ```

2.  **Genius API Token**: Required for fetching lyrics with `start.py`.
    -   Create an API client at [genius.com/api-clients](https://genius.com/api-clients).
    -   Set it as an environment variable:
        ```bash
        export GENIUS_API_TOKEN="YOUR_API_TOKEN"
        ```

3.  **Sieve Authentication**: Required for vocal separation (`start.py`) and Sieve-based transcription (`transcription.py`).
    -   Run the login command once:
        ```bash
        sieve login
        ```

## Script Details

### `start.py`

Fetches and prepares all the necessary assets for a song.

```bash
python start.py <youtube_url> [--name <custom_name>]
```
-   **`youtube_url`**: The URL of the YouTube video.
-   **`--name`**: (Optional) A custom name for the song. If not provided, the cleaned-up YouTube video title is used.

### `transcription.py`

Transcribes an audio file into a structured JSON format.

```bash
python transcription.py <audio_file> [--model <model_name>] [--lyrics <lyrics_file>] [...]
```
-   **`audio_file`**: Path to the audio file to transcribe (e.g., `vocals.mp3`).
-   **`--model`**: The transcription model to use.
    -   `gemini`: (Default) Uses the Gemini API directly. Fast but may be less accurate for complex songs.
    -   `whisper`: Uses a local `whisper-timestamped` model for transcription.
    -   `sieve`: Uses the Sieve platform for Whisper transcription.
    -   `whisper-gemini`: **(Recommended)** A powerful pipeline that gets a base transcription from Sieve/Whisper and uses Gemini to correct it against the provided lyrics file. Requires `--lyrics`.
-   **`--lyrics`**: Path to a text file with lyrics to guide transcription.
-   **`--output`**: Path to the output JSON file.

### `creator.py`

Generates `.ass` subtitle files and videos from a transcription JSON.

```bash
python creator.py <input_json> [--output <output.ass>] [--translate] [--instrumental-karaoke] [--generate-video]
```
-   **`input_json`**: Path to the structured input JSON file.
-   **`--translate`**: Translate sentences that don't have a translation (requires Gemini API key).
-   **`--update-json`**: Saves translations back into the input JSON file.
-   **`--instrumental-karaoke`**: Generates a second `.ass` file with larger, multi-line subtitles suitable for a video with no original vocals.
-   **`--generate-video`**: Creates an MP4 video by combining the instrumental `.ass` file with the `instrumental.mp3` audio in the same directory. Requires `--instrumental-karaoke`.