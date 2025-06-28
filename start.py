#!/usr/bin/env python3
import sys
import os
import subprocess
import yt_dlp
import requests
import re
from bs4 import BeautifulSoup
import lyricsgenius
import argparse
import time
import sieve
import shutil

# Optional: Use lyricsgenius or a simple web scraper for lyrics
# import lyricsgenius

def download_youtube(url, out_base):
    """Download audio from YouTube using yt-dlp, only if not already present."""
    wav_path = f"{out_base}.wav"
    title = None
    # Download audio (wav)
    if not os.path.exists(wav_path):
        ydl_opts = {
            'outtmpl': f'{out_base}.%(ext)s',
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'audio')
    else:
        # Get title from info json if present, else fallback
        ydl_opts = {'quiet': True, 'no_warnings': True, 'skip_download': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'audio')
    return title, wav_path

def clean_youtube_title(title):
    """Clean up YouTube video titles for better lyrics search."""
    # Remove content in brackets/parentheses (e.g., [Official Video], (Lyrics), etc.)
    title = re.sub(r'\[.*?\]|\(.*?\)', '', title)
    # Remove common YouTube tags/phrases
    title = re.sub(r'(?i)(official|full|video|mv|music|lyrics?|audio|HD|4K|remix|feat\.?|ft\.?|prod\.?|\|)', '', title)
    # Remove extra whitespace and non-alphanumeric at ends
    title = re.sub(r'^[^\w]+|[^\w]+$', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def fetch_lyrics(title):
    """Fetch lyrics for a single song using lyricsgenius."""
    api_token = os.environ.get('GENIUS_API_TOKEN')
    if not api_token:
        print("GENIUS_API_TOKEN environment variable not set.")
        return "Lyrics not found."
    genius = lyricsgenius.Genius(api_token, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)", "(Edit)"], remove_section_headers=True, timeout=10, retries=2)
    cleaned_title = clean_youtube_title(title)
    try:
        song = genius.search_song(cleaned_title)
        if song and song.lyrics:
            return song.lyrics
        else:
            return "Lyrics not found."
    except Exception as e:
        print(f"Error fetching lyrics: {e}")
        return "Lyrics not found."

def run_demucs(wav_path, song_name):
    """Use the sieve Python library to run Demucs and download vocals stem as ./{song_name}_vocals/song.mp3."""
    vocals_dir = f"{song_name}_vocals"
    vocals_path = f"{vocals_dir}/song.mp3"
    
    # Create vocals directory if it doesn't exist
    os.makedirs(vocals_dir, exist_ok=True)
    
    if os.path.exists(vocals_path):
        print(f"Demucs vocals output already exists: {vocals_path}")
        return
    print("Running Demucs using sieve Python library...")
    # Prepare Sieve File
    file = sieve.File(path=wav_path)
    model = "htdemucs_ft"
    two_stems = "vocals"
    overlap = 0.75
    shifts = 2
    audio_format = "mp3"
    demucs = sieve.function.get("sieve/demucs")
    output = demucs.run(
        file=file,
        model=model,
        two_stems=two_stems,
        overlap=overlap,
        shifts=shifts,
        audio_format=audio_format
    )
    print(f"Demucs separation complete. Output type: {type(output)}, value: {output}")
    # Output is a tuple of sieve.File objects: (vocals_file, accompaniment_file)
    if not isinstance(output, tuple) or len(output) != 2:
        print("Error: Unexpected output format from demucs.run.")
        return
    vocals_file = output[0]
    if not hasattr(vocals_file, 'path'):
        print("Error: Output object for vocals does not have a .path attribute.")
        return
    print(f"Processing vocals from {vocals_file.path}")
    if os.path.exists(vocals_file.path):
        shutil.copy(vocals_file.path, vocals_path)
        print(f"Copied vocals to {vocals_path}")
    elif vocals_file.path.startswith('http://') or vocals_file.path.startswith('https://'):
        print(f"Downloading vocals from {vocals_file.path}")
        with requests.get(vocals_file.path, stream=True) as s:
            s.raise_for_status()
            with open(vocals_path, 'wb') as f:
                for chunk in s.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Saved vocals to {vocals_path}")
    else:
        print(f"Error: {vocals_file.path} is not a valid file or URL for vocals.")
    print("Demucs separation complete.")

def main():
    parser = argparse.ArgumentParser(description="Download YouTube audio, fetch lyrics, and run demucs.")
    parser.add_argument("youtube_url", help="YouTube video URL")
    parser.add_argument("--name", type=str, default=None, help="Song name to use for lyrics search (overrides YouTube title)")
    args = parser.parse_args()

    url = args.youtube_url
    # Download to a temp base name first
    temp_base = "yt_audio"
    print("Downloading audio (if not already present)...")
    title, wav_path = download_youtube(url, temp_base)
    print(f"Audio: {wav_path}")
    print(f"Extracted title: {title}")

    # Use --name if provided, else fallback to YouTube title
    lyrics_query = args.name if args.name else title
    # Clean the title for filenames
    cleaned_title = clean_youtube_title(lyrics_query)
    safe_title = re.sub(r'[^\\w\- ]', '', cleaned_title).strip().replace(' ', '_')
    if not safe_title:
        safe_title = "audio"
    
    # New output filenames with the new naming convention
    new_wav = f"{safe_title}_vocals/song.wav"
    lyrics_path = f"{safe_title}_vocals/lyrics.txt"
    
    # Create vocals directory
    vocals_dir = f"{safe_title}_vocals"
    os.makedirs(vocals_dir, exist_ok=True)
    
    # Rename files if needed
    if os.path.exists(wav_path) and wav_path != new_wav:
        os.rename(wav_path, new_wav)
        wav_path = new_wav
    
    # Lyrics
    if not os.path.exists(lyrics_path):
        print("Fetching lyrics from Genius...")
        lyrics = fetch_lyrics(lyrics_query)
        if lyrics.strip() == "Lyrics not found.":
            print("Error: Lyrics not found. Exiting.")
            sys.exit(1)
        with open(lyrics_path, "w", encoding="utf-8") as f:
            f.write(lyrics)
        print(f"Lyrics saved to {lyrics_path}")
    else:
        print(f"{lyrics_path} already exists, skipping lyrics fetch.")
    
    # Demucs
    print("Running demucs (if not already processed)...")
    run_demucs(wav_path, safe_title)
    print("Done.")

if __name__ == "__main__":
    main() 