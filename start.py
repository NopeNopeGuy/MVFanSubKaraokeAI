                      
import sys
import os
import yt_dlp
import requests
import re
import lyricsgenius
import argparse
import sieve
import shutil

def get_youtube_title(url):
    """Fetch only the title from a YouTube URL without downloading the video."""
    ydl_opts = {'quiet': True, 'no_warnings': True, 'skip_download': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return info.get('title', 'youtube_audio')
        except yt_dlp.utils.DownloadError as e:
            print(f"Error fetching YouTube metadata: {e}")
            sys.exit(1)

def download_youtube(url, output_path):
    """Download audio from YouTube as an MP3, only if the output file doesn't exist."""
    if os.path.exists(output_path):
        print(f"Audio file already exists: {output_path}")
        return

    print(f"Downloading audio to {output_path}...")
                                                  
    out_template = os.path.splitext(output_path)[0]

    ydl_opts = {
        'outtmpl': f'{out_template}.%(ext)s',
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Download complete.")

def make_safe_filename(name):
    """Cleans a string to be a safe directory/file name."""
                                                                                     
    name = re.sub(r'\[.*?\]|\(.*?\)', '', name)
                                        
    name = re.sub(r'(?i)\b(official|full|video|mv|music|lyrics?|audio|hd|4k|remix|feat\.?|ft\.?|prod\.?|\|)\b', '', name)
                                                      
    name = re.sub(r'[^\w\s-]', '', name).strip()
                                         
    name = re.sub(r'\s+', '_', name)
    return name if name else "audio_track"

def fetch_lyrics(query, output_path):
    """Fetch lyrics using lyricsgenius and save to a file."""
    if os.path.exists(output_path):
        print(f"Lyrics file already exists: {output_path}")
        return

    print(f"Fetching lyrics for '{query}'...")
    api_token = os.environ.get('GENIUS_API_TOKEN')
    if not api_token:
        print("Warning: GENIUS_API_TOKEN environment variable not set. Cannot fetch lyrics.")
        return

    genius = lyricsgenius.Genius(api_token, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True, timeout=15, retries=2)
    try:
        song = genius.search_song(query)
        if song and song.lyrics:
                                              
            lyrics = re.sub(r'\d*Embed$', '', song.lyrics).strip()
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(lyrics)
            print(f"Lyrics saved to {output_path}")
        else:
            print("Lyrics not found on Genius.")
    except Exception as e:
        print(f"Error fetching lyrics: {e}")

def run_demucs(input_audio_path, output_vocals_path, output_instrumental_path):
    """Use the sieve Python library to run Demucs and save the vocals and instrumental stems."""
    if os.path.exists(output_vocals_path) and os.path.exists(output_instrumental_path):
        print(f"Vocals and instrumental files already exist: {output_vocals_path}, {output_instrumental_path}")
        return

    print("Running Demucs to separate vocals and instrumental...")
    try:
        file = sieve.File(path=input_audio_path)
        demucs = sieve.function.get("sieve/demucs")
                                                    
        output = demucs.run(
            file=file,
            model="htdemucs_ft",
            two_stems="vocals",
            audio_format="mp3",
     shifts=1,
            overlap=0.25
        )

                                                                                      
        if not isinstance(output, tuple) or len(output) < 2:
            print("Error: Unexpected output format from demucs.run. Expected two files (vocals, instrumental).")
                                             
            if isinstance(output, tuple) and len(output) == 1:
                output = (output[0], None)
            elif hasattr(output, 'path'):
                output = (output, None)
            else:
                return

        vocals_file, instrumental_file = output[0], output[1]

        def download_sieve_file(sieve_file, local_path):
            """Helper to download a file from a sieve.File object."""
            if not sieve_file or not hasattr(sieve_file, 'path'):
                stem_name = os.path.basename(local_path).split('.')[0]
                print(f"Error: No valid output object for {stem_name} received from Sieve.")
                return

            source_path = sieve_file.path
            print(f"Processing {os.path.basename(local_path)} from: {source_path}")
            
            if source_path.startswith('http://') or source_path.startswith('https://'):
                print(f"Downloading to {local_path}...")
                with requests.get(source_path, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"Saved {os.path.basename(local_path)}")
            elif os.path.exists(source_path):
                print(f"Copying local file to {local_path}...")
                shutil.copy(source_path, local_path)
                print(f"Saved {os.path.basename(local_path)}")
            else:
                print(f"Error: Sieve output path '{source_path}' is not a valid URL or local file.")

                                             
        if not os.path.exists(output_vocals_path):
            download_sieve_file(vocals_file, output_vocals_path)
        else:
            print(f"Vocals file already exists: {output_vocals_path}")

                                                   
        if not os.path.exists(output_instrumental_path):
            download_sieve_file(instrumental_file, output_instrumental_path)
        else:
            print(f"Instrumental file already exists: {output_instrumental_path}")

    except Exception as e:
        print(f"An error occurred while running Demucs via Sieve: {e}")
        print("Please ensure you have run 'sieve login' and have the necessary permissions.")

def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube audio, fetch lyrics, and separate vocals into a dedicated subdirectory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("youtube_url", help="YouTube video URL")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom name for the song. This will be used for the directory name and lyrics search.\n"
             "If not provided, the YouTube video title will be used."
    )
    args = parser.parse_args()

                                          
    if args.name:
        song_title_for_lyrics = args.name
        safe_dir_name = make_safe_filename(args.name)
    else:
        print("Fetching video title from YouTube...")
        youtube_title = get_youtube_title(args.youtube_url)
        print(f"YouTube Title: {youtube_title}")
        song_title_for_lyrics = youtube_title
        safe_dir_name = make_safe_filename(youtube_title)

    output_dir = safe_dir_name
    os.makedirs(output_dir, exist_ok=True)
    print(f"All files will be saved in: {output_dir}/")

    audio_path = os.path.join(output_dir, "song.mp3")
    lyrics_path = os.path.join(output_dir, "lyrics.txt")
    vocals_path = os.path.join(output_dir, "vocals.mp3")
    instrumental_path = os.path.join(output_dir, "instrumental.mp3")

                               
    download_youtube(args.youtube_url, audio_path)

                             
    fetch_lyrics(song_title_for_lyrics, lyrics_path)

                                             
    run_demucs(audio_path, vocals_path, instrumental_path)

    print("\nProcess complete.")
    print(f"Directory: {output_dir}")
    print(f"  - Song:   {os.path.basename(audio_path)}")
    print(f"  - Lyrics: {os.path.basename(lyrics_path)}")
    print(f"  - Vocals: {os.path.basename(vocals_path)}")
    print(f"  - Instrumental: {os.path.basename(instrumental_path)}")

if __name__ == "__main__":
    if not os.environ.get('GENIUS_API_TOKEN'):
        print("Info: GENIUS_API_TOKEN environment variable is not set. Lyrics fetching will be skipped.")
    main()
