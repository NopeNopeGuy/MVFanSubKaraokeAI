#!/usr/bin/env python3
"""
Audio Syllable/Character Transcription Script
Uses the Gemini model for transcription, with audio speed adjustment and
specific MM:SS:ss timestamp formatting.
"""

import argparse
import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import abc
import re

# --- Dependency Management ---
# Check for Gemini dependencies
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: Missing required packages for the Gemini model. Install with:")
    print("pip install google-genai")
    sys.exit(1)

# Check for audio processing dependencies (pydub is now required)
try:
    from pydub import AudioSegment
except ImportError:
    print("Error: Missing required 'pydub' library for audio processing.")
    print("Please install it with: pip install pydub")
    print("\nYou may also need to install ffmpeg on your system:")
    print("On Ubuntu/Debian: sudo apt-get install ffmpeg")
    print("On macOS (with Homebrew): brew install ffmpeg")
    print("On Windows: Download from https://ffmpeg.org/download.html")
    sys.exit(1)


# --- Abstract Base Class for Transcribers ---

class Transcriber(abc.ABC):
    """Abstract base class for all transcription engines."""

    @abc.abstractmethod
    def transcribe_audio(self, audio_file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribes an audio file and returns a dictionary with the result.
        The 'transcription' key should contain a JSON string.
        """
        pass

    def read_lyrics_from_file(self, lyrics_file_path: str) -> Optional[str]:
        """Reads lyrics from a specified text file."""
        try:
            lyrics_path = Path(lyrics_file_path)
            if not lyrics_path.exists():
                print(f"Warning: Lyrics file not found at {lyrics_file_path}")
                return None
            return lyrics_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error reading lyrics file: {e}")
            return None

    def save_transcription(self, transcription: str, output_file: str) -> bool:
        """Save transcription to a minified JSON file."""
        try:
            parsed_json = json.loads(transcription)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, separators=(',', ':'), ensure_ascii=False)
            return True
        except json.JSONDecodeError:
            print(f"Warning: Model output was not valid JSON. Saving as plain text.")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcription)
            return False
        except Exception as e:
            print(f"Error saving transcription: {e}")
            return False


# --- Gemini API Transcriber (With Audio Speed Adjustment and MM:SS:ss format) ---

class GeminiTranscriber(Transcriber):
    def __init__(self, api_key: str, base_url: str = None):
        """Initialize the transcriber with API credentials"""
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-pro"  
        
    def _mmss_to_seconds(self, mmss_str: str) -> float:
        """Converts an MM:SS:ss string to total seconds."""
        try:
            parts = mmss_str.split(':')
            if len(parts) != 3:
                raise ValueError("Expected 3 parts for MM:SS:ss")
            minutes = int(parts[0])
            seconds = int(parts[1])
            centiseconds = int(parts[2])
            return minutes * 60 + seconds + centiseconds / 100.0
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid timestamp format: '{mmss_str}'. Expected MM:SS:ss. Error: {e}")

    def _seconds_to_mmss(self, total_seconds: float) -> str:
        """Converts total seconds to an MM:SS:ss string."""
        if total_seconds < 0:
            total_seconds = 0
        minutes, rem_seconds = divmod(total_seconds, 60)
        seconds, centi_frac = divmod(rem_seconds, 1)
        centiseconds = round(centi_frac * 100)
        
        if centiseconds >= 100:
            seconds += 1
            centiseconds = 0
        if seconds >= 60:
            minutes += 1
            seconds = 0
            
        return f"{int(minutes):02d}:{int(seconds):02d}:{int(centiseconds):02d}"

    def get_audio_duration_str(self, audio_file_path: str) -> str:
        """Gets audio duration using pydub and formats it as MM:SS:ss."""
        try:
            audio = AudioSegment.from_file(audio_file_path)
            duration_seconds = audio.duration_seconds
            return self._seconds_to_mmss(duration_seconds)
        except Exception as e:
            raise Exception(f"Failed to get audio duration using pydub: {e}")

    # --- THIS IS THE CORRECTED FUNCTION ---
    def slow_down_audio(self, audio_file_path: str, speed_factor: float, preserve_pitch: bool = True) -> str:
        """
        Slow down audio and return path to temporary file.
        Uses a pitch-preserving method ('atempo' filter) by default.
        """
        if not 0 < speed_factor < 1:
            raise ValueError("Speed factor must be between 0 and 1 (exclusive) to slow down audio.")

        # FFmpeg's atempo filter has a range of [0.5, 100.0].
        if preserve_pitch and speed_factor < 0.5:
            print(f"Warning: Pitch-preserving slowdown with 'atempo' is most effective for speed factors >= 0.5. "
                  f"Your factor is {speed_factor}. Results may vary as filter chaining is not implemented.")

        print(f"Slowing down audio to {speed_factor}x speed (Pitch preserved: {preserve_pitch})...")
        try:
            audio = AudioSegment.from_file(audio_file_path)
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file_path).suffix)
            temp_path = temp_file.name
            temp_file.close()
            
            output_format = Path(audio_file_path).suffix.lstrip('.')

            if preserve_pitch:
                # CORRECT METHOD: Use the 'parameters' argument during export.
                audio.export(
                    temp_path,
                    format=output_format,
                    parameters=["-filter:a", f"atempo={speed_factor}"]
                )
            else:
                # The original, pitch-altering method.
                slowed_audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * speed_factor)
                }).set_frame_rate(audio.frame_rate)
                slowed_audio.export(temp_path, format=output_format)

            print(f"✓ Slowed audio saved to temporary file: {temp_path}")
            return temp_path
        except Exception as e:
            if "No such file or directory" in str(e) or "not found" in str(e):
                 raise Exception(f"Failed to slow down audio. This often means FFmpeg is not installed or not in your system's PATH. Original error: {e}")
            raise Exception(f"Failed to slow down audio: {e}")

    def adjust_timestamps_for_speed(self, transcription_json: str, speed_factor: float) -> str:
        """Adjust all MM:SS:ss timestamps in the transcription to account for the speed change."""
        try:
            data = json.loads(transcription_json)
            print(f"Adjusting timestamps by factor of {speed_factor:.3f} to restore original timing...")
            
            def update_times_recursive(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key in ("start_time", "end_time") and isinstance(value, str):
                            try:
                                original_seconds = self._mmss_to_seconds(value)
                                adjusted_seconds = original_seconds * speed_factor
                                obj[key] = self._seconds_to_mmss(adjusted_seconds)
                            except ValueError:
                                pass
                        else:
                            update_times_recursive(value)
                elif isinstance(obj, list):
                    for item in obj:
                        update_times_recursive(item)

            update_times_recursive(data)
            adjusted_json = json.dumps(data, ensure_ascii=False)
            return adjusted_json
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse transcription as JSON for timestamp adjustment: {e}")
            return transcription_json
        except Exception as e:
            print(f"Warning: Error adjusting timestamps: {e}")
            return transcription_json

    def create_system_prompt(self, audio_duration_str: str, lyrics: Optional[str] = None) -> str:
        """Create the system prompt for audio transcription using MM:SS:ss format."""
        base_prompt = f"""You are an advanced audio transcription AI specialized in converting speech/song audio into syllable-by-syllable JSON output.

## Core Requirements

### 1. Audio & Timing Constraints
- **Total Audio Duration:** The audio file is exactly `{audio_duration_str}` long (format is MM:SS:ss).
- **CRITICAL:** The `end_time` of the very last syllable in the entire transcription **MUST NOT** exceed `{audio_duration_str}`.

### 2. Syllable Segmentation & Timing
- Process audio input and identify individual syllables in the speech.
- **All timestamps MUST be strings in "MM:SS:ss" format** (Minutes:Seconds:Centiseconds). For example, `01:23:45` represents 1 minute, 23 seconds, and 45 hundredths of a second.
- Segment each syllable with an accurate `start_time` and `end_time`.
- Timestamps must be sequential and non-overlapping.
- The sentence-level `start_time` must equal the `start_time` of its first syllable.
- The sentence-level `end_time` must equal the `end_time` of its last syllable.
- The sentences should contain more than one letter. TRANSCRIBE SENTENCES WHERE YOU ARE SUPPOSED TO, AND TRANSCRIBE SYLLABLES WHERE YOU ARE SUPPOSED TO.
- IT IS COMPLETELY FINE AND OKAY AND EXPECTED TO HAVE LONG SYLLABLES SINCE IT'S A SONG.
- DO NOT HAVE SUPER SHORT SYLLABLES.
- THE SYLLABLES SHOULD BE LIKE FANSUBS.
- ALWAYS FOLLOW MM:SS:ss format. NO MATTTER WHAT.
- Centiseconds shouldn't be super small
- THE SYLLABLES SHOULD ALSO WHEN JOINED TOGETHER FORM THE ORIGINAL SENTENCE. BE ABSOLUTELY PERFECT IN THAT.
- DO NOT FORGET THE LAST SYLLABLE

### 3. Romanization Standards
- Convert all non-Latin script languages into romanized (Latin/English script) equivalents.
- Use standard romanization systems (e.g., Hepburn for Japanese, Pinyin for Chinese, etc.).
- THE SENTENCE TEXT MUST ALSO BE ROMANIZED. NOTHING SHOULD HAVE THEIR ORIGINAL SCRIPT.
- EVEN THE LETTER TEXT MUST ALSO BE ROMANIZED.
- NOTHING SHOULD HAVE THEIR ORIGINAL SCRIPT. NOTHING.
- EVERYTHING MUST BE ROMANIZED.
- THE SYLLABLES SHOULD ALSO WHEN JOINED TOGETHER FORM THE ORIGINAL SENTENCE.
- IF THE LANGUAGE HAS LOAN WORDS FROM ENGLISH. CONVERT THEM INTO THE ENGLISH WORD. EG: jiguzagu to zigzag, sutōbu to stove.
- THE LOAN WORDS OUTPUTTED SHOULD MAKE SENSE TO A ENGLISH AUDIENCE IF THE LOAN WORDS ARE BORROWED FROM ENGLISH.
- DO NOT MISS EVEN A SINGLE WORD
- DO NOT FORGET THE LAST SYLLABLE

## 4. CoT
- ALWAYS DOUBLE CHECK THE TIMING OF THE SYLLABLES.
- ALWAYS DOUBLE CHECK THE TIMING OF THE SENTENCES.
- YOU ARE TO NEVER BE WRONG
- YOU WILL CHECK EVERYTHING TWICE
- YOU ARE TO BE ACCURATE NO MATTER WHAT.
- HAVE PROPER START AND END TIMINGS.
- HAVE AS MANY SENTENCES AS POSSIBLE. AS LONG AS IT MAKES SENSE.
- THE SYLLABLES SHOULD ALSO WHEN JOINED TOGETHER FORM THE ORIGINAL SENTENCE. BE ABSOLUTELY PERFECT IN THAT.
- DO NOT MISS EVEN A SINGLE WORD

## Output Format

Provide output in the following JSON structure. **Pay close attention to the "MM:SS:ss" timing format.** Do not include the `duration` key. THERE SHOULD BE NO MARKDOWN.


[
  {{
    "sentence": {{
      "text": "complete sentence text romanized.",
      "start_time": "00:00:00",
      "end_time": "00:01:00",
      "letters": [
        {{
          "text": "syl", # romanized
          "start_time": "00:00:00",
          "end_time": "00:00:50"
        }},
        {{
          "text": "la", # romanized
          "start_time": "00:00:50",
          "end_time": "00:00:80"
        }},
        {{
          "text": "ble", # romanized 
          "start_time": "00:00:80",
          "end_time": "00:01:00"
        }}
      ]
    }}
  }}
]

"""
        if lyrics:
            base_prompt += f"\n\n## Reference Lyrics\nFor context, here are the expected lyrics for this audio:\n\n{lyrics}\n\nUse these lyrics as a reference to improve transcription accuracy, respect the timing of the spoken words."
        return base_prompt

    def filter_markdown(self, text: str) -> str:
        """Filter out markdown code blocks from the model response."""
        # Remove markdown code blocks (```json, ```, etc.)
        # This handles cases like ```json\ncontent\n``` or just ```\ncontent\n```
        text = re.sub(r'^```\w*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)
        
        # Also handle cases where there might be multiple code blocks
        text = re.sub(r'```\w*\n', '', text)
        text = re.sub(r'\n```', '', text)
        
        return text.strip()

    def transcribe_audio(self, audio_file_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio file using the Gemini model with optional speed adjustment"""
        lyrics = kwargs.get("lyrics")
        speed_factor = kwargs.get("speed_factor", 1.0)
        preserve_pitch = kwargs.get("preserve_pitch", True)
        
        temp_audio_path = None
        
        try:
            if not Path(audio_file_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

            audio_to_transcribe = audio_file_path
            
            if speed_factor != 1.0:
                temp_audio_path = self.slow_down_audio(audio_file_path, speed_factor, preserve_pitch)
                audio_to_transcribe = temp_audio_path

            duration_for_prompt = self.get_audio_duration_str(audio_to_transcribe)
            print(f"✓ Audio duration for model prompt: {duration_for_prompt}")
            system_prompt = self.create_system_prompt(duration_for_prompt, lyrics)

            print("Sending request to Gemini API via google-genai SDK...")

            # Upload the audio file using the Files API (see: https://ai.google.dev/gemini-api/docs/audio)
            myfile = self.client.files.upload(file=audio_to_transcribe)

            # Prepare the prompt and contents for Gemini API
            prompt = "Please transcribe this audio file into syllable-by-syllable JSON format as specified."
            contents = [prompt, myfile]

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=1.0,
                    max_output_tokens=65536,
                    system_instruction=system_prompt
                )
            )

            transcription = response.text
            
            # Filter out markdown code blocks
            transcription = self.filter_markdown(transcription)

            if speed_factor != 1.0:
                transcription = self.adjust_timestamps_for_speed(transcription, speed_factor)
            
            return {
                "success": True,
                "transcription": transcription,
                "model": self.model_name,
                "usage": None,
                "speed_factor_used": speed_factor
            }
        except Exception as e:
            print(f"API call failed: {e}.")
            return {"success": False, "error": str(e)}
        finally:
            if temp_audio_path and Path(temp_audio_path).exists():
                try:
                    os.unlink(temp_audio_path)
                    print(f"✓ Cleaned up temporary audio file: {temp_audio_path}")
                except Exception as e:
                    print(f"Warning: Could not clean up temporary file {temp_audio_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio to syllable-level JSON using the Gemini model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("--output", "-o", help="Output JSON file path", default="transcription.json")

    gemini_group = parser.add_argument_group('Gemini API Options')
    gemini_group.add_argument("--lyrics", help="Path to a text file with lyrics to guide transcription", default=None)
    gemini_group.add_argument("--api-key", help="API key for Gemini (or set GEMINI_API_KEY env var)")
    gemini_group.add_argument("--base-url", help="API base URL", default="https://generativelanguage.googleapis.com/v1beta/openai/")
    gemini_group.add_argument("--speed", 
                              help="Audio speed factor for transcription (e.g., 0.7 for 70%% speed). "
                                   "Values < 1 slow down audio, potentially improving accuracy.", 
                              type=float, default=1.0)
    gemini_group.add_argument("--no-preserve-pitch", 
                              action="store_true", 
                              help="Use the faster, pitch-altering method for slowing down audio instead of the default 'atempo' filter.")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: API key required for Gemini. Use --api-key or set GEMINI_API_KEY.")
        sys.exit(1)
    
    if args.speed <= 0:
        print("Error: Speed factor must be greater than 0.")
        sys.exit(1)
    
    transcriber = GeminiTranscriber(api_key, args.base_url)
    transcribe_kwargs = {
        'speed_factor': args.speed,
        'preserve_pitch': not args.no_preserve_pitch
    }
    
    if args.lyrics:
        print(f"Reading lyrics from: {args.lyrics}")
        transcribe_kwargs['lyrics'] = transcriber.read_lyrics_from_file(args.lyrics)
        if transcribe_kwargs['lyrics']:
            print("✓ Lyrics loaded successfully")

    print(f"\nTranscribing '{args.audio_file}' using Gemini model ({transcriber.model_name})...")
    if args.speed != 1.0:
        print(f"Using speed factor: {args.speed}x")
    
    result = transcriber.transcribe_audio(args.audio_file, **transcribe_kwargs)

    if result.get("success"):
        print("\n✓ Transcription completed successfully")
        
        if transcriber.save_transcription(result["transcription"], args.output):
            print(f"✓ Transcription saved to: {args.output}")
        else:
            print(f"⚠ Transcription saved with warnings to: {args.output}")

        if result.get("usage"):
            print(f"Usage: {result['usage']}")
        
        if result.get("speed_factor_used", 1.0) != 1.0:
            print(f"Speed factor used: {result['speed_factor_used']}x (timestamps adjusted back to original speed)")

        try:
            preview = json.loads(result["transcription"])
            print("\n--- Transcription Preview (from " + result.get('model', 'N/A') + ") ---")
            preview_text = json.dumps(preview, indent=2, ensure_ascii=False)
            print(preview_text[:500] + ("..." if len(preview_text) > 500 else ""))
        except (json.JSONDecodeError, KeyError):
            print("\n--- Raw Response (from " + result.get('model', 'N/A') + ") ---")
            raw_text = result.get("transcription", "")
            print(raw_text[:500] + ("..." if len(raw_text) > 500 else ""))
    else:
        print(f"\n✗ Transcription failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
