                      
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
from typing import Dict, Any, Optional, List, Tuple
import abc
import re
from pydantic import BaseModel, Field
import sieve                              

                               
                               
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: Missing required packages for the Gemini model. Install with:")
    print("pip install google-genai")
    sys.exit(1)

                                                                 
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

                                            
try:
    import whisper_timestamped as whisper
    import torch
except ImportError:
    print("Error: Missing required packages for the Whisper-timestamped model. Install with:")
    print("pip install whisper-timestamped torch torchaudio")
    print("For GPU support, ensure you install the correct torch version (e.g., pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118)")
    sys.exit(1)

                                 
try:
    from pykakasi import kakasi
except ImportError:
    print("Warning: 'pykakasi' library not found. Romanization for Whisper will fall back to Gemini if API key is provided.")
    kakasi = None                                      


                                              

class Transcriber(abc.ABC):
    """Abstract base class for all transcription engines."""

    @abc.abstractmethod
    def transcribe_audio(self, audio_file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribes an audio file and returns a dictionary with the result.
        The 'transcription' key should contain a JSON string.
        """
        pass

    def _mmss_to_seconds(self, mmss_str: str) -> float:
        """Converts an MM:SS:sss string to total seconds."""
        try:
            parts = mmss_str.split(':')
            if len(parts) != 3:
                raise ValueError("Expected 3 parts for MM:SS:sss")
            minutes = int(parts[0])
            seconds = int(parts[1])
            milliseconds = int(parts[2])
            return minutes * 60 + seconds + milliseconds / 1000.0
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid timestamp format: '{mmss_str}'. Expected MM:SS:sss. Error: {e}")

    def _seconds_to_mmss(self, total_seconds: float) -> str:
        """Converts total seconds to an MM:SS:sss string."""
        if total_seconds < 0:
            total_seconds = 0
        minutes, rem_seconds = divmod(total_seconds, 60)
        seconds, milli_frac = divmod(rem_seconds, 1)
        milliseconds = round(milli_frac * 1000)
        
        if milliseconds >= 1000:
            seconds += 1
            milliseconds = 0
        if seconds >= 60:
            minutes += 1
            seconds = 0
            
        return f"{int(minutes):02d}:{int(seconds):02d}:{int(milliseconds):03d}"

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

    def _validate_and_correct_transcription(self, json_string: str) -> str:
        """
        Validates the transcription JSON against the Pydantic schema.
        If validation fails, it attempts to correct common structural errors from the model
        and re-validates.
        """
        try:
                                                         
            TranscriptionResponse.model_validate_json(json_string)
            print("✓ Transcription structure validated successfully.")
            return json_string
        except Exception as e:
            print(f"Warning: Initial validation of model output failed. Attempting to correct... Error: {e}")
            try:
                data = json.loads(json_string)
                if "sentences" not in data:
                                                                               
                    raise ValueError("JSON missing 'sentences' key.")

                for sentence in data.get("sentences", []):
                    if "words" not in sentence:
                        continue
                    
                    corrected_words = []
                    for word_list in sentence.get("words", []):
                        if not isinstance(word_list, list) or len(word_list) < 3:
                            corrected_words.append(word_list)
                            continue

                                                                               
                        word, start, end = word_list[:3]
                        syllables_data = word_list[3:]
                        
                        syllables = []
                        i = 0
                        while i < len(syllables_data):
                            item = syllables_data[i]
                                                             
                            if isinstance(item, list) and len(item) == 3:
                                syllables.append(item)
                                i += 1
                                                                             
                            elif i + 2 < len(syllables_data) and all(isinstance(syllables_data[i+j], str) for j in range(3)):
                                syllables.append([syllables_data[i], syllables_data[i+1], syllables_data[i+2]])
                                i += 3
                            else:
                                                                              
                                i += 1
                        
                        corrected_words.append([word, start, end, syllables])
                    sentence["words"] = corrected_words
                
                corrected_json = json.dumps(data, ensure_ascii=False)
                
                                                                                             
                TranscriptionResponse.model_validate_json(corrected_json)
                
                print("✓ Successfully corrected and re-validated transcription structure.")
                return corrected_json

            except Exception as final_e:
                print(f"Error: Automatic correction failed. The output may be malformed. Error: {final_e}")
                                                                                         
                return json_string

    def filter_markdown(self, text: str) -> str:
        """Filter out markdown code blocks from the model response."""
                                                          
                                                                                 
        text = re.sub(r'^```\w*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)
        
                                                                     
        text = re.sub(r'```\w*\n', '', text)
        text = re.sub(r'\n```', '', text)
        
        return text.strip()


                                                       

                                                                      
                                                             
SyllableFormat = Tuple[str, str, str]

                                                                          
WordFormat = Tuple[str, str, str, List[SyllableFormat]]

class Sentence(BaseModel):
    text: str = Field(..., description="The full romanized text of the lyrical sentence.")
    translation: str = Field(..., description="The English translation of the sentence.")
    words: List[WordFormat] = Field(
        ...,
        description="A list of words. Each word is a JSON list: [word_text, start_time, end_time, [syllables_list]]. Each syllable is: [syllable_text, start_time, end_time]."
    )

class TranscriptionResponse(BaseModel):
    sentences: List[Sentence]

                                                

class RomanizationRequest(BaseModel):
    texts: List[str]

class RomanizationResponse(BaseModel):
    romanized_texts: List[str]

class TranslationResponse(BaseModel):
    translations: List[str]


class GeminiTranscriber(Transcriber):
    def __init__(self, api_key: str, base_url: str = None):
        """Initialize the transcriber with API credentials"""
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"
        
    def get_audio_duration_str(self, audio_file_path: str) -> str:
        """Gets audio duration using pydub and formats it as MM:SS:ss."""
        try:
            audio = AudioSegment.from_file(audio_file_path)
            duration_seconds = audio.duration_seconds
            return self._seconds_to_mmss(duration_seconds)
        except Exception as e:
            raise Exception(f"Failed to get audio duration using pydub: {e}")

                                            
    def slow_down_audio(self, audio_file_path: str, speed_factor: float, preserve_pitch: bool = True) -> str:
        """
        Slow down audio and return path to temporary file.
        Uses a pitch-preserving method ('atempo' filter) by default.
        """
        if not 0 < speed_factor < 1:
            raise ValueError("Speed factor must be between 0 and 1 (exclusive) to slow down audio.")

                                                             
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
                                                                              
                audio.export(
                    temp_path,
                    format=output_format,
                    parameters=["-filter:a", f"atempo={speed_factor}"]
                )
            else:
                                                      
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
        """Create the system prompt for audio transcription using the new list-based JSON format."""
        base_prompt = f"""### **1. Persona**

You are **Lyra**, a world-class AI system specializing in rhythmic transcription for karaoke. Your core mission is to convert the sung lyrics in an audio file into a precise, multi-layered JSON format. You are renowned for your meticulous accuracy, adherence to musical timing, and your ability to create a natural, enjoyable karaoke experience.

### **2. Task**

Your primary task is to process a given audio file and generate a single, raw JSON object. This object will contain a key `sentences`. You will transcribe the sung lyrics, romanize all non-Latin text, and provide an English translation for each sentence. Crucially, you will break down each sentence into `words`, and finally `syllables`, each with a hyper-accurate start and end timestamp, formatted as nested JSON lists.

### **3. Context & Constraints**

**A. Audio & Timing Constraints (The Ground Truth)**
*   The provided audio file is exactly `{audio_duration_str}` long (format is MM:SS:sss).
*   **Primary Directive:** The `end_time` of the very last syllable in the entire transcription **must not** exceed the total audio duration of `{audio_duration_str}`.
*   **Timestamp Format:** ALL timestamps **must** be strings in the exact `"MM:SS:sss"` format (Minutes:Seconds:Milliseconds).
*   **Sequential Integrity:** Timestamps must be sequential and logical. The `start_time` of any element must be greater than or equal to the `end_time` of the preceding element at the same level.
*   **No Zero-Duration Elements:** For every syllable and word, the start time must be strictly less than its end time.
*   **Readable Syllable Duration:** To ensure a readable karaoke experience, a syllable's duration (`end_time` - `start_time`) must not be impractically short. **Strictly enforce a minimum duration of 70 milliseconds (0.070 seconds).** If a phonetic sound is faster, merge it with an adjacent syllable.

**B. Structural & Content Integrity (The Core Logic)**
*   **Data Format (Hierarchical Lists):**
    1.  Each `sentence` object contains a `words` key, which is a list of word-lists.
    2.  Each **word-list** **must** be in the format: `["word_text", "start_time", "end_time", [syllable_list]]`.
    3.  Each **syllable-list** inside the word-list **must** be in the format: `["syllable_text", "start_time", "end_time"]`.
*   **Data Atomicity & Hierarchy:**
    1.  A word's start time (element 1) **must** exactly match its first syllable's start time. Its end time (element 2) **must** match its last syllable's end time.
    2.  The concatenation of `syllable_text` values (element 0) from the syllable-lists within a word-list **must perfectly match** the `word_text` (element 0).
    3.  The concatenation of `word_text` values from all word-lists within a sentence, joined by spaces, **must perfectly reconstruct** the `sentence.text`.
*   **Karaoke-Style Phrasing (Sentence Breaks):** Your goal is to mirror the song's musical phrasing. A long lyrical passage **should be broken into multiple `sentence` objects**. Each `sentence` should represent a phrase that is sung without a significant pause.
*   **Translation Mandate:** For each `sentence` object, you **MUST** provide an accurate and natural-sounding English translation in the `translation` field.
*   **Universal Romanization:** Convert all text from non-Latin scripts (e.g., Japanese) into their standard romanized equivalents. All `text` fields in the final JSON (`sentence.text` and all text elements in word/syllable lists) **must contain zero** non-Latin characters.
*   **Spoken Word Exclusion**: Do not transcribe spoken words. Focus exclusively on the **sung lyrical content**.

### **4. Format**

Generate the output as a single, raw JSON object conforming to the specified schema. **Output only the raw JSON.** Do not include any explanatory text or markdown formatting like ```json.

### **5. Mandatory Internal Pre-flight Check (Your Thought Process)**

Before outputting, you must silently verify the following:
1.  **Hierarchy & Logic Audit:** Is the JSON structure correct (list of sentences, list of word-lists, list of syllable-lists)? Are all timestamps `"MM:SS:sss"`? Is start < end for all elements? Are durations >= 70ms for syllables? Do parent start/end times match their children's? Is all timing chronological?
2.  **Final Timestamp Check:** Does the last syllable's `end_time` exceed `{audio_duration_str}`? If yes, cap it.
3.  **Data Atomicity Check:** Does syllable text -> word text? Does word text -> sentence text?
4.  **Content Check:** Is every sentence translated? Is all text romanized? Are sentence breaks logical for karaoke?
"""
        if lyrics:
            base_prompt += f"\n\n## Reference Lyrics\nFor context, here are the expected lyrics for this audio:\n\n{lyrics}\n\nUse these lyrics as a reference to improve transcription accuracy, respect the timing of the spoken words."
        return base_prompt

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

            myfile = self.client.files.upload(file=audio_to_transcribe)

            prompt = "Please transcribe this audio file into the specified hierarchical JSON format."
            contents = [prompt, myfile]

                                                                                   
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=1.0,
                    max_output_tokens=6400,
                    system_instruction=system_prompt,
                )
            )

            raw_text = self.filter_markdown(response.text)
            transcription = self._validate_and_correct_transcription(raw_text)

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

class WhisperTimestampedTranscriber(Transcriber):
    def __init__(self, model_name: str = "large-v3", device: Optional[str] = None, gemini_api_key: Optional[str] = None):
        """
        Initialize the Whisper-timestamped transcriber.
        Args:
            model_name (str): The name of the Whisper model to use (e.g., "base", "small", "medium", "large").
            device (str, optional): The device to use for inference (e.g., "cpu", "cuda").
                                    If None, it will try to use CUDA if available, otherwise CPU.
            gemini_api_key (str, optional): API key for Gemini, used for romanization.
        """
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Loading Whisper-timestamped model '{self.model_name}' on device: {self.device}...")
        self.model = whisper.load_model(self.model_name, device=self.device)
        print("✓ Whisper-timestamped model loaded.")

        self.gemini_client = None
        self.kakasi_converter = None

        if kakasi:                                              
            try:
                kks = kakasi()
                kks.setMode("H", "a")                    
                kks.setMode("K", "a")                    
                kks.setMode("J", "a")                    
                self.kakasi_converter = kks.getConverter()
                print("✓ pykakasi converter initialized for romanization.")
            except Exception as e:
                print(f"Warning: Failed to initialize pykakasi: {e}. Romanization will fall back to Gemini if API key is provided.")
                self.kakasi_converter = None
        
        if not self.kakasi_converter and gemini_api_key:
            self.gemini_client = genai.Client(api_key=gemini_api_key)
            self.gemini_model_name = "gemini-2.5-flash"                                
            print(f"✓ Gemini client initialized for romanization/translation (as fallback) with model {self.gemini_model_name}.")
        elif not self.kakasi_converter and not gemini_api_key:
            print("Warning: Neither pykakasi nor Gemini API key provided. Romanization and translation will not be performed.")

    def _romanize_texts(self, texts: List[str]) -> List[str]:
        """Romanizes a list of texts using pykakasi if available, otherwise falls back to Gemini."""
        if not texts:
            return []
            
        if self.kakasi_converter:
            try:
                                                                                    
                romanized_output = [
                    ' '.join(self.kakasi_converter.do(part) for part in text.split(' '))
                    for text in texts
                ]
                print("✓ Texts romanized using pykakasi.")
                return romanized_output
            except Exception as e:
                print(f"Warning: pykakasi romanization failed: {e}. Falling back to Gemini.")
        
        if self.gemini_client:
            prompt = f"""Romanize the following Japanese texts. Preserve spaces within each text string. For example, if an input string is "言葉 A 言葉 B", the romanized output string should be "kotoba A kotoba B". Provide the romanized versions as a JSON array of strings in the `romanized_texts` field.

Input JSON:
{{"texts": {json.dumps(texts, ensure_ascii=False)}}}

Output JSON:
"""
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type='application/json',
                        response_schema=RomanizationResponse,
                    )
                )
                romanized_response_data = json.loads(response.text)
                print("✓ Texts romanized using Gemini (fallback).")
                return romanized_response_data.get("romanized_texts", texts)
            except Exception as e:
                print(f"Warning: Gemini batch romanization failed: {e}. Returning original texts.")
                return texts
        
        print("Warning: No romanization method available. Returning original texts.")
        return texts

    def _translate_texts(self, texts: List[str]) -> List[str]:
        """Translates a list of texts to English using Gemini."""
        if not self.gemini_client or not texts:
            return [" " for _ in texts]
        
        prompt = f"""Translate the following Japanese texts to English. Provide the translations as a JSON array of strings in the `translations` field.

Input JSON:
{{"texts": {json.dumps(texts, ensure_ascii=False)}}}

Output JSON:
"""
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=len(texts) * 100,
                    response_mime_type='application/json',
                    response_schema=TranslationResponse,
                )
            )
            translation_data = json.loads(response.text)
            print("✓ Texts translated using Gemini.")
            return translation_data.get("translations", ["Translation failed." for _ in texts])
        except Exception as e:
            print(f"Warning: Gemini batch translation failed: {e}. Returning placeholders.")
            return [f"Translation error: {e}" for _ in texts]


    def transcribe_audio(self, audio_file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper-timestamped and format the output
        to match the hierarchical Gemini JSON structure.
        """
        skip_translation = kwargs.get("skip_translation", False)
        skip_syllable_extrapolation = kwargs.get("skip_syllable_extrapolation", False)
        
        try:
            if not Path(audio_file_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

            print(f"Transcribing '{audio_file_path}' using Whisper-timestamped model ({self.model_name})...")
            audio = whisper.load_audio(audio_file_path)
            result = whisper.transcribe(self.model, audio, language="ja", verbose=False, no_speech_threshold=0.0)

                                     
            texts_to_romanize: List[str] = []
            texts_to_translate: List[str] = []
            structured_data = []

            for segment in result.get('segments', []):
                sentence_text = segment.get('text', '').strip()
                if not sentence_text: continue
                
                texts_to_romanize.append(sentence_text)
                texts_to_translate.append(sentence_text)
                
                word_list = []
                for word_info in segment.get('words', []):
                    word_text = word_info.get('text', '').strip()
                    if not word_text: continue
                    texts_to_romanize.append(word_text)
                    word_list.append({
                        "text": word_text,
                        "start": word_info['start'],
                        "end": word_info['end']
                    })
                
                structured_data.append({
                    "start": segment['start'], "end": segment['end'], "words": word_list
                })

                                      
            print("Performing batch romanization...")
            romanized_texts = self._romanize_texts(texts_to_romanize)
            
            if not skip_translation:
                print("Performing batch translation...")
                translated_texts = self._translate_texts(texts_to_translate)
            else:
                print("Skipping translation step as requested for enhancement pipeline.")
                translated_texts = [" " for _ in texts_to_translate]
            
            romanized_iter = iter(romanized_texts)
            translated_iter = iter(translated_texts)
            
                                         
            final_sentences = []
            for segment_data in structured_data:
                romanized_sentence_text = next(romanized_iter, "")
                translation = next(translated_iter, " ")

                reconstructed_words = []
                for word_data in segment_data['words']:
                    romanized_word_text = next(romanized_iter, "")
                    
                    syllables = []
                    if not skip_syllable_extrapolation and len(romanized_word_text) > 0:
                        char_duration = (word_data['end'] - word_data['start']) / len(romanized_word_text)
                        current_char_time = word_data['start']
                        for char in romanized_word_text:
                            char_end_time = current_char_time + char_duration
                            syllables.append([
                                char,
                                self._seconds_to_mmss(current_char_time),
                                self._seconds_to_mmss(char_end_time)
                            ])
                            current_char_time = char_end_time
                    
                    if romanized_word_text:
                        reconstructed_words.append([
                            romanized_word_text,
                            self._seconds_to_mmss(word_data['start']),
                            self._seconds_to_mmss(word_data['end']),
                            syllables
                        ])
                
                if reconstructed_words:
                    final_sentences.append({
                        "text": romanized_sentence_text,
                        "translation": translation,
                        "words": reconstructed_words
                    })

            final_response_dict = {"sentences": final_sentences}
            transcription_json = json.dumps(final_response_dict, ensure_ascii=False)

            return {
                "success": True,
                "transcription": transcription_json,
                "model": f"whisper-timestamped-{self.model_name}",
                "usage": None,
                "speed_factor_used": 1.0
            }

        except Exception as e:
            print(f"Whisper-timestamped transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}


class WhisperGeminiTranscriber(Transcriber):
    """
    A transcriber that first uses Sieve to get a base transcription,
    and then uses Gemini to enhance it based on provided lyrics.
    """
    def __init__(self, gemini_api_key: str):
        """
        Initializes the Sieve-Gemini transcriber.
        Args:
            gemini_api_key (str): The API key for the Gemini model and Sieve.
        """
        self.sieve_transcriber = SieveTranscriber(api_key=gemini_api_key)
        self.gemini_client = genai.Client(api_key=gemini_api_key)
        self.gemini_model_name = "gemini-2.5-flash"                                            
        print(f"✓ Sieve-Gemini transcriber initialized for enhancement with model: {self.gemini_model_name}")

    def create_enhancement_prompt(self, audio_duration_str: str) -> str:
        """Creates the system prompt for Gemini to enhance the Whisper transcription into the new list-based format."""
        return f"""### **1. Persona**

You are **Lyra-Enhancer**, an AI assistant that refines and perfects karaoke transcriptions. You receive a preliminary JSON transcription from a Whisper model and a reference text of the correct lyrics. Your task is to meticulously correct the Whisper-generated JSON to perfectly match the provided lyrics in content, add accurate translations, and restructure it into the required list-based format for the best karaoke experience.

### **2. Task**

You are given a preliminary JSON object from a Whisper transcription, the ground-truth lyrics, and the original audio file. Your task is to enhance this JSON object. You must correct text, add translations, adjust timings, and restructure the data to produce a final, highly accurate JSON transcription that strictly follows the new schema.

### **3. Context & Constraints**

*   **Audio & Timing Constraints (The Ground Truth)**
*   The provided audio file is exactly `{audio_duration_str}` long (format is MM:SS:sss).
*   **Primary Directive:** The `end_time` of the very last syllable in the entire transcription **must not** exceed the total audio duration of `{audio_duration_str}`.
*   **Primary Directive: Content Correction:** The `text` fields in the output JSON must align with the **Reference Lyrics**. The Whisper JSON may have errors, hallucinations, or incorrect words. You must correct them. Use the audio file to align the corrected words to the right timestamps.
*   **CRITICAL: Output Format:** The final output must follow the new list-based schema.
    *   Each `sentence` object must have a `words` key, which is a list of word-lists.
    *   Each **word-list** must be in the format: `["word_text", "start_time", "end_time", [syllable_list]]`.
    *   Each **syllable-list** must be in the format: `["syllable_text", "start_time", "end_time"]`.
*   **Sentence Structure:** Prefer creating multiple, shorter sentence objects that correspond to natural lyrical phrases. **Do not merge distinct phrases into a single, long sentence object.** The goal is a granular, easy-to-follow structure.
*   **Translation Mandate:** The `translation` fields in the input JSON are placeholders. You **must replace them** with accurate, natural-sounding English translations for each sentence.
*   **Timing and Structure:** The Whisper timings are a good starting point, but not perfect. Prioritize correcting the words and syllables based on the lyrics and audio, and adjust start/end times as needed to ensure rhythmic accuracy. You can merge or split words and syllables to improve the flow.
*   **Backing Vocals/Background Vocals:** Make sure that the background vocals are properly encased in brackets and that they DO NOT interfere with the actual vocals. Sometimes the transcribing model completely halluncinates some backing vocals, and if they are not in the lyrics remove them.
*   **JSON Schema Adherence:** The output must be a single, raw JSON object conforming to the new list-based schema (`TranscriptionResponse`). Do not add any extra text or markdown.
*   **Hallucination:** If you cannot hear something that the transcription model says is present, then remove that something. The transcription model may hallucinate some things. It is okay to remove such things. PLEASE CHECK THE LYRICS WHEN YOU DO THIS, IF IT'S NOT IN THE LYRICS IT COULD BE A HALLUCINATION
*   **Internal Check:** Before outputting, silently verify all constraints from the original Lyra persona: hierarchical integrity (syllable -> word -> sentence), no zero-duration elements, proper syllable handling, you have properly put the backing vocals in brackets, no hallucinations, etc.
*   **CRITICAL: UNIVERSAL ROMANIZATION:** Absolutely all `text` fields in the final output (`sentence.text` and all text elements in word/syllable lists) **MUST BE ROMANIZED**. There should be zero non-Latin characters in the output. This is a non-negotiable rule.
    *   **Japanese:** Use standard Hepburn romanization (e.g., "さくら" → "sakura", "東京" → "Tokyo")
    *   **Korean:** Use Revised Romanization (e.g., "사랑" → "sarang", "안녕하세요" → "annyeonghaseyo")
    *   **Chinese:** Use Pinyin with tone marks removed (e.g., "你好" → "nihao", "中国" → "zhongguo")
    *   **Arabic:** Use standard Latin transliteration (e.g., "السلام عليكم" → "assalamu alaikum")
    *   **Russian:** Use standard Latin transliteration (e.g., "привет" → "privet", "спасибо" → "spasibo")
*   **TIMESTAMP FORMAT:** The timestamp MUST be in the format of `MM:SS:sss`, no other format will work.
*   **LOAN WORDS:** Any and all english loan words, from any language must be converted to english in the `text` field.

### **4. Syllable Duration Guidelines**

*   **Minimum Syllable Duration:** No syllable should be shorter than 100ms (0.1 seconds). If a syllable would be shorter, merge it with adjacent syllables or extend its duration.
*   **Maximum Reasonable Duration:** Single syllables should generally not exceed 3-5 seconds unless they represent:
    *   Held notes or sustained vocals (e.g., "Ahhhhh", "Ooooh")
    *   Vocal runs or melisma (multiple notes on one syllable)
    *   Emotional emphasis or screams
*   **Extended Syllables/Held Notes:** When a syllable is held or extended:
    *   Extend the syllable text to reflect the duration (e.g., "Ah" → "Ahhhh", "Love" → "Looove")
    *   The extension should be proportional to the duration (longer holds = more repeated letters)
    *   Use the most prominent vowel sound for extension
    *   **For phonetic languages:** Extend based on the actual phonetic sound:
        *   Japanese: "さ" (sa) → "saaa", "愛" (ai) → "aiii"
        *   Korean: "사" (sa) → "saaa", "아" (a) → "aaaa"
        *   Spanish: "amor" → "amoooor", "sí" → "sííí"
*   **Vocal Runs/Melisma:** For complex vocal runs on a single syllable:
    *   Keep as one syllable if it's the same phonetic sound
    *   Extend the text to indicate the held nature
    *   Ensure the timing covers the entire vocal run
*   **Screams/Emotional Vocals:** For screams, shouts, or emotional vocalizations:
    *   Extend appropriately (e.g., "No" → "Noooo", "Yeah" → "Yeahhhh")
    *   **Japanese examples:** "いや" (iya) → "iyaaa", "だめ" (dame) → "dameee"
    *   **Korean examples:** "아니" (ani) → "aniii", "싫어" (silheo) → "silheooo"
    *   Capture the full emotional duration
*   **Quality Control:** If you encounter syllables longer than 5 seconds, double-check:
    *   Is this actually one continuous vocal sound?
    *   Should this be split into multiple syllables?
    *   Is there a gap or breath that should create a word boundary?
    *   Does the reference lyrics support this duration?

### **5. Translation Guidelines: Persona-Driven Lyrical Adaptation**

Your role here is not a direct translator, but a master lyricist, adapting the song's soul into English. Your entire approach must be dictated by the original persona. This is a multi-step process.

**Step 1: Identify the Core Persona.**
Analyze the original lyrics to determine the speaker's voice. This is your most important task. Ask these questions:
*   **What is their emotional state?** (e.g., Angry, sad, hopeful, arrogant, vulnerable)
*   **What is their attitude?** (e.g., Cynical, sweet, demanding, pleading)
*   **What is the overall mood of the song?** (e.g., Aggressive rock, theatrical ballad, sweet pop)

**Step 2: Select the Lyrical Style.**
Based on your persona analysis, choose a corresponding English lyrical style. The goal is authenticity.
*   **For an Aggressive, Cynical, or Modern Persona:**
    *   **Use:** Sharp, colloquial, and even provocative modern slang.
    *   **Vocabulary:** `bitch`, `piss off`, `talking shit`, `loser`, `unalive`, `lol`, `GG`.
    *   **Goal:** Capture a raw, contemporary, often internet-savvy edge.
*   **For a Dramatic, Arrogant, or Tragic Persona:**
    *   **Use:** Poetic, weighty, and theatrical language with a slightly formal or elevated tone.
    *   **Vocabulary:** `privilege`, `eternity`, `filthy`, `traitor`, `disappear`, `pass right now`.
    *   **Goal:** Convey a sense of high drama, self-importance, and emotional weight.
*   **For a Sweet, Vulnerable, or Hopeful Persona:**
    *   **Use:** Simple, earnest, and sincere language. The tone is often pleading or gentle.
    *   **Vocabulary:** `Can I please?`, `I wanna be`, `secret feelings`, `tremble from fear`.
    *   **Goal:** Capture a sense of innocence, longing, and fairy-tale-like wonder.

**Step 3: Execute the Adaptation.**
Apply your chosen style while following these core principles:
*   **Impact Over Literalism:** The feeling of the line is more important than a word-for-word translation. Recreate the emotional impact.
*   **Cultural Equivalents:** Translate the *function* of slang or cultural references, not just their definition. `アッシー` (chauffeur) becomes "wheels." `乙` (dismissive slang) becomes "bye loser" or "GG."
*   **The Performance Test (Litmus Test):** Before finalizing, ask: **"Does this sound like a real, singable song lyric, or does it sound like a subtitle?"** If it sounds like a subtitle, rewrite it to be more punchy, rhythmic, and lyrical.

### **6. Example**
```json
{{
  "sentences": [
    {{
      "text": "Emotions I feel they turn to none",
      "translation": "Emotions I feel they turn to none",
      "words": [
        ["Emotions", "00:31:060", "00:31:780", [
          ["E", "00:31:060", "00:31:246"],
          ["mo", "00:31:246", "00:31:513"],
          ["tions", "00:31:513", "00:31:780"]
        ]],
        ["I", "00:31:780", "00:32:180", [
          ["I", "00:31:780", "00:32:180"]
        ]],
        ["feel", "00:32:180", "00:32:520", [
          ["feel", "00:32:180", "00:32:520"]
        ]],
        ["they", "00:32:600", "00:33:100", [
          ["they", "00:32:600", "00:33:100"]
        ]],
        ["turn", "00:33:100", "00:33:460", [
          ["turn", "00:33:100", "00:33:460"]
        ]],
        ["to", "00:33:460", "00:33:860", [
          ["to", "00:33:460", "00:33:860"]
        ]],
        ["none", "00:33:860", "00:34:360", [
          ["none", "00:33:860", "00:34:360"]
        ]]
      ]
    }},
    {{
      "text": "I broke, I broke, them one by one",
      "translation": "I broke, I broke, them one by one",
      "words": [
        ["I", "00:34:860", "00:35:000", [
          ["I", "00:34:860", "00:35:000"]
        ]],
        ["broke,", "00:35:000", "00:35:360", [
          ["broke,", "00:35:000", "00:35:360"]
        ]],
        ["I", "00:35:560", "00:35:940", [
          ["I", "00:35:560", "00:35:940"]
        ]],
        ["broke,", "00:35:940", "00:36:320", [
          ["broke,", "00:35:940", "00:36:320"]
        ]],
        ["them", "00:36:320", "00:36:880", [
          ["them", "00:36:320", "00:36:880"]
        ]],
        ["one", "00:36:880", "00:37:160", [
          ["one", "00:36:880", "00:37:160"]
        ]],
        ["by", "00:37:160", "00:37:600", [
          ["by", "00:37:160", "00:37:600"]
        ]],
        ["one", "00:37:600", "00:38:180", [
          ["one", "00:37:600", "00:38:180"]
        ]]
      ]
    }}
  ]
}}
```

### **7. Examples of Extended Syllables**

**English Example:**
```json
{{
  "sentences": [
    {{
      "text": "Looooove me tender",
      "translation": "Love me tender",
      "words": [
        ["Looooove", "00:15:000", "00:17:500", [
          ["Looooove", "00:15:000", "00:17:500"]
        ]],
        ["me", "00:17:500", "00:17:800", [
          ["me", "00:17:500", "00:17:800"]
        ]],
        ["tender", "00:17:800", "00:18:400", [
          ["ten", "00:17:800", "00:18:100"],
          ["der", "00:18:100", "00:18:400"]
        ]]
      ]
    }}
  ]
}}
```

**Japanese Example:**
```json
{{
  "sentences": [
    {{
      "text": "Kimi wo aishiteru yooo",
      "translation": "I love you so much",
      "words": [
        ["Kimi", "00:10:000", "00:10:600", [
          ["Ki", "00:10:000", "00:10:300"],
          ["mi", "00:10:300", "00:10:600"]
        ]],
        ["wo", "00:10:600", "00:10:900", [
          ["wo", "00:10:600", "00:10:900"]
        ]],
        ["aishiteru", "00:10:900", "00:12:400", [
          ["ai", "00:10:900", "00:11:200"],
          ["shi", "00:11:200", "00:11:500"],
          ["te", "00:11:500", "00:11:800"],
          ["ru", "00:11:800", "00:12:400"]
        ]],
        ["yooo", "00:12:400", "00:14:000", [
          ["yooo", "00:12:400", "00:14:000"]
        ]]
      ]
    }}
  ]
}}
```

**Korean Example:**
```json
{{
  "sentences": [
    {{
      "text": "Saranghaeee neol wihae",
      "translation": "I love you, for you",
      "words": [
        ["Saranghaeee", "00:08:000", "00:10:500", [
          ["Sa", "00:08:000", "00:08:400"],
          ["rang", "00:08:400", "00:08:900"],
          ["haeee", "00:08:900", "00:10:500"]
        ]],
        ["neol", "00:10:500", "00:11:100", [
          ["ne", "00:10:500", "00:10:800"],
          ["ol", "00:10:800", "00:11:100"]
        ]],
        ["wihae", "00:11:100", "00:11:800", [
          ["wi", "00:11:100", "00:11:450"],
          ["hae", "00:11:450", "00:11:800"]
        ]]
      ]
    }}
  ]
}}
```

**Spanish Example:**
```json
{{
  "sentences": [
    {{
      "text": "Te amooo con todo mi corazón",
      "translation": "I love you with all my heart",
      "words": [
        ["Te", "00:12:000", "00:12:300", [
          ["Te", "00:12:000", "00:12:300"]
        ]],
        ["amooo", "00:12:300", "00:14:800", [
          ["amooo", "00:12:300", "00:14:800"]
        ]],
        ["con", "00:14:800", "00:15:100", [
          ["con", "00:14:800", "00:15:100"]
        ]],
        ["todo", "00:15:100", "00:15:600", [
          ["to", "00:15:100", "00:15:350"],
          ["do", "00:15:350", "00:15:600"]
        ]],
        ["mi", "00:15:600", "00:15:900", [
          ["mi", "00:15:600", "00:15:900"]
        ]],
        ["corazón", "00:15:900", "00:16:800", [
          ["co", "00:15:900", "00:16:150"],
          ["ra", "00:16:150", "00:16:400"],
          ["zón", "00:16:400", "00:16:800"]
        ]]
      ]
    }}
  ]
}}
```

### **8. Language-Specific Syllable Guidelines**

*   **Japanese:**
    *   Each kana character typically represents one syllable
    *   Long vowels: "さあ" (saa), "そう" (sou) should be treated as extended syllables
    *   Particle extensions: "よ" (yo) → "yooo", "ね" (ne) → "neee"
    *   Respect mora timing - Japanese has predictable syllable durations
*   **Korean:**
    *   Each Hangul syllable block is one syllable
    *   Extend vowels within romanization: "사랑해" (saranghae) → "saranghaee"
    *   Common emotional extensions: "아" (a) → "aaaa", "오" (o) → "oooo"
*   **Spanish:**
    *   Clear vowel sounds make extension straightforward
    *   Extend based on vowel prominence: "amor" → "amoooor", "corazón" → "corazóooon"
    *   Maintain stress patterns in extensions
*   **Chinese (Mandarin):**
    *   Each character is typically one syllable
    *   Extend the vowel component: "爱" (ai) → "aiii", "好" (hao) → "haooo"
    *   Ignore tone marks in romanization extensions
*   **Arabic:**
    *   Extend long vowels: "حبيبي" (habibi) → "habiiiibi"
    *   Respect consonant clusters in syllable breaks
*   **Russian:**
    *   Extend vowels: "любовь" (lyubov) → "lyubooov"
    *   Soft/hard consonant distinctions maintained in romanization

### **9. Format**

Output only the raw, corrected, and enhanced JSON object in the specified list-based format.
"""

    def transcribe_audio(self, audio_file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribes audio by first using Sieve and then enhancing with Gemini.
        """
        lyrics = kwargs.get("lyrics")
        output_file_path = kwargs.get("output_file_path")
        debug_json_input_file = kwargs.get("debug_json_input_file")

        if not lyrics:
            return {"success": False, "error": "Lyrics are required for whisper-gemini mode."}

        sieve_json = None
        if debug_json_input_file:
            print(f"\n--- Debug Mode: Using provided JSON file '{debug_json_input_file}' for GeminiTranscriber ---")
            try:
                with open(debug_json_input_file, 'r', encoding='utf-8') as f:
                    sieve_json = f.read()
                print("✓ Debug JSON loaded successfully.")
            except Exception as e:
                return {"success": False, "error": f"Failed to read debug JSON file: {e}"}
        else:
                                                              
            print("\n--- Step 1: Running Sieve Transcription (pre-processing) ---")
            sieve_result = self.sieve_transcriber.transcribe_audio(audio_file_path, **kwargs)

            if not sieve_result.get("success"):
                print("Sieve pre-processing failed. Aborting.")
                return sieve_result

            sieve_json = sieve_result["transcription"]
            print("✓ Sieve pre-processing successful. (Structure generated for enhancement)")

                                                                                 
            if output_file_path:
                try:
                                                                               
                    debug_path = str(Path(output_file_path).parent / 'debug.json')
                    print(f"Saving intermediate Sieve output to {debug_path}...")
                    
                    parsed_json = json.loads(sieve_json)
                    with open(debug_path, 'w', encoding='utf-8') as f:
                        json.dump(parsed_json, f, indent=2, ensure_ascii=False)                             
                    print(f"✓ Intermediate output saved to {debug_path}.")
                except Exception as e:
                    print(f"Warning: Could not save intermediate debug.json file: {e}")
        
                                                       
        print("\n--- Step 2: Enhancing Transcription with Gemini ---")
        
        audio_file_gemini = None
        try:
            print(f"Uploading audio file '{audio_file_path}' to Gemini for enhancement context...")
            audio_file_gemini = self.gemini_client.files.upload(file=audio_file_path)
            print("✓ Audio file uploaded to Gemini.")
        except Exception as e:
            print(f"Error uploading audio file to Gemini: {e}")
            return {"success": False, "error": f"Failed to upload audio to Gemini: {e}"}

        audio_duration_seconds = AudioSegment.from_file(audio_file_path).duration_seconds
        audio_duration_str = self._seconds_to_mmss(audio_duration_seconds)
        print(f"✓ Audio duration for Gemini enhancement prompt: {audio_duration_str}")
        system_prompt = self.create_enhancement_prompt(audio_duration_str)
        
        try:
                                                                                            
                                                                                       
            prompt_parts = [
                "Please enhance this transcription based on the audio and the rules provided in the system instruction.",
                f"Input Sieve JSON (to be corrected):\n```json\n{json.dumps(json.loads(sieve_json), separators=(',', ':'), ensure_ascii=False)}\n```",
                f"Reference Lyrics (Ground Truth):\n```\n{lyrics}\n```",
                audio_file_gemini
            ]

            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt_parts,
                config=types.GenerateContentConfig(
                    temperature=1.0,                                                    
                    system_instruction=system_prompt,                                                   
                                                                            
                )
            )
            
            raw_text = self.filter_markdown(response.text)
            enhanced_transcription = self._validate_and_correct_transcription(raw_text)

            return {
                "success": True,
                "transcription": enhanced_transcription,
                "model": f"sieve-gemini ({self.gemini_model_name})",
                "usage": None,
                "speed_factor_used": 1.0
            }
        except Exception as e:
            print(f"Gemini enhancement failed: {e}.")
            return {"success": False, "error": f"Gemini enhancement failed: {e}"}
        finally:
            if audio_file_gemini:
                try:
                    self.gemini_client.files.delete(name=audio_file_gemini.name)
                    print(f"✓ Cleaned up Gemini uploaded file: {audio_file_gemini.name}")
                except Exception as e:
                    print(f"Warning: Could not delete Gemini uploaded file {audio_file_gemini.name}: {e}")


class SieveTranscriber(Transcriber):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.transcribe_function = sieve.function.get("sieve/transcribe")
        print("✓ Sieve transcriber initialized.")

    def transcribe_audio(self, audio_file_path: str, **kwargs) -> Dict[str, Any]:
        try:
            if not Path(audio_file_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

            print(f"Transcribing '{audio_file_path}' using Sieve...")

                                                                                         
                                               
            is_standalone_run = not kwargs.get("lyrics")

            sieve_file = sieve.File(path=audio_file_path)

            output = self.transcribe_function.run(
                file=sieve_file,
                backend="whisper-timestamped-whisper-large-v3",
                word_level_timestamps=True,
                source_language="auto",
                diarization_backend="None",
                min_speakers=-1,
                max_speakers=-1,
                custom_vocabulary={},
                translation_backend="None",
                target_language="",
                segmentation_backend="vad",
                min_segment_length=-1,
                min_silence_length=0.4,
                vad_threshold=0.85,
                pyannote_segmentation_threshold=0.8,
                chunks=[],
                denoise_backend="None",
                initial_prompt=""
            )

            final_sentences = []
                                         
            for transcription_output in output:
                for segment in transcription_output.get('segments', []):
                    sentence_text = segment.get('text', '').strip()
                    if not sentence_text:
                        continue

                    segment_words_data = []
                    for word_info in segment.get('words', []):
                        word_text = word_info.get('word', '').strip()
                        word_start = word_info.get('start')
                        word_end = word_info.get('end')

                        if not word_text or word_start is None or word_end is None:
                            continue
                        
                        syllables = []
                                                                                          
                                                                                                    
                        if is_standalone_run:
                            syllables.append([
                                word_text,
                                self._seconds_to_mmss(word_start),
                                self._seconds_to_mmss(word_end)
                            ])

                        segment_words_data.append([
                            word_text,
                            self._seconds_to_mmss(word_start),
                            self._seconds_to_mmss(word_end),
                            syllables
                        ])
                    
                    if segment_words_data:
                        final_sentences.append({
                            "text": sentence_text,
                            "translation": " ",                              
                            "words": segment_words_data
                        })
            
            transcription_json = json.dumps({"sentences": final_sentences}, ensure_ascii=False)
            
                                                                   
            transcription_json = self._validate_and_correct_transcription(transcription_json)

            return {
                "success": True,
                "transcription": transcription_json,
                "model": "sieve/transcribe",
                "usage": None,
                "speed_factor_used": 1.0
            }
        except Exception as e:
            print(f"Sieve transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio to syllable-level JSON using various models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("--output", "-o", help="Output JSON file path. Defaults to 'transcription.json' in the audio file's directory.", default=None)
    parser.add_argument("--debug", help="Path to a JSON file to use as input for GeminiTranscriber, skipping WhisperTranscription.", default=None)
    
                              
    parser.add_argument("--model",
                        choices=["gemini", "whisper", "whisper-gemini", "sieve"],
                        default="gemini",
                        help="Choose the transcription model. 'whisper' defaults to Sieve unless --local is specified (default: gemini).")
    parser.add_argument("--local",
                        action="store_true",
                        help="If set, forces the use of local Whisper model even if Sieve is available.")

                        
    gemini_group = parser.add_argument_group('Gemini API Options (used for gemini, whisper, and whisper-gemini)')
    gemini_group.add_argument("--lyrics", help="Path to a text file with lyrics to guide transcription (required for whisper-gemini)", default=None)
    gemini_group.add_argument("--api-key", help="API key for Gemini (or set GEMINI_API_KEY env var)")
    gemini_group.add_argument("--base-url", help="API base URL", default="https://generativelanguage.googleapis.com/v1beta/openai/")
    gemini_group.add_argument("--speed",
                               help="Audio speed factor for transcription with 'gemini' model (e.g., 0.7 for 70%% speed). "
                                    "Values < 1 slow down audio, potentially improving accuracy.",
                               type=float, default=1.0)
    gemini_group.add_argument("--no-preserve-pitch",
                               action="store_true",
                               help="Use the faster, pitch-altering method for slowing down audio instead of the default 'atempo' filter.")

                                 
    whisper_group = parser.add_argument_group('Whisper-timestamped Options (used for whisper and whisper-gemini)')
    whisper_group.add_argument("--whisper-model",
                                default=None,                                 
                                help="Whisper model size (default: large-v3 for whisper, large-v3 for whisper-gemini).")
    whisper_group.add_argument("--whisper-device",
                                help="Device to use for Whisper (e.g., 'cpu', 'cuda'). Defaults to auto-detect.",
                                default=None)

    args = parser.parse_args()

                                         
    if args.output:
        output_path = args.output
    else:
        audio_path = Path(args.audio_file)
        output_path = str(audio_path.parent / 'transcription.json')

    transcriber: Transcriber
    transcribe_kwargs: Dict[str, Any] = {}
    
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")

    if args.model == "gemini":
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

    elif args.model == "sieve" or (args.model == "whisper" and not args.local):
        if not api_key:
            print("Error: API key required for Sieve. Use --api-key or set GEMINI_API_KEY.")
            sys.exit(1)
        transcriber = SieveTranscriber(api_key=api_key)
        print(f"\nTranscribing '{args.audio_file}' using Sieve...")

    elif args.model == "whisper" and args.local:
        if not api_key and not kakasi:
            print("Warning: Gemini API key not provided and pykakasi not found. Whisper output will not be romanized or translated.")
        
        whisper_model_name = args.whisper_model or "large-v3"
        transcriber = WhisperTimestampedTranscriber(whisper_model_name, args.whisper_device, gemini_api_key=api_key)
        print(f"\nTranscribing '{args.audio_file}' using local Whisper-timestamped model ({transcriber.model_name})...")

    elif args.model == "whisper-gemini":
        if not api_key:
            print("Error: API key required for whisper-gemini. Use --api-key or set GEMINI_API_KEY.")
            sys.exit(1)

        if not args.lyrics:
            print("Error: --lyrics argument is required for whisper-gemini model.")
            sys.exit(1)

        transcriber = WhisperGeminiTranscriber(
            gemini_api_key=api_key
        )
        
        print(f"Reading lyrics from: {args.lyrics}")
        transcribe_kwargs['lyrics'] = transcriber.read_lyrics_from_file(args.lyrics)
        if not transcribe_kwargs['lyrics']:
            print(f"Error: Could not read lyrics from {args.lyrics}")
            sys.exit(1)
        print("✓ Lyrics loaded successfully")
        
        transcribe_kwargs['output_file_path'] = output_path                                      
        if args.debug:
            transcribe_kwargs['debug_json_input_file'] = args.debug
        
        print(f"\nTranscribing '{args.audio_file}' using Whisper-Gemini pipeline...")

    else:
        print(f"Error: Unknown model '{args.model}'. Choose 'gemini', 'whisper', 'whisper-gemini', or 'sieve'.")
        sys.exit(1)
    
    result = transcriber.transcribe_audio(args.audio_file, **transcribe_kwargs)

    if result.get("success"):
        print("\n✓ Transcription completed successfully")
        
                                                           
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if transcriber.save_transcription(result["transcription"], output_path):
            print(f"✓ Transcription saved to: {output_path}")
        else:
            print(f"⚠ Transcription saved with warnings to: {output_path}")

        if result.get("usage"):
            print(f"Usage: {result['usage']}")
        
                                                                            
        if result.get("speed_factor_used", 1.0) != 1.0:
            print(f"Speed factor used: {result['speed_factor_used']}x (timestamps adjusted back to original speed)")

        try:
            preview = json.loads(result["transcription"])
            print("\n--- Transcription Preview (from " + result.get('model', 'N/A') + ") ---")
            preview_text = json.dumps(preview, indent=2, ensure_ascii=False)
            print(preview_text[:1000] + ("..." if len(preview_text) > 1000 else ""))
        except (json.JSONDecodeError, KeyError):
            print("\n--- Raw Response (from " + result.get('model', 'N/A') + ") ---")
            raw_text = result.get("transcription", "")
            print(raw_text[:1000] + ("..." if len(raw_text) > 1000 else ""))
    else:
        print(f"\n✗ Transcription failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
