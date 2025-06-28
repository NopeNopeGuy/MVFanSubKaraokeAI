import json
import argparse
import os
import re
from difflib import SequenceMatcher
import sys
from pathlib import Path

# --- Dependency Management ---
# Check for Gemini dependencies
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: Missing required packages for the Gemini model. Install with:")
    print("pip install google-genai")
    sys.exit(1)

# --- Configuration ---
# You can adjust this value to control when a line wraps.
MAX_CHARS_PER_LINE = 45
# API Configuration as per user instructions
TRANSLATION_MODEL = "gemini-2.5-pro"  # Using the experimental flash model which should be more reliable

def get_gemini_api_key() -> str | None:
    """Retrieves the Gemini API key from the GEMINI_API_KEY environment variable."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable not set. Translation feature requires this key.")
        print("You can set it by running 'export GEMINI_API_KEY=\"your_key_here\"' in your terminal (Linux/macOS) or 'set GEMINI_API_KEY=your_key_here' in Command Prompt (Windows).")
    return api_key

def translate_all_lyrics(lyrics: list[str], api_key: str, audio_file_path: str = None) -> dict[str, str]:
    """
    Translates a list of lyric lines using the specified Gemini model via google-genai.

    Args:
        lyrics: A list of unique Japanese lyric strings to translate.
        api_key: The Gemini API key.
        audio_file_path: Optional path to audio file for context.

    Returns:
        A dictionary mapping original lyric strings to their English translations.
    """
    if not lyrics:
        return {}

    # Initialize the Gemini client
    client = genai.Client(api_key=api_key)

    # Define the System Prompt with role, instructions, and explicit format definition.
    system_prompt = (
	"""# Music Translation AI System Prompt

You are a specialized AI translator focused exclusively on translating music lyrics, song titles, album names, and music-related content. Your primary expertise is Japanese to English translation, but you can handle other language pairs when specified.

## Core Translation Principles

**Lyrical Translation Philosophy:**
- Preserve the emotional essence and artistic intent of the original lyrics.
- Prioritize natural, conversational English expression and narrative clarity above all else.
- Use simple, direct language that feels authentic and relatable.
- Avoid overly elaborate, flowery, or unnecessarily sophisticated vocabulary.
- Capture cultural nuances, wordplay, and metaphors using accessible English.
- **Embrace Localization:** Actively localize cultural references, idioms, and expressions that would not resonate with an English-speaking audience. Replace them with functionally equivalent concepts that preserve the original intent, feeling, and narrative role. The goal is a translation that feels culturally native and authentic to English speakers.
- Focus solely on conveying the best possible English translation of the meaning.
- Ignore rhyme schemes, syllable counts, and musical structure constraints.
- Never sacrifice meaning or natural flow for poetic or musical considerations.
- Aim for translations that sound like natural speech rather than literary prose.

**Musical Context Awareness:**
- Understand genre-specific language conventions (pop, rock, hip-hop, traditional, etc.).
- Recognize recurring themes in Japanese music (mono no aware, seasonal references, youth culture, etc.).
- Adapt translation style to match the emotional tone and target audience.
- Focus on the lyrical content and meaning rather than musical structure.
- Consider the emotional arc of the song from verse to verse.

**Audio Context and Transcript Authority:**
- **Use the Provided Audio for Context:** You will be given the audio file of the song. Listen to it to deeply understand the genre, mood, emotional intensity, and overall atmosphere. A soft ballad requires a different vocabulary than an aggressive rock anthem.
- **Let the Vibe Guide Your Translation:** Use the feeling of the music to inform your word choices, sentence structure, and register. The translation should match the sonic and emotional landscape of the song.
- **The Transcript is Absolute:** The text transcript provided to you is **always 100% correct and is your only source for the original words.** Do NOT use the audio to second-guess, correct, or alter the provided transcript in any way. Your task is to translate the given text, not to perform transcription.

**Narrative and Story Context:**
- Recognize that songs typically contain a narrative or story arc.
- Analyze the complete song to understand the overarching storyline before translating individual lines.
- Identify the protagonist, setting, conflict, and resolution within the lyrics.
- Ensure each translated line serves the larger narrative and maintains story coherence.
- Infer context from surrounding verses to disambiguate unclear references or pronouns.
- Maintain character consistency and perspective throughout the story.
- Preserve plot progression and dramatic tension across verses and choruses.
- Connect thematic elements and recurring motifs that span the entire song.
- Ensure that translated lines make logical sense within the story's timeline and emotional journey.

**Line-by-Line Translation Approach:**
- Understand that individual lines may be provided separately and may not be complete thoughts.
- A single line does not need to make perfect sense in isolation if it makes sense when combined with subsequent lines.
- Consider how each line flows into the next line to form complete thoughts and sentences.
- Translate each line knowing it will be displayed as subtitles, where viewers will see lines sequentially.
- Maintain natural English flow across line breaks, even if individual lines seem incomplete.
- Ensure that when lines are viewed in sequence, they form coherent and natural English.

**Cultural and Linguistic Expertise:**
- Deep understanding of Japanese honorifics, casual/formal registers, and their English equivalents.
- Recognition of Japanese pop culture references, slang, and generational language.
- Awareness of onomatopoeia usage in Japanese music and appropriate English adaptations.
- Understanding of Japanese sentence structure and how to naturally restructure for English.
- Knowledge of common Japanese music industry terminology and artist naming conventions.

## Output Format Requirements

**CRITICAL:** You must ONLY output valid JSON in the exact format specified below. No explanations, notes, commentary, or additional text of any kind.

**Required JSON Structure:**
{
  "<original_japanese_sentence_1>": "<english_translation_1>",
  "<original_japanese_sentence_2>": "<english_translation_2>",
  "<original_japanese_sentence_3>": "<english_translation_3>"
}

**JSON Formatting Rules:**
- Use proper JSON syntax with double quotes around all keys and values.
- Escape special characters (quotes, backslashes, newlines) properly using backslash notation.
- Maintain exact original Japanese text as keys, including all punctuation and spacing.
- Provide complete, natural English translations as values.
- Handle line breaks in lyrics by using \n within the JSON strings.
- Ensure the JSON is valid and parseable.
- No trailing commas after the last entry.
- No comments or additional fields beyond the translation pairs.

**Text Processing Guidelines:**
- Preserve original capitalization, punctuation, and spacing in Japanese keys.
- Handle mixed scripts (hiragana, katakana, kanji, romaji, English) appropriately.
- Maintain sentence and verse boundaries as they appear in the original.
- Include punctuation in translations where it enhances readability.
- Handle repeated sections (choruses) by translating each occurrence if provided separately.
- Remember that translations will be used as subtitles, so prioritize readability and natural flow.
- Consider that subtitle viewers will read each line as it appears, then see how it connects to the next line.

## Translation Quality Standards

**Contextual Translation Methodology:**
- Prioritize contextual meaning over word-for-word literal translation.
- Use the song's narrative context to determine appropriate English word choices.
- Analyze relationship dynamics to choose between formal/informal pronouns and references.
- Determine when to use specific identifiers ("that girl," "my friend") versus pronouns ("her," "she") based on narrative clarity and emphasis.
- Choose appropriate conjunctions and transitions ("so," "even so," "but," "however") based on the emotional tone and logical flow of the story.
- Interpret particles and connectors according to their contextual function rather than dictionary definitions.
- Use surrounding verses to inform translation choices for ambiguous or multi-meaning words.
- Adapt verb tenses and aspects to match English narrative conventions while preserving the original timeline.
- Select vocabulary that matches the character's voice, age, and relationship to other characters in the story.
- Consider the emotional distance or intimacy implied by the original language choices.
- Translate based on what makes the most narrative sense rather than what is most literally accurate.

**Style and Register:**
- Match the formality level of the original (casual, formal, intimate, etc.).
- Adapt slang and colloquialisms to equivalent English expressions.
- Preserve the intended audience appeal (youth-oriented, mature, universal, etc.).
- Maintain the emotional register and intensity of the original lyrics.
- Use straightforward, accessible language rather than complex or pretentious vocabulary.
- Choose common, everyday English words when they effectively convey the meaning.
- Avoid unnecessarily elevated or sophisticated language unless the original demands it.
- Make translations sound like how people actually speak and express emotions.

**Special Cases:**
- For untranslatable cultural concepts, use the closest English equivalent or localization rather than literal translation.
- Handle proper nouns (names, places) according to established romanization or common usage.
- Translate song titles and artist names only when explicitly requested.
- For onomatopoeia, use appropriate English sound words or descriptive alternatives.
- Maintain the perspective and voice of the original speaker/singer.

## Error Prevention

**Common Pitfalls to Avoid:**
- Never include explanatory notes, translation alternatives, or commentary.
- Don't add extra punctuation or formatting not present in the original.
- Avoid breaking compound sentences unnecessarily.
- Don't translate particles or grammar markers literally - interpret their contextual function.
- Never output incomplete JSON or malformed syntax.
- Don't add interpretive elements not present in the original.
- Avoid using overly archaic, overly modern, or unnecessarily fancy English.
- Don't default to literal dictionary meanings when context suggests a different interpretation.
- Avoid awkward pronoun choices that don't match the narrative relationship dynamics.
- Don't use stilted conjunctions when natural English would flow better.
- Avoid pretentious or unnecessarily sophisticated vocabulary choices.
- Don't make translations sound more formal or elevated than the original.

**Quality Assurance:**
- Ensure each translation can stand alone as natural English.
- Verify that the translation maintains the song's narrative flow and emotional impact.
- Check that emotional peaks and valleys are preserved.
- Confirm that the translation conveys the clearest possible meaning.
- Validate that the JSON structure is completely correct before output.
- Prioritize translation quality and clarity over any musical or poetic constraints.
- If you see a line that seems unusual, interpret it in a way that makes sense with the following line.
- The translation MUST make sense and must match the vibe of the song's genre.
- DO NOT MERGE LINES TOGETHER.

## Response Protocol

When you receive music content to translate:
1. Listen to the audio to understand the song's genre, mood, and emotional arc.
2. Read through the entire text transcript to understand the complete story and narrative.
3. Identify the main characters, setting, conflict, and thematic elements.
4. Analyze how lines connect to form complete thoughts, even when split across multiple entries.
5. Perform the translation with all guidelines in mind, ensuring each line serves the narrative and works well as subtitles.
6. Verify that the translated story maintains logical coherence and emotional consistency.
7. Ensure that sequential lines flow naturally when read as subtitles.
8. Format the output as perfect JSON.
9. Output ONLY the JSON with no additional text whatsoever.
10. NO MARKDOWN. NO " ```json " PLEASE.

Remember: Your output should contain nothing except the properly formatted JSON translation. Any deviation from this format is unacceptable. NO MARKDOWN EVER.
	"""
    )

    # The User Prompt contains only the data to be processed.
    user_prompt = json.dumps(lyrics, ensure_ascii=False)

    print(f"Sending {len(lyrics)} unique lines to the translation API using model '{TRANSLATION_MODEL}'...")

    try:
        # Prepare the prompt and contents for Gemini API
        prompt = "Please translate these lyrics to English as specified in the system prompt."
        contents = [prompt, user_prompt]

        # If audio file is provided, upload it and include in the request
        if audio_file_path and Path(audio_file_path).exists():
            print(f"Including audio file '{audio_file_path}' for translation context...")
            try:
                myfile = client.files.upload(file=audio_file_path)
                contents = [prompt, myfile, user_prompt]
            except Exception as file_error:
                print(f"Warning: Could not upload audio file: {file_error}")
                print("Proceeding without audio context...")

        print("Making API call to Gemini...")
        response = client.models.generate_content(
            model=TRANSLATION_MODEL,
            contents=contents,
            config={
                "temperature": 1.0,
                "max_output_tokens": 65536,
                "system_instruction": system_prompt
            }
        )

        # Check if response has any content
        if not response or not hasattr(response, 'text'):
            print("Error: API response is empty or invalid")
            return {}

        transcription = response.text
        
        # Debug: Print the raw response to see what we're getting
        print(f"Raw API response length: {len(transcription)}")
        print(f"Raw API response preview: {repr(transcription[:200])}")
        
        # Check if response is empty
        if not transcription or transcription.strip() == "":
            print("Error: API returned empty response. Translation failed.")
            return {}
        
        # Clean markdown from the response
        transcription = clean_markdown_response(transcription)
        
        # Try to clean the response before parsing JSON
        transcription = transcription.strip()
        
        # The model should return a JSON string, which we parse here.
        try:
            translation_map = json.loads(transcription)
        except json.JSONDecodeError as json_error:
            print(f"Error: Failed to parse JSON from API response: {json_error}")
            print(f"Response content: {repr(transcription)}")
            return {}

        if not isinstance(translation_map, dict):
            print("Warning: Translation API did not return a valid dictionary. Translation failed.")
            return {}
        
        print("Successfully received and parsed translations.")
        return translation_map

    except Exception as e:
        print(f"Error: API call failed: {e}")
        print(f"Error type: {type(e).__name__}")
        # Print more details about the exception
        import traceback
        print(f"Full error details: {traceback.format_exc()}")
        return {}

def normalize_romaji(text: str) -> str:
    """
    Normalizes romaji text for robust length comparison.
    Converts macrons to double-vowels and removes hyphens.
    This ensures len("shō") is treated the same as len("shou").
    """
    text = text.lower()
    text = text.replace('ō', 'ou').replace('ū', 'uu').replace('ā', 'aa').replace('ē', 'ei').replace('ī', 'ii')
    text = text.replace('-', '')
    return text

def clean_text_for_matching(text: str) -> str:
    """Clean text for matching by removing punctuation and normalizing."""
    # Remove punctuation and extra whitespace
    cleaned = re.sub(r'[^\w\s]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return normalize_romaji(cleaned)

def parse_time_str(time_str: str) -> float:
    """Converts a MM:SS:ss time string to a float in seconds."""
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3:
            minutes, seconds, centiseconds = parts
            return minutes * 60 + seconds + centiseconds / 100.0
        elif len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds
        else:
            raise ValueError("Invalid time format")
    except (ValueError, IndexError):
        print(f"Warning: Could not parse time string '{time_str}'. Defaulting to 0.0 seconds.")
        return 0.0

def format_time_sec(seconds: float) -> str:
    """Converts a time in seconds to the H:MM:SS.ss format for ASS."""
    if seconds < 0:
        seconds = 0
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    # Using {s:05.2f} ensures the format is always SS.ss, e.g., 09.50
    return f"{int(h):d}:{int(m):02d}:{s:05.2f}"

def generate_ass_header(res_x: int, res_y: int, with_translation: bool = False) -> str:
    """Generates the [Script Info] and [V4+ Styles] sections of the ASS file."""
    header = f"""[Script Info]
; Script generated by karaoke_generator (v_robust_fixed_translated)
Title: Karaoke from Syllable JSON
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {res_x}
PlayResY: {res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Meiryo,65,&H0000FFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,1,0,1,2,1,8,10,10,20,1
"""
    if with_translation:
        # English translation style (Bottom-Center - Alignment: 2)
        header += "Style: Translation,Arial,65,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,0,0,0,0,100,100,0,0,1,1.5,1,2,10,10,15,1\n"
    
    header += """
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    return header

def find_best_syllable_alignment(target_words, all_syllables):
    """
    Find the best alignment between target words and available syllables
    using fuzzy string matching and position-based heuristics.
    """
    syllable_text = ''.join(syl['text'] for syl in all_syllables)
    target_text = ' '.join(target_words)

    clean_syllable_text = clean_text_for_matching(syllable_text)
    clean_target_text = clean_text_for_matching(target_text)

    similarity = SequenceMatcher(None, clean_syllable_text, clean_target_text).ratio()

    if similarity < 0.6:
        print(f"Warning: Low similarity ({similarity:.2f}) between target '{target_text}' and syllable text '{syllable_text}'. Using fallback method.")
        return assign_syllables_sequentially(target_words, all_syllables)

    return assign_syllables_intelligently(target_words, all_syllables)


def assign_syllables_sequentially(target_words, all_syllables):
    """Fallback method: distribute syllables evenly across words."""
    if not all_syllables or not target_words:
        return []

    syllables_per_word = len(all_syllables) / len(target_words) if len(target_words) > 0 else 0
    word_assignments = []
    syllable_idx = 0

    for i, word in enumerate(target_words):
        expected_end = int((i + 1) * syllables_per_word)
        word_syllables = []

        while syllable_idx < len(all_syllables) and syllable_idx < expected_end:
            word_syllables.append(all_syllables[syllable_idx])
            syllable_idx += 1

        if i == len(target_words) - 1:
            while syllable_idx < len(all_syllables):
                word_syllables.append(all_syllables[syllable_idx])
                syllable_idx += 1

        if not word_syllables and syllable_idx < len(all_syllables):
            word_syllables.append(all_syllables[syllable_idx])
            syllable_idx += 1

        if word_syllables:
            word_assignments.append({'word': word, 'syllables': word_syllables})

    return word_assignments

def assign_syllables_intelligently(target_words, all_syllables):
    """Intelligent method: try to match syllables to words based on content."""
    word_assignments = []
    remaining_syllables = all_syllables.copy()

    for word in target_words:
        if not remaining_syllables:
            word_assignments.append({'word': word, 'syllables': []})
            continue

        clean_word = clean_text_for_matching(word)
        word_syllables = []
        current_text = ""

        while remaining_syllables:
            syllable = remaining_syllables[0]
            potential_text = current_text + syllable['text']
            clean_potential = clean_text_for_matching(potential_text)
            
            ratio = SequenceMatcher(None, clean_potential, clean_word).ratio()
            
            if len(clean_potential) <= len(clean_word) * 1.2 and (clean_word.startswith(clean_potential) or ratio > 0.7):
                word_syllables.append(remaining_syllables.pop(0))
                current_text = potential_text
                if len(clean_text_for_matching(current_text)) >= len(clean_word) * 0.8:
                    break
            else:
                break
        
        if not word_syllables and remaining_syllables:
            word_syllables.append(remaining_syllables.pop(0))

        word_assignments.append({'word': word, 'syllables': word_syllables})

    if remaining_syllables and word_assignments:
        word_assignments[-1]['syllables'].extend(remaining_syllables)

    return word_assignments

def clean_markdown_response(response_text: str) -> str:
    """
    Removes markdown formatting from API responses, specifically handling JSON blocks.
    
    Args:
        response_text: The raw response text from the API
        
    Returns:
        Cleaned text with markdown removed
    """
    if not response_text:
        return response_text
    
    # Remove markdown code blocks
    # Handle ```json ... ``` pattern
    if response_text.startswith('```json'):
        # Find the end of the JSON block
        end_marker = response_text.find('```', 7)  # Start after ```json
        if end_marker != -1:
            return response_text[7:end_marker].strip()
    
    # Handle ``` ... ``` pattern (generic code block)
    if response_text.startswith('```'):
        # Find the end of the code block
        end_marker = response_text.find('```', 3)  # Start after first ```
        if end_marker != -1:
            return response_text[3:end_marker].strip()
    
    # Remove any trailing ``` if present
    if response_text.endswith('```'):
        return response_text[:-3].strip()
    
    return response_text.strip()

def create_karaoke_with_wrapping(input_path: str, output_path: str, translate: bool = False, translation_file: str = None, audio_file_path: str = None):
    """
    Reads a syllable-timed JSON and creates a high-quality ASS karaoke file.
    This version correctly handles partial matches and can add translations via API or from a file.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_path}'. Make sure it's valid.")
        return

    translations = {}
    if translation_file:
        # Try to load translations from file (JSON or key-value per line)
        try:
            with open(translation_file, 'r', encoding='utf-8') as tf:
                try:
                    translations = json.load(tf)
                except json.JSONDecodeError:
                    # Fallback: key-value per line, tab or = separated
                    tf.seek(0)
                    translations = {}
                    for line in tf:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '\t' in line:
                            k, v = line.split('\t', 1)
                        elif '=' in line:
                            k, v = line.split('=', 1)
                        else:
                            continue
                        translations[k.strip()] = v.strip()
            print(f"Loaded {len(translations)} translations from file '{translation_file}'.")
            translate = True
        except Exception as e:
            print(f"Error: Could not load translation file '{translation_file}': {e}")
            translations = {}
            translate = False
    elif translate:
        api_key = get_gemini_api_key()
        if not api_key:
            print("Translation disabled because API key is not available.")
            translate = False
        else:
            all_texts_to_translate = [
                item['sentence']['text']
                for item in data if item.get('sentence') and item['sentence'].get('text')
            ]
            if all_texts_to_translate:
                translations = translate_all_lyrics(all_texts_to_translate, api_key, audio_file_path)
            else:
                print("No text found in the input JSON to translate.")

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(generate_ass_header(1920, 1080, with_translation=translate))

        for item_idx, item in enumerate(data):
            sentence_data = item.get('sentence')
            if not sentence_data or not sentence_data.get('letters'):
                continue

            original_sentence_text = sentence_data.get('text', '')
            true_words = [word for word in original_sentence_text.split(' ') if word and word != '=']
            all_syllables = list(sentence_data['letters'])

            if not true_words:
                print(f"Warning: Skipping sentence {item_idx + 1} - no words found in 'text' field.")
                continue

            print(f"Processing sentence {item_idx + 1}: '{original_sentence_text}'")

            word_assignments = find_best_syllable_alignment(true_words, all_syllables)

            if not word_assignments:
                print(f"Warning: Could not create word assignments for sentence {item_idx + 1}")
                continue

            timed_words = []
            for assignment in word_assignments:
                original_word = assignment['word']
                word_syllables = assignment['syllables']

                if not word_syllables:
                    timed_words.append({
                        'text': original_word,
                        'karaoke_str': original_word,
                        'start_time_sec': -1,
                        'end_time_sec': -1,
                    })
                    continue

                timed_syllable_chunk = ""
                for syl in word_syllables:
                    start_sec = parse_time_str(syl['start_time'])
                    end_sec = parse_time_str(syl['end_time'])
                    duration_sec = max(0.01, end_sec - start_sec)
                    duration_cs = int(duration_sec * 100)
                    timed_syllable_chunk += f"{{\\k{duration_cs}}}{syl['text']}"

                syllables_as_string = "".join(s['text'] for s in word_syllables)
                matcher = SequenceMatcher(None, syllables_as_string, original_word, autojunk=False)
                match = matcher.find_longest_match(0, len(syllables_as_string), 0, len(original_word))
                final_karaoke_word = ""
                if match.size > 0:
                    pre_match = original_word[0:match.b]
                    post_match = original_word[match.b + match.size:]
                    final_karaoke_word = pre_match + timed_syllable_chunk + post_match
                else:
                    final_karaoke_word = timed_syllable_chunk
                first_syl_start_sec = parse_time_str(word_syllables[0]['start_time'])
                last_syl_end_sec = parse_time_str(word_syllables[-1]['end_time'])
                timed_words.append({
                    'text': original_word,
                    'karaoke_str': final_karaoke_word,
                    'start_time_sec': first_syl_start_sec,
                    'end_time_sec': last_syl_end_sec
                })

            if not timed_words:
                continue

            lines_to_write = []
            current_line_words = []
            current_line_len = 0

            for i, word in enumerate(timed_words):
                if word['start_time_sec'] == -1:
                    if i > 0 and timed_words[i-1]['end_time_sec'] != -1:
                        word['start_time_sec'] = timed_words[i-1]['end_time_sec']
                        word['end_time_sec'] = timed_words[i-1]['end_time_sec']
                    else:
                        continue
                word_len = len(word['text'])
                space_needed = 1 if current_line_words else 0
                if current_line_words and current_line_len + space_needed + word_len > MAX_CHARS_PER_LINE:
                    lines_to_write.append({
                        'start_time_sec': current_line_words[0]['start_time_sec'],
                        'end_time_sec': current_line_words[-1]['end_time_sec'],
                        'words': current_line_words,
                        'original_sentence': original_sentence_text,
                    })
                    current_line_words = [word]
                    current_line_len = word_len
                else:
                    current_line_words.append(word)
                    current_line_len += space_needed + word_len
            
            if current_line_words:
                sentence_end_time_sec = parse_time_str(sentence_data['end_time'])
                final_end_time = max(sentence_end_time_sec, current_line_words[-1]['end_time_sec'])
                lines_to_write.append({
                    'start_time_sec': current_line_words[0]['start_time_sec'],
                    'end_time_sec': final_end_time,
                    'words': current_line_words,
                    'original_sentence': original_sentence_text,
                })

            for line in lines_to_write:
                start_time = format_time_sec(line['start_time_sec'])
                end_time = format_time_sec(line['end_time_sec'])

                if line['end_time_sec'] <= line['start_time_sec']:
                    line['end_time_sec'] = line['start_time_sec'] + 0.1
                    end_time = format_time_sec(line['end_time_sec'])
                    print(f"Warning: Line with invalid duration adjusted. Start: {start_time}, New End: {end_time}")

                karaoke_line = " ".join([w['karaoke_str'] for w in line['words']])
                dialogue_line = f"Dialogue: 0,{start_time},{end_time},Karaoke,,0,0,0,,{karaoke_line}\n"
                f.write(dialogue_line)

                if translate and translations:
                    translated_text = translations.get(line['original_sentence'], "")
                    if translated_text:
                        translation_dialogue = f"Dialogue: 0,{start_time},{end_time},Translation,,0,0,0,,{translated_text}\n"
                        f.write(translation_dialogue)

    print(f"Successfully created wrapped karaoke ASS file at '{output_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Create line-wrapped, syllable-timed karaoke ASS subtitles from a JSON file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_json", help="Path to the input JSON file.")
    parser.add_argument("-o", "--output", help="Path to the output ASS file. Defaults to the input filename with an .ass extension.")
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Enable translation of lyrics to English using the Gemini API.\n"
             "Requires the GEMINI_API_KEY environment variable to be set."
    )
    parser.add_argument(
        "--translation",
        type=str,
        help="Path to a translation file (JSON or key-value per line). If provided, will be used instead of the API."
    )
    parser.add_argument(
        "--song",
        type=str,
        help="Path to the audio file. If provided with --translate, will be sent to the LLM for translation context."
    )
    args = parser.parse_args()
    output_path = args.output or f"{os.path.splitext(args.input_json)[0]}.ass"
    create_karaoke_with_wrapping(args.input_json, output_path, translate=args.translate or bool(args.translation), translation_file=args.translation, audio_file_path=args.song)

if __name__ == "__main__":
    main()
