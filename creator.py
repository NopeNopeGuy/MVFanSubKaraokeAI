import json
import argparse
import os
import re
import sys
from pathlib import Path
import shutil
import subprocess

                               
try:
    from google import genai
    from google.genai import types
except ImportError:
    pass

                       
TRANSLATION_MODEL = "gemini-2.5-pro"

def get_gemini_api_key() -> str | None:
    """Retrieves the Gemini API key from the GEMINI_API_KEY environment variable."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable not set. Translation feature requires this key.")
    return api_key

def create_system_prompt() -> str:
    """Creates the system prompt for the Gemini translation API call."""
    return """
# Ultimate Japanese-to-English Localization System Prompt v2.0

You are an elite Japanese-to-English localizer with deep understanding of otaku culture, internet slang, emotional nuance, and linguistic adaptation. Your localizations must feel natural to English speakers while preserving Japanese cultural essence.

## SECTION 1: CULTURAL PRESERVATION MATRIX

### 1.1 MANDATORY PRESERVATION LIST (NEVER TRANSLATE)

#### Otaku/Anime Culture Terms
- **oshi** / **oshi-p** / **oshi-pi** → Keep as-is (one's favorite idol/character)
- **senpai** / **kouhai** → Keep as-is (senior/junior relationship)
- **sensei** → Keep as-is (teacher/master)
- **-san**, **-kun**, **-chan**, **-sama** → Keep all honorifics
- **kawaii** → Keep when culturally significant
- **sugoi** → Keep in otaku contexts
- **baka** → Keep when used as endearment/playful
- **tsundere**, **yandere**, **kuudere** → Always keep

#### Cultural Concepts
- **sakura** → Keep when discussing cherry blossoms culturally
- **onigiri**, **bento**, **ramen** → Keep food names
- **yukata**, **kimono** → Keep clothing terms
- **karaoke** → Keep as-is
- **otaku** → Keep as-is

### 1.2 CONTEXTUAL PRESERVATION (TRANSLATE OR KEEP BASED ON CONTEXT)

| Japanese | Keep When | Translate When | Examples |
|----------|-----------|----------------|----------|
| 頑張って (ganbatte) | Encouraging someone in Japanese context | General encouragement | Keep: "Ganbatte, senpai!" / Translate: "I'll do my best" |
| 可愛い (kawaii) | Otaku/anime context | General description | Keep: "That's so kawaii!" / Translate: "She's cute" |
| 先輩 (senpai) | School/work hierarchy | Generic "senior" | Keep: "Notice me, senpai!" / Translate: "my senior colleague" |

## SECTION 2: INTERNET SLANG CONVERSION GUIDE

### 2.1 LAUGHTER VARIATIONS

| Japanese | English Localization | Usage Context |
|----------|---------------------|---------------|
| w | lol | Single, casual laugh |
| ww | lol | Moderate laughter |
| www | lmao | Strong laughter |
| wwwww | LMAOOO | Very strong laughter |
| 草 | lol/lmao | Internet slang for laughter |
| 大草原 | LMFAOOOOO | Extreme laughter |
| (笑) | (laughs) / lol | Parenthetical laughter |
| ワロタ | lmao / "I'm dead" | Slang laughter |
| 笑笑 | lolol | Casual double laugh |

### 2.2 EMOTIONAL EXPRESSIONS

| Japanese | English Localization | Emotional Context |
|----------|---------------------|-------------------|
| やばい | "oh shit" / "damn" / "sick" | Context-dependent |
| まじで | "for real" / "seriously" | Emphasis |
| うざい / うぜぇ | "annoying AF" / "so annoying" | Irritation |
| きもい | "gross" / "creepy" | Disgust |
| だるい | "such a drag" / "tiring" | Exhaustion/annoyance |
| めんどくさい | "what a pain" / "too much effort" | Reluctance |

### 2.3 INTERNET CULTURE EXPRESSIONS

| Japanese | English Localization | Context |
|----------|---------------------|---------|
| 乙 (otsu) | "gg" / "good work" / "thanks" | End of stream/activity |
| どんまい | "no worries!" / "don't sweat it!" | Consolation |
| ｋｗｓｋ | "details pls" / "tell me more" | Requesting info |
| ｇｋｂｒ | *shaking* / "I'm shook" | Fear/nervousness |
| orz | *facepalm* / "I'm dead" / "RIP me" | Defeat/disappointment |
| ＞＜ | >< | Keep emoticon as-is |
| (´・ω・`) | (´・ω・`) | Keep kaomoji as-is |

## SECTION 3: SPEECH PATTERN LOCALIZATION

### 3.1 GENDER-SPECIFIC SPEECH PATTERNS

#### Feminine Speech (女性語)
| Japanese Pattern | English Adaptation | Example |
|-----------------|-------------------|---------|
| ～わ/～わよ/～わね | Softer endings, "you know" | 綺麗だわ → "It's beautiful, you know" |
| ～かしら | "I wonder..." | 来るかしら → "I wonder if they'll come..." |
| あら/まあ | "Oh my" / "Well" | あら、そうなの → "Oh my, is that so?" |
| ～のよ/～なのよ | Explanatory tone | 違うのよ → "That's not it, you see" |

#### Masculine Speech (男性語)
| Japanese Pattern | English Adaptation | Example |
|-----------------|-------------------|---------|
| ～だろ/～だろう | "...right?" / "probably" | 行くだろ → "You're going, right?" |
| ～ぜ/～ぞ | Assertive endings | やるぜ → "Let's do this!" |
| おい | "Hey" / "Yo" | おい、待て → "Yo, wait up" |
| ～じゃねえ | "...ain't" / rougher speech | 違うじゃねえか → "That ain't right!" |

### 3.2 AGE/PERSONALITY SPEECH PATTERNS

#### Childish Speech
| Japanese | English Adaptation | Example |
|----------|-------------------|---------|
| ～なの | Explanatory, innocent | これなの → "This is it!" |
| ～だもん | "...'cause" / "'cuz" | できないもん → "'Cuz I can't!" |
| Repetition | Keep repetition | 痛い痛い → "Owie owie" |

#### Tsundere Speech
| Japanese Pattern | English Adaptation | Example |
|-----------------|-------------------|---------|
| 別に...じゃない | "It's not like..." | 別に嬉しくないし → "It's not like I'm happy or anything" |
| ～んだからね！ | "...you know!" / "...got it?!" | 勘違いしないでよね → "Don't get the wrong idea, got it?!" |
| ばか！ | "Idiot!" / "Dummy!" | Keep the sharpness |

## SECTION 4: EMOTIONAL NUANCE GUIDE

### 4.1 INTENSITY MARKERS

| Japanese | English Localization | Emotional Weight |
|----------|---------------------|------------------|
| ちょっと | "kinda" / "a bit" | Mild |
| かなり | "pretty" / "quite" | Moderate |
| すごく/とても | "really" / "super" | Strong |
| めっちゃ/超 | "hella" / "crazy" / "mad" | Very strong |
| マジで | "seriously" / "for real" | Emphasis |
| ガチで | "legit" / "actually" | Serious emphasis |

### 4.2 UNCERTAINTY/HESITATION

| Japanese | English Localization | Usage |
|----------|---------------------|-------|
| ～かな | "I wonder..." / "...maybe?" | Soft uncertainty |
| ～かも | "might be..." / "probably..." | Possibility |
| たぶん | "probably" / "maybe" | General uncertainty |
| ～と思う | "I think..." / "I feel like..." | Opinion softening |
| ～みたい | "seems like..." / "looks like..." | Observation |

## SECTION 5: SPECIFIC CONTENT TYPE GUIDELINES

### 5.1 SONG LYRICS LOCALIZATION

#### Approach Priority:
1. **Emotional Impact** > Literal Meaning
2. **Flow/Rhythm** > Word-for-word accuracy
3. **Metaphor Adaptation** > Direct translation

#### Examples:
```
Original: 君を守りたくて 守りたくて
Literal: "I want to protect you, want to protect you"
Good Localization: "I wanted to protect, I wanted to protect"
(Maintains repetition and emotional urgency)

Original: 心が痛むほど共鳴するんだ
Literal: "My heart resonates so much it hurts"
Good Localization: "My heart resonates / So much that it hurts"
(Preserves poetic break and emotional weight)
```

### 5.2 DIALOGUE LOCALIZATION

#### Natural Conversation Flow:
```
Original: まじうぜぇあの女
Literal: "Seriously annoying that woman"
Good Localization: "That bitch is so annoying"
(Natural English cursing pattern)

Original: え？そうかな？私のほうが上・・・？
Literal: "Eh? Is that so? I'm better...?"
Good Localization: "What? Really? You think I'm cuter...?"
(Context-aware, natural response)
```

### 5.3 INTERNAL MONOLOGUE

#### Stream of Consciousness:
```
Original: 痛い　痛い　痛い　痛いのよ
Good Localization: "It hurts... It hurts... It hurts... It really hurts!"
(Building intensity)

Original: やめてよ やめてよ なんて
Good Localization: "Stop! Stop! If only I could"
(Natural thought interruption)
```

## SECTION 6: COMPLEX LOCALIZATION SCENARIOS

### 6.1 CULTURAL JOKES/PUNS

| Scenario | Approach | Example |
|----------|----------|---------|
| Untranslatable pun | Adapt to similar English pun or explain briefly | 布団が吹っ飛んだ → "The futon flew away" (keep simple) |
| Cultural reference | Keep with minimal context | 正座で足が痺れた → "My legs fell asleep from sitting seiza" |

### 6.2 ONOMATOPOEIA ADAPTATION

| Japanese | Context | English Adaptation |
|----------|---------|-------------------|
| ドキドキ | Heartbeat | "badump badump" / "thump thump" |
| ワクワク | Excitement | "so pumped" / "getting excited" |
| イライラ | Irritation | "getting pissed" / "so annoyed" |
| ぐずぐず | Dawdling | "dragging feet" / "being wishy-washy" |
| ずぶずぶ | Sinking | "deeper and deeper" / "getting dragged down" |

## SECTION 7: OUTPUT FORMATTING RULES

### 7.1 Line Break Preservation
```
Original:
痛い
痛い
痛いのよ

Good Output:
"It hurts"
"It hurts"
"It really hurts!"
```

### 7.2 Emphasis Preservation
- Multiple exclamation marks: Keep them all (！！！ → !!!)
- Elongated sounds: あぁぁぁ → "Ahhhhh"
- Capitals for shouting: Keep capitals
- Ellipses for pauses: Preserve all dots

### 7.3 Special Formatting
```json
{
  "line_with_break\nhere": "line_with_break\nhere",
  "♪「Song Title」♪": "♪ Song Title ♪",
  "『Quote』": "\"Quote\"" 
}
```

## SECTION 8: QUALITY ASSURANCE CHECKLIST

### Before Finalizing Each Line:
1. ✓ Does it sound natural when spoken aloud?
2. ✓ Are Japanese cultural elements preserved appropriately?
3. ✓ Is the emotional tone matching?
4. ✓ Are character voices consistent?
5. ✓ Is internet slang properly localized?
6. ✓ Are honorifics and cultural terms intact?

## SECTION 9: ADVANCED EXAMPLES

### 9.1 Complex Emotional Scene
```
Original: もう帰るって 傘も差さずに飛び出した君
Literal: "Saying 'I'm going home' you rushed out without even an umbrella"
Good Localization: "You rushed out without even holding an umbrella!"
(Focuses on action and emotion over literal words)
```

### 9.2 Layered Internet Culture
```
Original: マジでチョロすぎｗ どいつもこいつも馬鹿上等
Literal: "Seriously too easy w / Every one of them is perfectly stupid"
Good Localization: "Y'all are so easy lol / Every single one of you are fools"
(Natural English internet speak + attitude)
```

### 9.3 Poetic Imagery
```
Original: 風薫る空の下
Literal: "Under the sky where wind is fragrant"
Good Localization: "Under the summer breeze" / "Under the light breeze of summer"
(Captures poetic feeling in natural English)
```

## SECTION 10: EDGE CASES AND SPECIAL RULES

### 10.1 Mixed Language Input
- English words in Japanese text: Keep as-is
- Romanized Japanese: Convert to proper English meaning
- Brand names: Always keep unchanged

### 10.2 Ambiguous Pronouns
- Japanese often omits subjects: Infer from context
- Gender-neutral "they" when unclear
- Maintain ambiguity if intentional

### 10.3 Cultural Time/Place References
- Japanese school years: "first-year" not "freshman"
- Seasons with cultural weight: "cherry blossom season" for specific spring references
- Place names: Keep Japanese names, add descriptor if needed

## FINAL REMINDERS:
- Prioritize: Naturalness > Accuracy > Literalness
- Preserve: Culture > Style > Exact words
- Adapt: Slang > Formal speech > Technical terms
- Always output ONLY the JSON format requested
- Never explain or justify translation choices in output
- Trust your instincts for natural English flow
    """

def translate_sentences(sentences: list[dict], api_key: str, audio_file_path: str = None) -> None:
    """
    Translates sentences in-place within the data structure if they don't already have a translation.
    """
    sentences_to_translate = [s['text'] for s in sentences if s.get('text') and not s.get('translation')]
    if not sentences_to_translate:
        print("No new sentences to translate.")
        return

    print(f"Found {len(sentences_to_translate)} sentences needing translation.")

    if 'google.genai' not in sys.modules:
        print("Error: Missing required packages for Gemini. Install with: pip install google-genai")
        sys.exit(1)

    # Create client with the new API
    client = genai.Client(api_key=api_key)
    system_prompt = create_system_prompt()
    
    # Preserve original order - create ordered dictionary
    user_prompt_data = {}
    for text in sentences_to_translate:
        user_prompt_data[text] = ""
    
    json_to_translate = json.dumps(user_prompt_data, ensure_ascii=False, indent=2)

    try:
        # Prepare content parts
        content_parts = [
            types.Part.from_text(text="Please translate the lyrics in the following JSON object based on the rules provided in the system instruction."),
            types.Part.from_text(text=f"Input JSON to translate:\n```json\n{json_to_translate}\n```")
        ]
        
        # Upload audio file if provided
        audio_file_gemini = None
        if audio_file_path and Path(audio_file_path).exists():
            print(f"Uploading audio file '{audio_file_path}' for context...")
            try:
                audio_file_gemini = client.files.upload(file=audio_file_path)
                content_parts.insert(0, types.Part.from_text(text="Use the provided audio file to better capture the song's emotion and context in your translation."))
                content_parts.append(types.Part.from_uri(file_uri=audio_file_gemini.uri, mime_type=audio_file_gemini.mime_type))
                print("✓ Audio file uploaded to Gemini.")
            except Exception as e:
                print(f"Warning: Could not upload audio file: {e}")
        
        # Create content structure
        contents = [
            types.Content(
                role="user",
                parts=content_parts
            )
        ]
        
        # Generate content with new API
        response = client.models.generate_content(
            model=TRANSLATION_MODEL,  # gemini-2.5-flash
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.7,
                system_instruction=system_prompt,
                response_mime_type="application/json"
            )
        )
        
        # Extract JSON from response
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Handle the case where response might be an object with candidates
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    response_text = candidate.content.parts[0].text
        
        # Ensure we have a string
        if not isinstance(response_text, str):
            response_text = str(response_text)
        
        # Try to extract JSON
        match = re.search(r'{\s*".*"}', response_text, re.DOTALL)
        if match:
            response_text = match.group(0)
        
        translation_map = json.loads(response_text)

        # Apply translations while preserving original order
        for sentence in sentences:
            if sentence['text'] in translation_map:
                sentence['translation'] = translation_map[sentence['text']]
        
        print("Successfully received and applied translations.")

    except Exception as e:
        print(f"Error: API call failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up uploaded file
        if audio_file_gemini:
            try:
                client.files.delete(name=audio_file_gemini.name)
                print(f"✓ Cleaned up Gemini uploaded file: {audio_file_gemini.name}")
            except Exception as e:
                print(f"Warning: Could not delete Gemini uploaded file {audio_file_gemini.name}: {e}")

def parse_time_str(time_str: str) -> float:
    """Converts a MM:SS:sss time string to a float in seconds."""
    try:
        parts = time_str.split(':')
        return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 1000.0
    except (ValueError, IndexError):
        return 0.0

def format_time_ass(seconds: float) -> str:
    """Converts seconds to H:MM:SS.cs format for ASS subtitles."""
    if seconds < 0: seconds = 0
    m, s = divmod(seconds, 60); h, m = divmod(m, 60)
    return f"{int(h):d}:{int(m):02d}:{int(s):02d}.{int((s - int(s)) * 100):02d}"

def generate_ass_header(res_x: int, res_y: int, with_translation: bool, is_instrumental_karaoke: bool) -> str:
    header = f"[Script Info]\n; Script generated by karaoke_generator (v6_structured_format)\nTitle: Karaoke from Structured JSON\nScriptType: v4.00+\nWrapStyle: 0\nPlayResX: {res_x}\nPlayResY: {res_y}\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
    if is_instrumental_karaoke:
        header += "Style: Karaoke-Active,Arial Black,80,&H002EFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,8,10,10,0,1\n"
        header += "Style: Karaoke-Upcoming,Arial Black,60,&H00FFFFFF,&H00CCCCCC,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,8,10,10,85,1\n"
        header += "Style: Karaoke-Upcoming2,Arial Black,60,&H00DDDDDD,&H00CCCCCC,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,8,10,10,170,1\n"
        header += "Style: Karaoke-Past,Arial Black,60,&H00AAAAAA,&H00CCCCCC,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,8,10,10,0,1\n"
    else:
        header += "Style: Karaoke,Trebuchet MS,65,&H002EFFFF,&H00FFFFFF,&H00800000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,10,10,30,1\n"
        if with_translation:
            header += "Style: Translation,Verdana,48,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,-1,0,0,100,100,0,0,1,2,1,8,10,10,25,1\n"
    header += "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    return header

                                                                                
                                                                
                                                                                

def create_karaoke_from_structured_json(input_path: str, output_path: str, translate: bool, audio_file_path: str | None, is_instrumental_karaoke: bool, update_json: bool):
    """
    Reads a well-structured, hierarchical JSON and creates an ASS file.
    This logic is simple and reliable due to the superior data format.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        sentences = data.get('sentences', [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing structured JSON file '{input_path}': {e}")
        return

                                 
    if translate and not is_instrumental_karaoke:
        api_key = get_gemini_api_key()
        if api_key:
            translate_sentences(sentences, api_key, audio_file_path)
            if update_json:
                                                                              
                try:
                    with open(input_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print(f"Updated JSON file with new translations at '{input_path}'")
                except Exception as e:
                    print(f"Warning: Could not save updated JSON file: {e}")
        else:
            print("Translation skipped: API key not available.")
    
                                                     
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        has_any_translation = any(s.get('translation') for s in sentences)
        f.write(generate_ass_header(1920, 1080, has_any_translation, is_instrumental_karaoke))

        prev_sentence_end_s = 0.0
        for i, sentence in enumerate(sentences):
            words = sentence.get('words', [])
            if not words:
                continue

                                                                         
                                                             
                                                    
            try:
                start_time_s = parse_time_str(words[0][1])
                end_time_s = parse_time_str(words[-1][2])
            except (IndexError, TypeError):
                print(f"Warning: Skipping malformed sentence due to missing time data: {sentence.get('text')}")
                continue

            if start_time_s >= end_time_s: continue

            start_time_ass = format_time_ass(start_time_s)
            end_time_ass = format_time_ass(end_time_s)

                                             
            karaoke_parts = []
            for word_data in words:
                try:
                    word_text = word_data[0]
                    syllables = word_data[3] if len(word_data) > 3 else []
                    
                    if not syllables:
                        karaoke_parts.append(word_text)
                    else:
                        syllable_k_parts = []
                        for s in syllables:
                                                     
                            duration_cs = max(1, int((parse_time_str(s[2]) - parse_time_str(s[1])) * 100))
                            syllable_k_parts.append(f"{{\\k{duration_cs}}}{s[0]}")
                        karaoke_parts.append("".join(syllable_k_parts))
                except (IndexError, TypeError):
                    if isinstance(word_data, list) and len(word_data) > 0:
                        karaoke_parts.append(word_data[0])
                    continue
            
            karaoke_text = " ".join(karaoke_parts)

            if is_instrumental_karaoke:
                # Line appears at the end of the previous line, but karaoke highlighting is delayed until its actual start time.
                appearance_time_s = prev_sentence_end_s if i > 0 and prev_sentence_end_s < start_time_s else start_time_s
                delay_cs = max(0, int((start_time_s - appearance_time_s) * 100))
                
                line_start_ass = format_time_ass(appearance_time_s)
                line_end_ass = format_time_ass(end_time_s)

                final_karaoke_text = f"{{\\k{delay_cs}}}{karaoke_text}" if delay_cs > 0 else karaoke_text
                
                # Active line
                f.write(f"Dialogue: 0,{line_start_ass},{line_end_ass},Karaoke-Active,,0,0,0,,{{\\pos(960,100)}}{final_karaoke_text}\n")

                # Past and upcoming lines are displayed for the same duration as the main line.
                if i > 0:
                    past_text = sentences[i-1].get('text', '')
                    f.write(f"Dialogue: 0,{line_start_ass},{line_end_ass},Karaoke-Past,,0,0,0,,{{\\pos(960,40)}}{past_text}\n")
                if i + 1 < len(sentences):
                    next_text = sentences[i+1].get('text', '')
                    f.write(f"Dialogue: 0,{line_start_ass},{line_end_ass},Karaoke-Upcoming,,0,0,0,,{{\\pos(960,185)}}{next_text}\n")
                if i + 2 < len(sentences):
                    next2_text = sentences[i+2].get('text', '')
                    f.write(f"Dialogue: 0,{line_start_ass},{line_end_ass},Karaoke-Upcoming2,,0,0,0,,{{\\pos(960,270)}}{next2_text}\n")
            else:
                f.write(f"Dialogue: 0,{start_time_ass},{end_time_ass},Karaoke,,0,0,0,,{karaoke_text}\n")
                
                translation = sentence.get('translation', '').strip()
                if translation:
                    f.write(f"Dialogue: 0,{start_time_ass},{end_time_ass},Translation,,0,0,0,,{translation}\n")
            
            prev_sentence_end_s = end_time_s
    
    mode_message = "instrumental karaoke" if is_instrumental_karaoke else "karaoke"
    print(f"Successfully created {mode_message} ASS file at '{output_path}' from structured JSON.")

                                                                                
                                                          
                                                                                
def get_audio_duration(file_path: str) -> float:
    if not shutil.which("ffprobe"): return 0.0
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
    try:
        return float(subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip())
    except Exception: return 0.0

def generate_karaoke_video(instrumental_audio_path: str, instrumental_ass_path: str, output_video_path: str):
    if not shutil.which("ffmpeg"): print("Error: ffmpeg not found."); return
    print(f"\n--- Generating Karaoke Video ---")
    duration = get_audio_duration(instrumental_audio_path)
    if duration == 0.0: print("Aborting video creation."); return
    escaped_ass_path = Path(instrumental_ass_path).as_posix().replace(":", "\\:")
    command = ["ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=10", "-i", instrumental_audio_path, "-vf", f"ass='{escaped_ass_path}'", "-t", str(duration + 2.0), "-c:v", "libx264", "-tune", "stillimage", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-y", output_video_path]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"Successfully created karaoke video: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: ffmpeg failed.\n{e.stderr}")

                                                                                
                                 
                                                                                

def main():
    parser = argparse.ArgumentParser(
        description="Create karaoke ASS subtitles from a well-structured hierarchical JSON file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_json", help="Path to the structured input JSON file.")
    parser.add_argument("-o", "--output", help="Path to the output ASS file. Defaults to input filename with .ass extension.")
    parser.add_argument("--translate", action="store_true", help="Translate sentences that don't have a translation. Requires GEMINI_API_KEY.")
    parser.add_argument("--update-json", action="store_true", help="If --translate is used, save the new translations back into the input JSON file.")
    parser.add_argument("--song", help="Path to the audio file. Used for translation context if --translate is active.")
    parser.add_argument("--instrumental-karaoke", action="store_true", help="Generate a second ASS file with large, multi-line subtitles.")
    parser.add_argument("--generate-video", action="store_true", help="If --instrumental-karaoke is used, also generate an MP4 video.\nRequires an 'instrumental.mp3' (or .wav, etc.) in the same directory.")
    args = parser.parse_args()
    
                                   
    output_path = args.output or str(Path(args.input_json).with_suffix('.ass'))
    print("--- Generating Standard Karaoke File ---")
    create_karaoke_from_structured_json(
        args.input_json, 
        output_path, 
        translate=args.translate, 
        audio_file_path=args.song,
        is_instrumental_karaoke=False,
        update_json=args.update_json
    )

                                                            
    if args.instrumental_karaoke:
        p = Path(args.input_json)
        instrumental_ass_path = p.with_name(f"{p.stem}_instrumental.ass")
        print("\n--- Generating Instrumental Karaoke File ---")
        create_karaoke_from_structured_json(
            args.input_json,
            str(instrumental_ass_path),
            translate=False,                                           
            audio_file_path=None,
            is_instrumental_karaoke=True,
            update_json=False
        )

        if args.generate_video:
            parent_dir = p.parent
            found_audio = next((parent_dir / f"instrumental{ext}" for ext in ['.mp3', '.wav', '.flac', '.opus', '.m4a'] if (parent_dir / f"instrumental{ext}").exists()), None)
            if found_audio:
                output_video_path = p.with_name(f"{p.stem}_karaoke_video.mp4")
                generate_karaoke_video(str(found_audio), str(instrumental_ass_path), str(output_video_path))
            else:
                print(f"\nWarning: Video generation skipped. Could not find 'instrumental' audio file (e.g., instrumental.mp3) in '{parent_dir}'.")

if __name__ == "__main__":
    main()