from google import genai
from google.genai.types import GenerateContentConfig
from prompt import query_prompt
from dotenv import load_dotenv
import json
from dotenv import get_key
import re
import sys
import os
import logging

load_dotenv()
api_key = get_key(".env", "API_KEY")

if not api_key:
    print("Please put your api_key in .env")
    sys.exit(0)

client = genai.Client(api_key=api_key)
logger = logging.getLogger(__name__)

# def extract_json_object(text):
#     # incase there ends up being ``` type fences
#     text = re.sub(r"^```.*\n", "", text, flags=re.MULTILINE)
#     # same for at the last
#     text = re.sub(r"\s*```$", "", text.strip())

#     # windows specific problem apparently
#     if os.name == "nt":
#         text = text.replace("\r\n", "\n").replace("\r", "\n")
#         # Remove any BOM or zero-width spaces just in case
#         text = text.replace("\ufeff", "").replace("\u200b", "")


#     # only the matching portion in { }
#     match = re.search(r"\{.*\}", text, re.DOTALL)
#     if match:
#         return match.group(0)
#     else:
#         raise ValueError(f"No JSON object found in model response: {text}")


def _fix_single_quote_values_and_keys(s: str) -> str:
    # keys like: {'key':  -> {"key":
    s = re.sub(r"(?<=\{|,)\s*'([^']+)'\s*:", r'"\1":', s)
    # values like: : 'value' , or : 'value'}
    s = re.sub(r":\s*'([^']*)'\s*(?=,|\})", r': "\1"', s)
    return s

def _fix_python_literals(s: str) -> str:
    # Convert Python-style literals into JSON literals (none->null etc)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    return s

def _remove_wrapping_quotes(s: str) -> str:
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    return s

def _strip_control_chars(s: str) -> str:
    # keep tab (0x09), LF (0x0A), CR (0x0D); drop the rest ≤0x1F
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', s)

def _first_json_block(text: str) -> str | None:
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        return obj_match.group(0)

    arr_match = re.search(r"\[.*\]", text, re.DOTALL)
    return arr_match.group(0) if arr_match else None

def _strip_trailing_commas(s: str) -> str:
    # remove trailing commas before } or ] (common sanitizationp problem)
    s = re.sub(r",\s*\}", "}", s)
    s = re.sub(r",\s*\]", "]", s)
    return s

def _remove_bom_and_cr(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\u200b", "")
    # normalize newlines on windows (windows sepficially)
    if os.name == "nt":
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text

#### DO NOT REMOVE. Literally breaks everything if not included:
def _remove_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*\r?\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\r?\n```$", "", text.strip())
    return text

def try_load_json_from_text(text: str):

    # if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
    #     text = text[1:-1]
    text = _remove_bom_and_cr(text)
    text = _strip_control_chars(text)    

    text = _remove_code_fences(text)


    text = text.strip()
    if text.startswith("\\'") and text.endswith("\\'"):
        text = text[1:-1]
    if text.startswith('\\"') and text.endswith('\\"'):
        text = text[1:-1]
    text = text.replace("\\'", "'").replace('\\"', '"')

    text = _remove_wrapping_quotes(text)
    text = _fix_python_literals(text)


    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass  

    candidate = _first_json_block(text)
    if not candidate:
        return None

    candidate = _remove_wrapping_quotes(candidate)

    # attempt to parse raw candidate first
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass


    ## Just a bunch of common sanitization problem:

    # remove trailing commas
    candidate = _strip_trailing_commas(candidate)

    # fix Python literals (None/True/False)
    candidate = _fix_python_literals(candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # attempt to fix single-quoted keys/values conservatively
    candidate = _fix_single_quote_values_and_keys(candidate)
    candidate = candidate.replace("\\'", "'")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e_final:
        return None


def check_inconsistencies(formatted_data):
    #pre-made prompt in prompt.py
    prompt = query_prompt(formatted_data)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
        raw_text = response.candidates[0].content.parts[0].text
        result_dict = try_load_json_from_text(raw_text)
    except (AttributeError, IndexError, json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Unexpected response structure or JSON parse error: {response}") from e
    
    return result_dict


#Incase the number of pptx slides are too long!
def preprocess_slides_with_llm(raw_collected_data):

    SYSTEM_MSG = (
        "You are a function that returns ONE JSON object. "
        "Do NOT wrap it in quotes or code fences. "
        "Do NOT output a top-level array—always an object that begins with '{'. "
        "No extra text."
    )

    prompt = f"""
    You are a function that outputs ONLY minified JSON matching the
    given schema. No markdown, no explanations, no code fences. 

    You will be given a JSON object extracted from a PowerPoint file. Your job:
    - Keep all meaningful text (slide titles, bullet points, OCR text from images, key data from tables).
    - Remove any fields that are null, empty, or obviously redundant.
    - Keep the same high-level keys (slide_titles, texts, images, charts, tables, notes, elements),
      but only if they contain actual information.
    - For verbose text blocks, condense them while keeping the meaning.
    - Maintain valid JSON format.

    OUTPUT RULES (strict):
    - Your final output MUST be only the JSON object as a string with no additional formatting, code fences, or commentary.
    - The ONLY output permitted is a single JSON object starting with '{' and ending with '}'. Nothing else.
    - Do NOT use code fences, do NOT add any commentary, explanation or quotation marks.
    - If your output contains anything else, it is invalid.
    - Immediately start your output with '{' and end with '}'.

    Here is the JSON to reduce:
    {json.dumps(raw_collected_data, ensure_ascii=False)}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(
            temperature=0
        )
    )

    try:
        raw_text = response.candidates[0].content.parts[0].text

        # print("\n===== RAW MODEL OUTPUT =====")
        # print(raw_text)
        # print("===== END RAW MODEL OUTPUT =====\n")


        result_dict = try_load_json_from_text(raw_text)

        if isinstance(result_dict, list):
            result_dict = {"slides": result_dict}

        # print("\n===== RAW MODEL OUTPUT =====")
        # print(result_dict)
        # print("===== END RAW MODEL OUTPUT =====\n")

        if not isinstance(result_dict, dict):
            raise ValueError("LLM did not return a JSON object")
        return result_dict
    except Exception as e:
        raise ValueError(f"Error during pre-processing: {e}")