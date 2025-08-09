from google import genai
from prompt import query_prompt
from dotenv import load_dotenv
import json
from dotenv import get_key
import re
import sys

load_dotenv()
api_key = get_key(".env", "GOOGLE_API_KEY")

if not api_key:
    print("Please put your api_key in .env")
    sys.exit(0)

client = genai.Client(api_key=api_key)

def extract_json_object(text):
    # incase there ends up being ``` type fences
    text = re.sub(r"^```.*\n", "", text, flags=re.MULTILINE)
    # same for at the last
    text = re.sub(r"\s*```$", "", text.strip())
    # only the matching portion in { }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        raise ValueError(f"No JSON object found in model response: {text}")

def check_inconsistencies(formatted_data):
    #pre-made prompt in prompt.py
    prompt = query_prompt(formatted_data)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
        raw_text = response.candidates[0].content.parts[0].text
        clean_json_string = extract_json_object(raw_text)
        result_dict = json.loads(clean_json_string)
    except (AttributeError, IndexError, json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Unexpected response structure or JSON parse error: {response}") from e
    
    return result_dict
