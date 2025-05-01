import sys
import os
from pathlib import Path
import requests
import re

# Přidání kořenového adresáře do Python cesty pro testování v terminálu
# current_dir = Path(__file__).resolve().parent  # Adresář models/
# project_root = current_dir.parent  # Kořenový adresář SegmenStory/
# sys.path.append(str(project_root))

# Nyní lze importovat config z kořene
from config import PER_API_TOKEN

class PerplexityLLM:
    def __init__(self, api_key=None, model="sonar"):
        self.api_key = api_key or PER_API_TOKEN
        self.model = model
        self.base_url = "https://api.perplexity.ai/chat/completions"

    def generate(self, prompt, max_tokens=2000, temperature=0.7):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Jsi největší odborník na evoluční antropologii s neodolatelným smyslem pro humor. Tvé znalosti sahají od prehistorických nástrojů po moderní technologie a vždy dokážeš vykouzlit úsměv na tváři."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            # return response.json()["choices"][0]["message"]["content"]
            generated_text = response.json()["choices"][0]["message"]["content"]
            # Odstranění samostatných [číslo] výskytů (ne na konci věty)
            cleaned_text = re.sub(r'\[(?!\^)\d+\]', '', generated_text)
            return cleaned_text
        else:
            raise Exception(f"Chyba API {response.status_code}: {response.text}")

# Testování v terminálu
# if __name__ == "__main__":
#     from utils.prompt_utils_2 import PromptBuilder

#     builder = PromptBuilder()
#     segment = "příborník"
#     prompt = (
#         f"Vymysli vtipný příběh vývoje tohoto \"{segment}\" jeho vliv na historii lidstva, a jakou roli by mohl hrát v budoucnosti. Využij svůj osobitý humor a znalosti o evoluční antropologii. Ať je příběh vtipný a maximálně na půl stránky."
#     )
#     llm = PerplexityLLM()
#     print(llm.generate(prompt))
