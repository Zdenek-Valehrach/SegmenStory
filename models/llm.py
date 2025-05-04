import sys
import os
from pathlib import Path
import requests
import re
import openai
from config import PER_API_TOKEN, OPENAI_API_KEY

# """Absolutní cesta ke kořenovému adresáři projektu pro testování v terminálu"""
# project_root = Path(__file__).resolve().parent.parent  # ← o úroveň výš než models/
# sys.path.append(str(project_root))

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
            generated_text = response.json()["choices"][0]["message"]["content"]
            # Odstranění samostatných [číslo] výskytů (ne na konci věty)
            cleaned_text = re.sub(r'\[(?!\^)\d+\]', '', generated_text)
            return cleaned_text
        else:
            raise Exception(f"Chyba API {response.status_code}: {response.text}")

class OpenAILLM:
    def __init__(self, api_key=None, model="gpt-4o"):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(self, prompt, max_tokens=2000, temperature=0.7):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Jsi největší odborník na evoluční antropologii s neodolatelným smyslem pro humor. Tvé znalosti sahají od prehistorických nástrojů po moderní technologie a vždy dokážeš vykouzlit úsměv na tváři."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            generated_text = response.choices[0].message.content
            cleaned_text = re.sub(r'\[(?!\^)\d+\]', '', generated_text)
            return cleaned_text
        except Exception as e:
            raise Exception(f"Chyba OpenAI API: {str(e)}")


# """Testování v terminálu"""
# if __name__ == "__main__":
#     from utils.prompt_utils import PromptBuilder
    
#     # Volba API přes proměnnou prostředí
#     use_openai = os.getenv("TEST_OPENAI", "0") == "1"
    
#     builder = PromptBuilder()
#     segment = "toustovač"
#     prompt = (
#         f"Vymysli vtipný příběh vývoje tohoto \"{segment}\" jeho vliv na historii lidstva, a jakou roli by mohl hrát v budoucnosti. "
#         "Využij svůj osobitý humor a znalosti o evoluční antropologii. Ať je příběh vtipný a maximálně na půl stránky."
#     )

#     if use_openai:
#         print("=== TESTUJI OPENAI ===")
#         llm = OpenAILLM()
#     else:
#         print("=== TESTUJI PERPLEXITY ===")
#         llm = PerplexityLLM()

#     print(llm.generate(prompt))