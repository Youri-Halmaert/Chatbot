import openai
import os
from transformers import pipeline

class LLM:
    def __init__(self, mode="openai", api_key=None, model_name="gpt-4o", local_model="mistralai/Mistral-7B-Instruct-v0.2"):
        # You can change model_name to "gpt-4o" or any model you have access to.
        self.mode = mode
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.local_model = local_model
        if mode == "local":
            self.generator = pipeline("text-generation", model=local_model)
        if mode == "openai":
            self.client = openai.OpenAI(api_key=self.api_key)

    def generate(self, prompt):
        if self.mode == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"OpenAI API error: {e}"
        else:
            output = self.generator(prompt, max_new_tokens=256)
            return output[0]["generated_text"]
