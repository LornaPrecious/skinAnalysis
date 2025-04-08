from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import os

model_name = "microsoft/phi-1_5"  # "microsoft/phi-1_5" "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

def get_recommendation(skin_condition: str) -> str:
    prompt = f"Give skincare advice for someone with {skin_condition}. Keep it medically accurate and easy to follow."

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id  
    )

    recommendation = tokenizer.decode(output[0], skip_special_tokens=True)
    return recommendation.strip()


"""
# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1", trust_remote_code=True)
pipe(messages)     Copy  # Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)

"""

client = openai.OpenAI()

openai.api_key = os.getenv("OPENAI_API_KEY")  # Or directly assign the key (not recommended for production)

def get_skin_recommendation(skin_condition):
    prompt = f"""I have been diagnosed with {skin_condition}. 
    Please provide gentle skincare tips, over-the-counter treatments, and lifestyle changes that can help manage or reduce symptoms."""

    response = client.completions.create(
        model="gpt-3.5-turbo",  # Or "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful and knowledgeable skincare assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7,
    )

    advice = response['choices'][0]['message']['content']
    return advice

def gpt_recommendation(condition):
    prompt = f"""What are the best recommendations for someone with {condition}? 
    Please provide gentle skincare tips, over-the-counter treatments, and lifestyle changes that can help manage or reduce symptoms.  
    """
    
    response = client.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if you’re using it
        stream_options=[
            {"role": "system", "content": "You are a helpful and knowledgeable dermatologist and skincare assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7,
        prompt=prompt
    )
    
    return response['choices'][0]['message']['content'].strip()

