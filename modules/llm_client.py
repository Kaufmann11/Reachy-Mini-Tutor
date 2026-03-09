from openai import OpenAI

client = OpenAI()

def get_response(system_prompt: str, messages: list[dict]) -> str:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": system_prompt}, *messages],
        temperature=0.7,
        max_tokens=250,
    )
    return resp.choices[0].message.content.strip()

