import re

import anthropic
import json
from environs import Env
from openai import OpenAI

env = Env()
env.read_env()
claude_key = env.str('CLAUDE_KEY')
openai_key = env.str('OPENAI_KEY')

system_prompt = f'You are assistant that carefully answer the question about code generation.'


def call_anthropic_api(message, max_tokens=20, model="claude-3-5-sonnet-20240620"):
    api_client = anthropic.Anthropic(api_key=claude_key)
    try:
        message = api_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=1,
            system=system_prompt,
            messages=[
                {"role": "user", "content": message}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


def call_openai_api(message, max_tokens=150, model="gpt-4-turbo"):
    client = OpenAI(api_key=openai_key)
    try:
        completion = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
        )
        text = completion.choices[0].message.content
        pattern = r'\{[^{}]*\}'
        json_object = re.findall(pattern, text, re.VERBOSE | re.DOTALL)[0]
        response = json.loads(json_object)
        return response
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
