import anthropic
from environs import Env

env = Env()
env.read_env()
claude_key = env.str('CLAUDE')
system_prompt = f'You are assistant that carefully answer the question about code generation.'


def call_anthropic_api(message, system_prompt=system_prompt, max_tokens=20, model="claude-3-5-sonnet-20240620"):
    api_client = anthropic.Anthropic(
        api_key=claude_key,
    )
    system_prompt = system_prompt
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

