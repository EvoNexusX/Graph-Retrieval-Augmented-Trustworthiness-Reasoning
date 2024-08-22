from openai import OpenAI

temperature = 0.2
method = 1
if method == 0:
    base_url = "https://api.xiaoai.plus/v1"
    api_key = "sk-lfHDOEV9YpiZsS5SDc2fA0475f754eC0A6Ff72EdAc7f0aB5"
    model = "gpt-3.5-turbo"
else:
    base_url = "http://localhost:11434/v1/"
    api_key = "ollama"
    model = "llama3"
max_tokens = 200

client = OpenAI(base_url=base_url,
                api_key=api_key,
                )
completion = client.chat.completions.create(
    model=model,
    messages=[{"role":"user","content":"hello"}],
    temperature=temperature,
    max_tokens=max_tokens,
)
print(completion)
