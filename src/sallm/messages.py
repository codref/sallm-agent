DEFAULT_MODEL = "ollama/gemma4:e4b-it-qat"
DEFAULT_API_BASE = "http://localhost:11434"


def user(content):
    return {"role": "user", "content": content}


def assistant(content):
    return {"role": "assistant", "content": content}


def system(content):
    return {"role": "system", "content": content}
