import dspy

def setup_dspy():
    lm = dspy.LM(
        model="ollama_chat/llama3.1:8b",
        api_base="http://localhost:11434",
    )

    dspy.configure(lm=lm)

    return lm