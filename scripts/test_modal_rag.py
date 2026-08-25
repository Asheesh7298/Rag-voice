import modal

app = modal.App("test-query-app")

@app.local_entrypoint()
def main():
    rag_cls = modal.Cls.from_name("voice-rag", "VoiceRAG")
    rag = rag_cls()
    print("Calling VoiceRAG.query remotely on A100...")
    res = rag.query.remote("what county is columbus city in")
    print("Remote result:", res)
