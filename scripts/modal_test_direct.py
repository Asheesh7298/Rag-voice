import modal

app = modal.App("voice-rag-direct-test")

@app.local_entrypoint()
def main():
    print("Connecting to deployed VoiceRAG on Modal Cloud...")
    vr_cls = modal.Cls.from_name("voice-rag", "VoiceRAG")
    vr = vr_cls()
    
    queries = [
        ("what county is columbus city in", "EN"),
        ("ब्राइटन टाउनशिप फोन नंबर", "HI"),
        ("फ्रान्सचे सध्याचे चलन काय आहे", "MR"),
        ("who won the cricket match on mars tomorrow", "Off-Topic Refusal"),
    ]
    
    for q, label in queries:
        print(f"\n--- Testing [{label}]: {q!r} ---")
        try:
            res = vr._run_query.remote(q)
            print("Answer:", res.get("answer"))
            print("Grounded:", res.get("grounded"), "| Guardrail:", res.get("guardrail_triggered"))
            print("Timings:", res.get("timings_ms"))
        except Exception as e:
            print("Error:", e)
