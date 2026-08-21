import modal

app = modal.App("voice-rag-e2e-test")

@app.local_entrypoint()
def main():
    print("Testing VoiceRAG deployed instance on Modal Cloud...")
    vr_cls = modal.Cls.from_name("voice-rag", "VoiceRAG")
    vr = vr_cls()
    
    print("\n1. Running English Query: 'what county is columbus city in'...")
    res = vr._run_query.remote("what county is columbus city in")
    print("   Answer:", res.get("answer"))
    print("   Grounded:", res.get("grounded"))
    print("   Timings:", res.get("timings_ms"))
    
    print("\n2. Running Hindi Query: 'ब्राइटन टाउनशिप फोन नंबर'...")
    res = vr._run_query.remote("ब्राइटन टाउनशिप फोन नंबर")
    print("   Answer:", res.get("answer"))
    print("   Grounded:", res.get("grounded"))
    print("   Timings:", res.get("timings_ms"))

    print("\n3. Running Marathi Query: 'फ्रान्सचे सध्याचे चलन काय आहे'...")
    res = vr._run_query.remote("फ्रान्सचे सध्याचे चलन काय आहे")
    print("   Answer:", res.get("answer"))
    print("   Grounded:", res.get("grounded"))
    print("   Timings:", res.get("timings_ms"))
