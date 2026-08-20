with open('modal_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """    def _run_query(self, query: str) -> dict:
        import time, re
        timings = {}
        t_start = time.perf_counter()

        q_lower = query.lower().strip()

        # Direct High-Accuracy Fact Grounding (<0.1ms) for Core World Knowledge Queries
        CORE_FACTS = [
            (["भारत", "राजधानी"], "भारत की राजधानी नई दिल्ली है।", "hi"),
            (["भारताची", "राजधानी"], "भारताची राजधानी नवी दिल्ली आहे.", "mr"),
            (["capital", "india"], "The capital of India is New Delhi.", "en"),
            (["capital", "france"], "The capital of France is Paris.", "en"),
            (["फ्रांस", "राजधानी"], "फ्रांस की राजधानी पेरिस है।", "hi"),
            (["photosynthesis"], "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.", "en"),
            (["प्रकाश", "संश्लेषण"], "प्रकाश संश्लेषण वह प्रक्रिया है जिसके द्वारा हरे पौधे सूर्य के प्रकाश, जल और कार्बन डाइऑक्साइड का उपयोग करके भोजन और ऑक्सीजन का निर्माण करते हैं।", "hi"),
            (["प्रकाशसंश्लेषण"], "प्रकाशसंश्लेषण ही अशी प्रक्रिया आहे ज्याद्वारे हिरव्या वनस्पती सूर्यप्रकाश, पाणी आणि कार्बन डायऑक्साईड वापरून अन्न आणि ऑक्सिजन तयार करतात.", "mr"),
            (["symptoms", "diabetes"], "Common symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, unexplained weight loss, and blurred vision.", "en"),
            (["मधुमेह", "लक्षण"], "मधुमेह के प्रमुख लक्षणों में अत्यधिक प्यास लगना, बार-बार पेशाब आना, अत्यधिक थकान, वजन कम होना और धुंधला दिखाई देना शामिल हैं।", "hi"),
            (["मधुमेहाची", "लक्षणे"], "मधुमेहाच्या मुख्य लक्षणांमध्ये जास्त तहान लागणे, वारंवार लघवी होणे, खूप थकवा येणे आणि अस्पष्ट दृष्टी यांचा समावेश होतो.", "mr"),
            (["blood", "pressure"], "A normal blood pressure reading for adults is typically less than 120/80 mm Hg.", "en"),
            (["रक्तचाप", "सामान्य"], "वयस्कों के लिए सामान्य रक्तचाप 120/80 mm Hg से कम माना जाता है।", "hi"),
        ]
        for keywords, ans_text, flang in CORE_FACTS:
            if all(kw in q_lower for kw in keywords):
                timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
                return {
                    "query": query, "transcript": None,
                    "answer": ans_text,
                    "sources": [{"text": ans_text, "score": 1.0, "lang": flang, "lang_name": "Fact Grounding", "strategy": "grounded_fact"}],
                    "confidence": 0.99, "grounded": True,
                    "guardrail_triggered": None, "timings_ms": timings,
                    "lang_detected": flang,
                }

        # Guardrail 1 — unsafe input (regex, ~0ms)"""

replacement = """    def _run_query(self, query: str) -> dict:
        import time, re
        timings = {}
        t_start = time.perf_counter()

        # Guardrail 1 — unsafe input (regex, ~0ms)"""

if target in content:
    content = content.replace(target, replacement)
    with open('modal_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully removed CORE_FACTS from modal_app.py!")
else:
    print("Target string not found, check matching.")
