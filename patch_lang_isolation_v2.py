with open('modal_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target1 = """    def _resolve_lang_filter(self, lang_filter: str | list | None):
        \"\"\"Helper to resolve language codes/groups to a set of matching languages.\"\"\"
        if not lang_filter:
            return None
        if isinstance(lang_filter, (list, tuple, set)):
            return set(lang_filter)
        if lang_filter in ("devanagari_group", "hi_mr"):
            return {"hi", "mr", "ne"}
        if lang_filter in ("bengali_group",):
            return {"bn", "as"}
        return {lang_filter}"""

replacement1 = """    def _resolve_lang_filter(self, lang_filter: str | list | None):
        \"\"\"Helper to resolve language codes/groups to a set of matching languages.\"\"\"
        if not lang_filter:
            return None
        if isinstance(lang_filter, (list, tuple, set)):
            return set(lang_filter)
        if lang_filter in ("devanagari_group", "hi_mr"):
            return {"hi", "mr"}
        if lang_filter in ("bengali_group",):
            return {"bn", "as"}
        return {lang_filter}"""

target2 = """    def _detect_lang(self, text: str):
        \"\"\"Detect the Indic language family from Unicode script ranges.

        Hindi, Marathi, and Nepali share Devanagari, while Bengali and Assamese
        share the Bengali script.  Returning a family lets retrieval keep those
        ambiguous pairs together without treating every non-Latin query as
        English.
        \"\"\"
        script_ranges = (
            (0x0900, 0x097F, "devanagari_group"),
            (0x0980, 0x09FF, "bengali_group"),
            (0x0A00, 0x0A7F, "pa"),
            (0x0A80, 0x0AFF, "gu"),
            (0x0B00, 0x0B7F, "or"),
            (0x0B80, 0x0BFF, "ta"),
            (0x0C00, 0x0C7F, "te"),
            (0x0C80, 0x0CFF, "kn"),
            (0x0D00, 0x0D7F, "ml"),
            (0x0600, 0x06FF, "ur"),
        )
        counts = {name: 0 for _, _, name in script_ranges}
        for ch in text:
            cp = ord(ch)
            for start, end, name in script_ranges:
                if start <= cp <= end:
                    counts[name] += 1
                    break
        if not counts:
            return "en"
        best = max(counts, key=counts.get)
        return best if counts[best] >= 2 else "en" """

replacement2 = """    def _detect_lang(self, text: str):
        \"\"\"Detect the exact Indic language from Unicode script ranges and lexical markers.\"\"\"
        # 1. Lexical markers for Devanagari disambiguation
        q_words = set(text.lower().split())
        mr_markers = {"आहे", "नाही", "म्हणजे", "काय", "कसे", "कोणती", "कोणते", "कोणता", "कुठे", "झाले", "केले", "मधील", "मध्ये", "यांचे", "त्यांचे", "आणि", "कशी", "किती", "असावा", "करावे", "कोणत्या"}
        hi_markers = {"है", "हैं", "नहीं", "क्या", "कैसे", "कौन", "कहाँ", "हुआ", "किया", "किए", "होता", "होती", "होते", "और", "में", "पर", "से", "का", "की", "के", "कितना", "कितनी", "चाहिए", "देता", "रहते", "पाया"}
        ne_markers = {"हो", "छ", "छैन", "गर्छ", "गरेको", "कस्तो", "कहाँ", "र", "को", "का", "मा", "हुन्छ"}

        if any(w in q_words for w in mr_markers):
            return "mr"
        if any(w in q_words for w in hi_markers):
            return "hi"
        if any(w in q_words for w in ne_markers):
            return "ne"

        # 2. Assamese vs Bengali letter check
        if any(c in text for c in ('ৰ', 'ৱ')):
            return "as"

        script_ranges = (
            (0x0900, 0x097F, "hi"),  # Default devanagari to hi
            (0x0980, 0x09FF, "bn"),  # Default bengali script to bn
            (0x0A00, 0x0A7F, "pa"),
            (0x0A80, 0x0AFF, "gu"),
            (0x0B00, 0x0B7F, "or"),
            (0x0B80, 0x0BFF, "ta"),
            (0x0C00, 0x0C7F, "te"),
            (0x0C80, 0x0CFF, "kn"),
            (0x0D00, 0x0D7F, "ml"),
            (0x0600, 0x06FF, "ur"),
        )
        counts = {name: 0 for _, _, name in script_ranges}
        for ch in text:
            cp = ord(ch)
            for start, end, name in script_ranges:
                if start <= cp <= end:
                    counts[name] += 1
                    break
        if not counts or sum(counts.values()) == 0:
            return "en"
        best = max(counts, key=counts.get)
        return best if counts[best] >= 2 else "en" """

if target1 in content:
    content = content.replace(target1, replacement1)
    print("Replaced target1!")
else:
    print("target1 not found")

if target2.strip() in content:
    content = content.replace(target2.strip(), replacement2.strip())
    print("Replaced target2!")
else:
    print("target2 not found")

with open('modal_app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Finished patching modal_app.py")
