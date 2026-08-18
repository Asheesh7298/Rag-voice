with open('modal_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target_str = """    # ── STT ───────────────────────────────────────────────────────────────────

    def _transcribe(self, audio_bytes: bytes, lang=None):
        import time, httpx
        LANG_MAP = {
            "as": "as-IN", "bn": "bn-IN", "gu": "gu-IN", "hi": "hi-IN",
            "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN", "ne": "ne-IN",
            "or": "od-IN", "pa": "pa-IN", "ta": "ta-IN", "te": "te-IN",
            "ur": "ur-IN", "en": "en-IN",
        }
        news_pattern = re.compile(
            r"\\b(today|right now|currently|latest news|breaking)\\b",
            re.IGNORECASE
        )
        if news_pattern.search(q_lower):
            return True

        # 3. Real-time data
        realtime_pattern = re.compile(
            r"\\b(stock price|exchange rate|live score|current price)\\b",
            re.IGNORECASE
        )
        if realtime_pattern.search(q_lower):
            return True

        # 4. Personal/location queries
        location_pattern = re.compile(
            r"^(where am i|my location|near me)\\b|\\bnear me\\b",
            re.IGNORECASE
        )
        if location_pattern.search(q_lower):
            return True

        return False"""

replacement_str = """    # ── STT ───────────────────────────────────────────────────────────────────

    def _transcribe(self, audio_bytes: bytes, lang=None):
        import time, httpx
        LANG_MAP = {
            "as": "as-IN", "bn": "bn-IN", "gu": "gu-IN", "hi": "hi-IN",
            "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN", "ne": "ne-IN",
            "or": "od-IN", "pa": "pa-IN", "ta": "ta-IN", "te": "te-IN",
            "ur": "ur-IN", "en": "en-IN",
        }
        bcp47 = LANG_MAP.get(lang) if lang else None
        headers = {"api-subscription-key": self.SARVAM_KEY}
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"model": "saaras:v3", "mode": "transcribe"}
        if bcp47: data["language_code"] = bcp47
        t0 = time.perf_counter()
        r = httpx.post(self.SARVAM_URL, headers=headers, files=files, data=data, timeout=10)
        r.raise_for_status()
        stt_ms = round((time.perf_counter() - t0) * 1000, 2)
        result = r.json()
        return {
            "transcript": result.get("transcript", ""),
            "language_detected": result.get("language_code"),
            "latency_ms": stt_ms,
        }

    # ── Guardrails ────────────────────────────────────────────────────────────

    def _check_unsafe(self, query: str) -> bool:
        import re
        pattern = re.compile(
            r"\\bhow to (make|build) (a )?(bomb|weapon|explosive)\\b"
            r"|\\bself[- ]?harm\\b|\\bhack (into|someone)\\b", re.IGNORECASE
        )
        return bool(pattern.search(query))

    def _is_current_events_query(self, query: str) -> bool:
        \"\"\"
        Pattern-based off-topic detector for real-time, current events, weather, or location queries.
        Runs in <1ms (pure regex) before retrieval.
        \"\"\"
        import re
        q_lower = query.strip().lower()

        # 1. Weather queries
        weather_pattern = re.compile(
            r"\\b(weather|temperature|forecast|rain|sunny)\\b|मौसम|हवामान",
            re.IGNORECASE
        )
        if weather_pattern.search(q_lower):
            return True

        # 2. Current news
        news_pattern = re.compile(
            r"\\b(today|right now|currently|latest news|breaking)\\b",
            re.IGNORECASE
        )
        if news_pattern.search(q_lower):
            return True

        # 3. Real-time data
        realtime_pattern = re.compile(
            r"\\b(stock price|exchange rate|live score|current price)\\b",
            re.IGNORECASE
        )
        if realtime_pattern.search(q_lower):
            return True

        # 4. Personal/location queries
        location_pattern = re.compile(
            r"^(where am i|my location|near me)\\b|\\bnear me\\b",
            re.IGNORECASE
        )
        if location_pattern.search(q_lower):
            return True

        # 5. Current political office holders / heads of state
        political_pattern = re.compile(
            r"\\b(prime minister|president of|current pm|pm of india|who is the pm|who is the president)\\b|प्रधानमंत्री|राष्ट्रपती|पंतप्रधान",
            re.IGNORECASE
        )
        if political_pattern.search(q_lower):
            return True

        return False"""

if target_str in content:
    content = content.replace(target_str, replacement_str)
    with open('modal_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated modal_app.py!")
else:
    print("Target string not found, check matching.")
