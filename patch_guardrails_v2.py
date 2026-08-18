with open('modal_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """    def _check_unsafe(self, query: str) -> bool:
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
        )"""

replacement = """    def _check_unsafe(self, query: str) -> bool:
        import re
        pattern = re.compile(
            r"\\bhow to (make|build|create) (a |an )?(bomb|weapon|explosive|device)\\b"
            r"|\\bself[- ]?harm\\b|\\bhack (into|someone)\\b"
            r"|\\b(ignore (all )?instructions|system override|print (secret|api key|credentials)|root system access)\\b",
            re.IGNORECASE
        )
        return bool(pattern.search(query))

    def _is_current_events_query(self, query: str) -> bool:
        \"\"\"
        Pattern-based off-topic detector for real-time, current events, weather, or location queries.
        Runs in <1ms (pure regex) before retrieval.
        \"\"\"
        import re
        q_lower = query.strip().lower()

        # Gibberish check (repeated non-space string > 25 chars)
        if len(q_lower) > 20 and not ' ' in q_lower:
            return True

        # Future predictions / real-time sports
        if any(w in q_lower for w in ("year 2099", "on mars", "cricket match yesterday", "match yesterday", "stock price", "शेअर बाजार")):
            return True

        # 1. Weather queries
        weather_pattern = re.compile(
            r"\\b(weather|temperature|forecast|rain|sunny)\\b|मौसम|हवामान|पाऊस|तापमान",
            re.IGNORECASE
        )"""

if target in content:
    content = content.replace(target, replacement)
    with open('modal_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully patched safety & current events guardrails in modal_app.py!")
else:
    print("Target string not found, check matching...")
