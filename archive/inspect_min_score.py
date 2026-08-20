with open('modal_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Let's inspect where queries are processed in modal_app.py
print("Current MIN_QA_SCORE:", [line for line in content.split('\n') if 'MIN_QA_SCORE' in line])
