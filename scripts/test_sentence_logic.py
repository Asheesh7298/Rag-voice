import re

chosen_text = "Asia > South Asia > India > Western India > Goa. Goa, a state on India's West coast, is a former Portuguese colony with a rich history. Spread over 3,700 square kilometres with a population of approximately 1.4 million, Goa is small by Indian standards."
q_keywords = ['goa']

# Strip breadcrumbs like "A > B > C > "
cleaned_text = re.sub(r'^(?:[^>]+>\s*)+', '', chosen_text).strip()
print('cleaned_text:', cleaned_text)
raw_sents = [s.strip() for s in re.split(r'(?<=[.!?\n।])\s+', cleaned_text) if len(s.strip()) > 8]
print('raw_sents:', raw_sents)
matched = [s for s in raw_sents if q_keywords and any(k in s.lower() for k in q_keywords)]
print('matched:', matched)
final_answer = ' '.join(matched[:2]) if matched else ' '.join(raw_sents[:2])
print('final_answer:', final_answer)
