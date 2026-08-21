import urllib.request
import json
import time

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run/query"

# Curated high-precision verified questions
CANDIDATE_HI = [
    "ब्राइटन टाउनशिप फोन नंबर",
    "सबसे बड़ा उड़ने वाला सरीसृप अब तक",
    "सैंटेंडर स्पेन के बारे में क्या प्रसिद्ध है",
    "फ्लोरिडा में गोल्फ कार्ट चलाने की उम्र क्या है",
    "कौन सी थोक परिवहन प्रक्रिया सामग्री को कोशिकाओं में ले जाती है",
    "एक भर्ती किए गए ट्रक चालक कितना कमाते हैं",
    "श्रोणि तल की मांसपेशियाँ कहाँ स्थित हैं",
    "कौन सी काउंटी पिट्सफोर्ड, न्यूयॉर्क में है",
    "कौन से उपपरमाण्विक कण होते हैं",
    "टैटू लगवाने के बाद पानी में जाने से पहले कितना इंतजार करना चाहिए",
    "रमजान मुबारक क्या है",
    "कौन सी काउंटी में गोल्ड रिवर सीए स्थित है",
    "कोपर शहर से हॉलीवुड कितना दूर है",
    "आप किस उम्र के हो सकते हैं जीसी टेक्सास",
    "दिल्ली कैलिफोर्निया कौन सी काउंटी में है",
    "मीठे पानी में एन.एच.3 का स्तर क्या है",
    "ब्राज़ील में मुद्रा को क्या कहते हैं",
    "कौन से खाद्य पदार्थ विटामिन ए के अच्छे स्रोत हैं",
    "आड़ू के पेड़ों के लिए किस तरह का उर्वरक चाहिए",
    "किस शहर में एलोन विश्वविद्यालय है",
    "एमिली नाम का क्या अर्थ है",
    "केप टोरोकिना कहाँ स्थित है",
    "सोया का क्या अर्थ है",
    "आवर्त सारणी में कौन सा तत्व सबसे अधिक प्रतिक्रियाशील है",
    "लड़के कब यौवन में पहुँचते हैं",
    "मेन में सबसे अमीर व्यक्ति कौन है",
    "क्रिस मैथ्यू कितने लंबे हैं",
    "बाल पूरी तरह से सूखने में कितना समय लगता है",
    "एक वैज्ञानिक जो जीव विज्ञान का अध्ययन करता है उसे क्या कहते हैं",
    "किस शो में क्रिस टाइटस थे",
    "अंडकोष के दर्द के लिए कौन सा उपचार सबसे अच्छा है",
    "जॉन जैक्स रूसो किस सिद्धांत के दार्शनिक थे",
    "मधुमेह मेलिटस टाइप 1 क्या है",
    "ऑक्सीकोडोन और लोर्टैब में कौन अधिक मजबूत है",
    "डेफ़ुनियाक स्प्रिंग्स किस काउंटी में है",
    "उत्तरी यूरोप में पुनर्जागरण का क्या अर्थ है",
    "डामर सील करने की प्रक्रिया क्या है",
    "सिटालोप्राम को काम करने में कितना समय लगता है",
    "अज़हद्रिचिडे किस प्रकार के जीव थे",
    "कैंटाब्रिया की राजधानी क्या है",
]

CANDIDATE_MR = [
    "फ्रान्सचे सध्याचे चलन काय आहे",
    "पृथ्वी किती जुनी आहे",
    "यकृताच्या कर्करोगाला काय म्हणतात",
    "जगातील सर्वात उंच इमारत किती फूट आहे",
    "अल्फा हेलिक्स कुठे आढळते",
    "हेंडरसनव्हिल एनसी कोणत्या काउंटीमध्ये आहे",
    "पार्किंग ब्रेक म्हणजे काय",
    "आफ्रिकन गुलामांचा व्यापार किती काळ चालला",
    "कॅम्प हिल पा फिली पासून किती लांब आहे",
    "सिंको डी मायो कुठे झाले",
    "स्टीव्ह जॉब्सने ॲपल सार्वजनिक कधी केले",
    "गॅस सिटी इंडियाना कोणत्या काउंटीमध्ये आहे",
    "किती आठवडे गर्भवती असताना लिंग कळते",
    "वॉशिंग्टन डी.सी. मधील सर्वात जवळचा विमानतळ कोणता",
    "रॉबर्ट ब्लेक कोण आहेत",
    "ज्वालामुखीच्या राखेमुळे कोणते नुकसान होते",
    "तंतुमय आणि गोलाकार प्रथिने यातील फरक",
    "एन.सी.ए.ए. बास्केटबॉल कोर्टचा आकार",
    "सौदी अरेबियात संध्याकाळी तापमान कसे असते",
    "स्नॅप वाटाणे कसे शिजवायचे",
    "दुग्धजन्य उत्पादने कोणती आहेत",
    "द बुलगॉड पुस्तक कोणी लिहिले",
    "व्रे नाव कोणत्या नदीवरून आले",
    "सरासरी ताशी वेतन किती आहे",
    "मांजरीवर रेबीज लस किती काळ काम करते",
    "प्रौढांना किती तास झोप आवश्यक आहे",
    "कोणत्या खाद्यपदार्थात जीवनसत्व डी असते",
    "मेकॅनिकल इंजिनिअर सरासरी किती कमावतात",
    "एलेक बाल्डविनची पत्नी कोण आहे",
    "भूक न लागण्याचे कारण काय असू शकते",
    "आठवड्यातून एक दिवस उपवास करणे",
    "कार पेंटच्या कामाचा खर्च किती",
    "ग्वाम बेट कुठे आहे",
    "शाळेचे सचिव दरवर्षी किती कमावतात",
    "हॉलमार्क गोल्ड क्राउन स्टोअर्स कुठे आहेत",
    "केट मुलग्रो कोण आहे",
    "बुर्ज खलिफाची उंची किती आहे",
    "कॅलिफोर्निया न्यायालयीन दुभाषी कोणत्या काउंटीत",
    "चारकोल ग्रिलचा प्रकार कोणता",
    "युरो हे कोणत्या देशाचे चलन आहे",
]

CANDIDATE_EN = [
    "what county is columbus city in",
    "what does folic acid protect against",
    "siriusxm cost per month internet only",
    "how to pronounce tiguan",
    "what county is van nuys ca in",
    "what is an x ray photon",
    "what are the common geriatric digestive issues",
    "what is a slip coin",
    "what is the medical definition of obesity metric",
    "is julia kim still alive",
    "what is a hill-sachs lesion",
    "meaning of the arabic name nayla",
    "average cost of wedding dinner",
    "what is a values clarification process",
    "is the mitochondria an organelle",
    "webster definition of theft",
    "what active ingredients are in midrin",
    "what county in florida is alligator point in",
    "what does homophone mean in english",
    "recommended daily intake of vitamin e",
    "what is abo incompatibility in newborns",
    "what nerves regulate the autonomic system",
    "what is the name of st patricks parade in new york",
    "what is an omurice omelet made with",
    "are whiskers on cats used for balance",
    "average cost of assisted living in 2015",
    "what disability classification is copd",
    "what county is seminole fl in",
    "what does granuloma annulare look like",
    "what minerals are important in diet",
    "is prize money taxable income",
    "what county is ninemile creek located in",
    "what is a salpingectomy surgery",
    "what are triglycerides in blood",
    "are cheetahs an endangered species",
    "meaning of conquest",
    "what county is honeoye in new york",
    "what does als stand for in medicine",
    "what distinguishes a macronutrient from micronutrient",
    "what is a hyperlink on a webpage",
]

print("Testing candidate queries against live Modal A100...")

def test_query(q):
    payload = json.dumps({"query": q}).encode("utf-8")
    req = urllib.request.Request(MODAL_URL, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            lat = round((time.perf_counter() - t0) * 1000, 1)
            ans = data.get("answer", "")
            gr = data.get("grounded", False)
            return {"query": q, "answer": ans, "grounded": gr, "latency_ms": lat}
    except Exception as e:
        return {"query": q, "answer": str(e), "grounded": False, "latency_ms": 0}

hi_results = [test_query(q) for q in CANDIDATE_HI[:35]]
mr_results = [test_query(q) for q in CANDIDATE_MR[:35]]
en_results = [test_query(q) for q in CANDIDATE_EN[:35]]

print(f"HI Valid: {sum(1 for r in hi_results if r['grounded'])} / {len(hi_results)}")
print(f"MR Valid: {sum(1 for r in mr_results if r['grounded'])} / {len(mr_results)}")
print(f"EN Valid: {sum(1 for r in en_results if r['grounded'])} / {len(en_results)}")

with open("data/verified_results.json", "w", encoding="utf-8") as f:
    json.dump({"hi": hi_results, "mr": mr_results, "en": en_results}, f, ensure_ascii=False, indent=2)
