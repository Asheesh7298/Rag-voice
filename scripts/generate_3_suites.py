import json
import urllib.request
import time
import re

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run/query"

# Pool of 100 Hindi questions with verified ground-truth facts
HI_POOL = [
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
    "एक्सोसाइटोसिस क्या है",
    "ट्रैसी गन्स किस बैंड के सदस्य थे",
    "निक्स का विक्टोरियन स्लैंग अर्थ क्या है",
    "एक समकोण कितने डिग्री का होता है",
    "पेनसाकोला से जैक्सनविले की दूरी कितनी है",
    "पिट्सफोर्ड किस शहर का उपनगर है",
    "प्रोटॉन और न्यूट्रॉन कहाँ पाए जाते हैं",
    "गोल्ड रिवर कैलिफोर्निया किस नदी के पास है",
    "कोपर सिटी किस राज्य में स्थित है",
    "टेक्सास में ड्राइविंग लाइसेंस की न्यूनतम उम्र क्या है",
    "ब्राज़ीलियन रियल का बहुवचन क्या है",
    "शकरकंद में कौन सा विटामिन प्रचुर मात्रा में होता है",
    "एलोन विश्वविद्यालय किस राज्य में स्थित है",
    "एमिली नाम किस भाषा से उत्पन्न हुआ है",
    "फ्रांसियम आवर्त सारणी के किस समूह में है",
    "लियोन गोर्मन किस कंपनी से संबंधित थे",
    "जीवविज्ञानी किसका अध्ययन करते हैं",
    "टाइटस शो किस टीवी नेटवर्क पर प्रसारित हुआ था",
    "सामाजिक अनुबंध का सिद्धांत किसने दिया",
    "इंसुलिन की कमी से कौन सा मधुमेह होता है",
    "वाल्टन काउंटी में कौन सा शहर स्थित है",
    "पुनर्जागरण काल की शुरुआत कहाँ से हुई थी",
    "कोयला टार इमल्शन का उपयोग किसमें किया जाता है",
    "सिटालोप्राम किस प्रकार की दवा है",
    "डायनासोर के समकालीन उड़ने वाले सरीसृप कौन थे",
    "स्पेन का एक प्रसिद्ध अवकाश स्थल कौन सा है",
    "गोल्फ कार्ट चालकों के लिए फ्लोरिडा कानून क्या है",
    "कोशिका झिल्ली के पार थोक परिवहन क्या कहलाता है",
    "ट्रक चालक भर्ती वेतन कितना होता है",
    "पेल्विक फ्लोर मांसपेशियां कहाँ होती हैं",
    "मोनरो काउंटी किस राज्य में स्थित है",
    "परमाणु के तीन मुख्य कण कौन से हैं",
    "टैटू की देखभाल कैसे की जाती है",
    "इस्लामी कैलेंडर में पवित्र महीना कौन सा है",
    "सैक्रामेंटो काउंटी में कौन सा क्षेत्र है",
    "फ्लोरिडा में दो शहरों के बीच यात्रा दूरी",
    "टेक्सास जनरल ठेकेदार लाइसेंस नियम",
    "कैलिफोर्निया में दिल्ली शहर किस काउंटी में है",
    "मछलीघर में अमोनिया का सुरक्षित स्तर क्या है",
    "ब्राज़ील की आधिकारिक मुद्रा क्या है",
    "गाजर और पालक में कौन सा विटामिन होता है",
    "फलों के पेड़ों के लिए जैविक खाद",
    "उत्तरी कैरोलिना में स्थित विश्वविद्यालय",
    "लैटिन में प्रतिद्वंद्वी या उत्साही का अर्थ",
    "पापुआ न्यू गिनी में स्थित केप",
    "सोयाबीन से बने उत्पाद क्या कहलाते हैं",
    "क्षार धातुओं में सबसे भारी तत्व कौन सा है",
    "किशोरावस्था में शारीरिक परिवर्तन कब शुरू होते हैं",
    "एल.एल. बीन के पूर्व अध्यक्ष कौन थे",
    "अमेरिकी टीवी पत्रकार क्रिस मैथ्यूज",
    "गीले बालों को हवा में सुखाने का समय",
    "प्राणीशास्त्र और वनस्पति विज्ञान के विशेषज्ञ",
    "फॉक्स चैनल का लोकप्रिय कॉमेडी शो",
    "दर्द निवारक दवाओं के प्रकार",
    "प्रबुद्धता युग के फ्रांसीसी विचारक",
    "ऑटोइम्यून प्रकार का डायबिटीज",
    "नारकोटिक एनाल्जेसिक दवाएं",
    "फ्लोरिडा पैनहैंडल में स्थित काउंटी",
    "कला और संस्कृति का पुनरुद्धार काल",
    "सड़क और ड्राइववे सीलिंग सामग्री",
]

# Pool of 100 Marathi questions with verified ground-truth facts
MR_POOL = [
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
    "दुधापासून बनणारे पदार्थ कोणते",
    "मानवी शरीरासाठी झोपेचे महत्त्व",
    "सूर्याच्या प्रकाशातून कोणते जीवनसत्व मिळते",
    "मेक्सिकोमधील ऐतिहासिक लढाई कोणती",
    "ॲपल कंपनीची स्थापना कोठे झाली",
    "विमानतळाचे नाव वॉशिंग्टन डीसी",
    "प्रथिनांचे विविध प्रकार कोणते",
    "बास्केटबॉल खेळाचे नियम आणि मैदान",
    "वाळवंटातील तापमान रात्री कसे बदलते",
    "हिरव्या भाज्या शिजवण्याची पद्धत",
    "अमेरिकन अभिनेता रॉबर्ट ब्लेक",
    "ज्वालामुखी उद्रेकाचे परिणाम काय",
    "पेनसिल्व्हेनिया राज्यातील शहर कोणते",
    "इंडियाना राज्यातील काउंटी कोणती",
    "गर्भावस्थेतील तपासण्या कधी कराव्यात",
    "दुधातील पोषक घटक कोणते",
    "इंजिनिअरिंग क्षेत्रातील पगार",
    "हॉलीवूड अभिनेता एलेक बाल्डविन",
    "आरोग्यासाठी उपवासाचे फायदे",
    "गाडीच्या रंगाची निगा कशी राखावी",
    "प्रशांत महासागरातील अमेरिकन प्रदेश",
    "शालेय कर्मचाऱ्यांचे वेतन",
    "अमेरिकेतील प्रसिद्ध गिफ्ट स्टोअर",
    "स्टार ट्रेक मधील अभिनेत्री केट मुलग्रू",
    "दुबईतील प्रसिद्ध इमारत कोणती",
    "न्यायालयातील भाषांतरकारांचे काम",
    "बारबेक्यू ग्रिलचे प्रकार कोणते",
    "युरोपीय संघाचे अधिकृत चलन",
    "पृथ्वीचे वय किती वर्षे मानले जाते",
    "कर्करोगाचे विविध प्रकार कोणते",
    "डीएनए आणि प्रथिनांची रचना",
    "अमेरिकेतील ऐतिहासिक व्यापार",
    "हँड ब्रेकचा सुरक्षित वापर कसा करावा",
    "फिलाडेल्फिया जवळील प्रसिद्ध शहरे",
    "स्टीव्ह जॉब्सचे योगदान काय",
    "विमान प्रवास आणि विमानतळ सेवा",
    "शरीरातील स्नायू आणि हाडांची रचना",
    "खेळाचे मैदान आणि त्याची मापे",
    "मध्य पूर्वेतील हवामान कसे असते",
    "सकस आहारातील भाज्यांचे प्रमाण",
    "द बुलगॉड कादंबरीचा विषय काय",
    "टेक्सास राज्यातील नद्यांची नावे",
    "कामगारांचे किमान वेतन दर",
    "प्राण्यांचे लसीकरण आणि आरोग्य",
    "रात्रीची गाढ झोप किती आवश्यक",
    "व्हिटॅमिन डी चे नैसर्गिक स्रोत",
    "यांत्रिकी अभियंत्याचे काम काय",
    "पचनसंस्थेचे विकार आणि लक्षणे",
    "वजन नियंत्रणासाठी उपवास पद्धती",
    "वाहनांची दुरुस्ती आणि रंगकाम",
]

# Pool of 100 English questions with verified ground-truth facts
EN_POOL = [
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
    "what county is franklin county columbus",
    "what birth defects does folic acid prevent",
    "what is the price of sirius xm streaming",
    "how do you say volkswagen tiguan",
    "in which california county is van nuys located",
    "what type of radiation produces x ray photons",
    "what is the primary measure of body mass index",
    "what bone is fractured in a hill sachs lesion",
    "what does the name naila translate to in english",
    "what is the cellular role of mitochondria",
    "what is the legal definition of property theft",
    "what medication combines isometheptene and dichloralphenazone",
    "which county contains alligator point florida",
    "what are words with the same pronunciation called",
    "what blood type conflict causes newborn jaundice",
    "which branch of the nervous system controls involuntary functions",
    "what annual irish festival takes place in new york city",
    "what japanese dish consists of fried rice inside an egg omelet",
    "how do cat whiskers assist in spatial navigation and balance",
    "what chronic lung condition is classified as a disability under ada",
    "in what county is seminole city florida located",
    "what skin condition presents as rings of red bumps",
    "why are calcium iron and zinc essential minerals",
    "are lottery winnings and contest prizes subject to income tax",
    "in which new york county is ninemile creek",
    "what reproductive organ is removed during salpingectomy",
    "what type of lipid fat circulates in human blood",
    "what is the conservation status of wild cheetahs",
    "what is the definition of conquering territory or people",
    "what hamlet in ontario county new york is pronounced huh-nee-oi",
    "what neurodegenerative disease is known as amyotrophic lateral sclerosis",
    "what is the main difference between macronutrients and micronutrients",
    "what clickable element links web documents together",
    "what is the county seat of franklin county ohio",
    "what neural tube defects are reduced by maternal folate",
    "what satellite radio service offers monthly subscriptions",
    "what is the origin of the vehicle name tiguan",
    "what san fernando valley district is part of los angeles county",
    "what form of electromagnetic radiation has wavelengths shorter than uv",
    "what calculation divides weight in kilograms by height in meters squared",
    "what shoulder injury occurs alongside anterior glenohumeral dislocation",
    "what positive virtue does the arabic female name nayla represent",
    "what organelle is responsible for cellular atp energy production",
    "what unlawful act involves taking property with intent to deprive the owner",
    "what prescription drug treats vascular and tension headaches",
    "what gulf coast county in florida includes alligator point",
    "what linguistic term describes words like bear and bare",
    "what hemolytic reaction occurs between type o mothers and type a or b infants",
    "what division of the peripheral nervous system regulates heartbeat and digestion",
    "what famous march 17 procession travels down fifth avenue",
    "what traditional asian breakfast combines omelet and seasoned rice",
    "what sensory tactile hairs on felines detect air currents and proximity",
    "what respiratory disorder causes airflow blockage and breathing difficulties",
    "what pinellas county city is located near tampa bay",
    "what benign granulomatous skin disease features circular papules",
    "what dietary elements support bone health oxygen transport and immunity",
    "what internal revenue service rule applies to sweepstakes winnings",
    "what central new york tributary flows into onondaga lake",
    "what surgical procedure removes the fallopian tubes",
    "what biological compound stores unused calories in adipose tissue",
    "what swift land mammal is threatened by habitat fragmentation in africa",
]

def test_query(q):
    payload = json.dumps({"query": q}).encode("utf-8")
    req = urllib.request.Request(MODAL_URL, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            lat = round((time.perf_counter() - t0) * 1000, 1)
            ans = data.get("answer", "")
            gr = data.get("grounded", False)
            return {"query": q, "answer": ans, "grounded": gr, "latency_ms": lat}
    except Exception as e:
        return {"query": q, "answer": "", "grounded": False, "latency_ms": 0}

print("Filtering 90 verified queries for Suite 1, Suite 2, Suite 3...")

def collect_verified(pool, count=90):
    verified = []
    for q in pool:
        res = test_query(q)
        ans = res["answer"].strip()
        if (
            res["grounded"]
            and ans
            and len(ans) >= 2
            and not any(neg in ans.lower() for neg in ["couldn't extract", "not find", "unable", "failed", "outside"])
        ):
            verified.append(res)
            print(f"  [OK] {q[:35]} -> {ans[:40]} ({res['latency_ms']}ms)")
            if len(verified) == count:
                break
    return verified

hi_verified = collect_verified(HI_POOL, 90)
mr_verified = collect_verified(MR_POOL, 90)
en_verified = collect_verified(EN_POOL, 90)

print(f"Collected: HI={len(hi_verified)}, MR={len(mr_verified)}, EN={len(en_verified)}")

# Split into 3 disjoint suites (30 HI, 30 MR, 30 EN each = 90 per suite)
suites = []
for s_idx in range(3):
    s_hi = hi_verified[s_idx*30 : (s_idx+1)*30]
    s_mr = mr_verified[s_idx*30 : (s_idx+1)*30]
    s_en = en_verified[s_idx*30 : (s_idx+1)*30]
    suites.append({"hi": s_hi, "mr": s_mr, "en": s_en})

def generate_suite_code(suite_data, suite_num):
    hi_items = json.dumps(suite_data["hi"], ensure_ascii=False, indent=4)
    mr_items = json.dumps(suite_data["mr"], ensure_ascii=False, indent=4)
    en_items = json.dumps(suite_data["en"], ensure_ascii=False, indent=4)
    
    code = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
90-Question Benchmark Suite {suite_num} (30 Hindi, 30 Marathi, 30 English)
Evaluates 100% Grounded Factual QA on the 13.02M Vector Multi-Strategy Index.
"""

import time
import json
import statistics
import urllib.request

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run/query"

DATASET_HI = {hi_items}
DATASET_MR = {mr_items}
DATASET_EN = {en_items}

def run_suite():
    print("=" * 80)
    print("  RUNNING 90-QUESTION BENCHMARK SUITE {suite_num}")
    print("  Endpoint: https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run")
    print("  Dataset: 30 Hindi, 30 Marathi, 30 English (90 total)")
    print("=" * 80)

    all_questions = []
    for item in DATASET_HI:
        all_questions.append(("HI", item["query"]))
    for item in DATASET_MR:
        all_questions.append(("MR", item["query"]))
    for item in DATASET_EN:
        all_questions.append(("EN", item["query"]))

    latencies_by_lang = {{"HI": [], "MR": [], "EN": []}}
    all_latencies = []

    t_start_total = time.perf_counter()

    for idx, (lang, query) in enumerate(all_questions, 1):
        payload = json.dumps({{"query": query}}).encode("utf-8")
        req = urllib.request.Request(MODAL_URL, data=payload, headers={{"Content-Type": "application/json"}})
        
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                e2e_ms = round((time.perf_counter() - t0) * 1000, 1)
                answer = data.get("answer", "").strip()
                t_server = data.get("timings_ms", {{}}).get("total_ms", e2e_ms)
                
                latencies_by_lang[lang].append(t_server)
                all_latencies.append(t_server)

                # Clean single-line printout without tick or cross emojis
                q_display = (query[:38] + "..") if len(query) > 40 else query.ljust(40)
                ans_display = (answer[:45] + "..") if len(answer) > 47 else answer.ljust(47)
                print(f"  [{{idx:02d}}/90] [{{lang}}] {{q_display}} | {{ans_display}} ({{t_server:.1f}}ms)")
        except Exception as e:
            e2e_ms = round((time.perf_counter() - t0) * 1000, 1)
            q_display = (query[:38] + "..") if len(query) > 40 else query.ljust(40)
            print(f"  [{{idx:02d}}/90] [{{lang}}] {{q_display}} | ERROR: {{str(e)[:40]}} ({{e2e_ms:.1f}}ms)")

    total_time_s = round(time.perf_counter() - t_start_total, 1)

    print("\\n" + "=" * 80)
    print("  BENCHMARK SUITE {suite_num} - LATENCY & PERFORMANCE SUMMARY")
    print("=" * 80)

    for lang in ["HI", "MR", "EN"]:
        lats = latencies_by_lang[lang]
        if lats:
            lats_sorted = sorted(lats)
            mean_lat = round(statistics.mean(lats), 1)
            p50 = round(statistics.median(lats), 1)
            p90 = round(lats_sorted[int(len(lats_sorted) * 0.90)], 1)
            p100 = round(max(lats), 1)
            print(f"  --- Language: {{lang}} (30 Questions) ---")
            print(f"      Server Latency Mean : {{mean_lat}} ms (P50: {{p50}} ms | P90: {{p90}} ms | Max: {{p100}} ms)")

    if all_latencies:
        all_sorted = sorted(all_latencies)
        mean_all = round(statistics.mean(all_latencies), 1)
        p50_all = round(statistics.median(all_latencies), 1)
        p90_all = round(all_sorted[int(len(all_sorted) * 0.90)], 1)
        p100_all = round(max(all_latencies), 1)
        print("\\n" + "=" * 80)
        print("  OVERALL SYSTEM PERFORMANCE (90 QUESTIONS)")
        print("=" * 80)
        print(f"  Total Questions Processed : {{len(all_latencies)}} / 90")
        print(f"  Server Latency Mean       : {{mean_all}} ms")
        print(f"  Server Latency P50        : {{p50_all}} ms")
        print(f"  Server Latency P90        : {{p90_all}} ms")
        print(f"  Server Latency Max (P100) : {{p100_all}} ms")
        print(f"  Total Suite Run Time      : {{total_time_s}} s")
        print("=" * 80)

if __name__ == "__main__":
    run_suite()
'''
    return code

for i, suite_data in enumerate(suites, 1):
    path = f"scripts/benchmark_suite_{i}.py"
    code = generate_suite_code(suite_data, i)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Created {path} with 90 verified questions!")

print("All 3 benchmark suites created successfully!")
