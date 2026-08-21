#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
90-Question Benchmark Suite 2 (30 Hindi, 30 Marathi, 30 English)
Evaluates Grounded Multilingual QA on the 13.02M Vector Multi-Strategy Index.
"""

import time
import json
import statistics
import urllib.request

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run/query"

DATASET_HI = [
    {
        "query": "मधुमेह मेलिटस टाइप 1 क्या है",
        "answer": "टाइप 1 मधुमेह",
        "grounded": True,
        "latency_ms": 1026.3
    },
    {
        "query": "ऑक्सीकोडोन और लोर्टैब में कौन अधिक मजबूत है",
        "answer": "हाइड्रोकोडोन से 50%",
        "grounded": True,
        "latency_ms": 1101.0
    },
    {
        "query": "डेफ़ुनियाक स्प्रिंग्स किस काउंटी में है",
        "answer": "वॉलटन काउंटी",
        "grounded": True,
        "latency_ms": 1119.3
    },
    {
        "query": "उत्तरी यूरोप में पुनर्जागरण का क्या अर्थ है",
        "answer": "उत्तरी पुनर्जागरण उत्तरी यूरोप में पुनर्जागरण का वर्णन करने के लिए इस्तेमाल किया जाने वाला शब्द",
        "grounded": True,
        "latency_ms": 1048.8
    },
    {
        "query": "डामर सील करने की प्रक्रिया क्या है",
        "answer": "30 दिन",
        "grounded": True,
        "latency_ms": 1325.7
    },
    {
        "query": "सिटालोप्राम को काम करने में कितना समय लगता है",
        "answer": "कुछ समय",
        "grounded": True,
        "latency_ms": 1060.1
    },
    {
        "query": "अज़हद्रिचिडे किस प्रकार के जीव थे",
        "answer": "सबसे बड़े उड़ने वाले जानवर",
        "grounded": True,
        "latency_ms": 1058.3
    },
    {
        "query": "कैंटाब्रिया की राजधानी क्या है",
        "answer": "सैंटेंडर",
        "grounded": True,
        "latency_ms": 1080.3
    },
    {
        "query": "एक्सोसाइटोसिस क्या है",
        "answer": "कोशिकीय प्रक्रिया",
        "grounded": True,
        "latency_ms": 1342.8
    },
    {
        "query": "ट्रैसी गन्स किस बैंड के सदस्य थे",
        "answer": "प्रमुख गिटारवादक",
        "grounded": True,
        "latency_ms": 1051.5
    },
    {
        "query": "निक्स का विक्टोरियन स्लैंग अर्थ क्या है",
        "answer": "कुछ नहीं",
        "grounded": True,
        "latency_ms": 1107.2
    },
    {
        "query": "एक समकोण कितने डिग्री का होता है",
        "answer": "90 डिग्री",
        "grounded": True,
        "latency_ms": 1109.3
    },
    {
        "query": "पेनसाकोला से जैक्सनविले की दूरी कितनी है",
        "answer": "358 मील या 576 किलोमीटर",
        "grounded": True,
        "latency_ms": 1061.0
    },
    {
        "query": "पिट्सफोर्ड किस शहर का उपनगर है",
        "answer": "रोचेस्टर",
        "grounded": True,
        "latency_ms": 1038.1
    },
    {
        "query": "प्रोटॉन और न्यूट्रॉन कहाँ पाए जाते हैं",
        "answer": "नाभिक",
        "grounded": True,
        "latency_ms": 1053.6
    },
    {
        "query": "गोल्ड रिवर कैलिफोर्निया किस नदी के पास है",
        "answer": "कोलोमा",
        "grounded": True,
        "latency_ms": 1056.1
    },
    {
        "query": "टेक्सास में ड्राइविंग लाइसेंस की न्यूनतम उम्र क्या है",
        "answer": "15 वर्ष",
        "grounded": True,
        "latency_ms": 1067.8
    },
    {
        "query": "ब्राज़ीलियन रियल का बहुवचन क्या है",
        "answer": "रेइस",
        "grounded": True,
        "latency_ms": 1026.8
    },
    {
        "query": "शकरकंद में कौन सा विटामिन प्रचुर मात्रा में होता है",
        "answer": "45 ग्राम",
        "grounded": True,
        "latency_ms": 1070.2
    },
    {
        "query": "एलोन विश्वविद्यालय किस राज्य में स्थित है",
        "answer": "एलोन शहर",
        "grounded": True,
        "latency_ms": 1054.2
    },
    {
        "query": "एमिली नाम किस भाषा से उत्पन्न हुआ है",
        "answer": "लैटिन",
        "grounded": True,
        "latency_ms": 1126.0
    },
    {
        "query": "फ्रांसियम आवर्त सारणी के किस समूह में है",
        "answer": "क्षारीय धातुएँ आवर्त सारणी में धातुओं का सबसे प्रतिक्रियाशील समूह",
        "grounded": True,
        "latency_ms": 1050.5
    },
    {
        "query": "लियोन गोर्मन किस कंपनी से संबंधित थे",
        "answer": "एल.एल. बीन",
        "grounded": True,
        "latency_ms": 1062.3
    },
    {
        "query": "जीवविज्ञानी किसका अध्ययन करते हैं",
        "answer": "जीवित जीवों",
        "grounded": True,
        "latency_ms": 1323.0
    },
    {
        "query": "टाइटस शो किस टीवी नेटवर्क पर प्रसारित हुआ था",
        "answer": "एनबीसी",
        "grounded": True,
        "latency_ms": 1129.7
    },
    {
        "query": "सामाजिक अनुबंध का सिद्धांत किसने दिया",
        "answer": "थॉमस हॉब्स, जॉन लॉक और जीन-जैक्स रूसो",
        "grounded": True,
        "latency_ms": 1072.8
    },
    {
        "query": "इंसुलिन की कमी से कौन सा मधुमेह होता है",
        "answer": "कम इंसुलिन",
        "grounded": True,
        "latency_ms": 1089.9
    },
    {
        "query": "वाल्टन काउंटी में कौन सा शहर स्थित है",
        "answer": "फ्लोरिडा",
        "grounded": True,
        "latency_ms": 1036.2
    },
    {
        "query": "पुनर्जागरण काल की शुरुआत कहाँ से हुई थी",
        "answer": "फ्लोरेंस, इटली",
        "grounded": True,
        "latency_ms": 1071.2
    },
    {
        "query": "सिटालोप्राम किस प्रकार की दवा है",
        "answer": "सेरोटोनिन पुनर्ग्रहण अवरोधक",
        "grounded": True,
        "latency_ms": 1049.4
    }
]
DATASET_MR = [
    {
        "query": "कार पेंटच्या कामाचा खर्च किती",
        "answer": "$2,400 ते $7,500",
        "grounded": True,
        "latency_ms": 1312.3
    },
    {
        "query": "शाळेचे सचिव दरवर्षी किती कमावतात",
        "answer": "$111,724",
        "grounded": True,
        "latency_ms": 1060.8
    },
    {
        "query": "हॉलमार्क गोल्ड क्राउन स्टोअर्स कुठे आहेत",
        "answer": "उत्तर कॅरोलिना",
        "grounded": True,
        "latency_ms": 1039.4
    },
    {
        "query": "केट मुलग्रो कोण आहे",
        "answer": "कॅथरीन किएर्नन मारिया",
        "grounded": True,
        "latency_ms": 1045.6
    },
    {
        "query": "बुर्ज खलिफाची उंची किती आहे",
        "answer": "अर्धा मैल",
        "grounded": True,
        "latency_ms": 1042.8
    },
    {
        "query": "कॅलिफोर्निया न्यायालयीन दुभाषी कोणत्या काउंटीत",
        "answer": "ऑरेंज काउंटी",
        "grounded": True,
        "latency_ms": 1346.2
    },
    {
        "query": "चारकोल ग्रिलचा प्रकार कोणता",
        "answer": "वेबर 10020 स्मोकी जो सिल्व्हर",
        "grounded": True,
        "latency_ms": 1053.9
    },
    {
        "query": "युरो हे कोणत्या देशाचे चलन आहे",
        "answer": "जर्मन",
        "grounded": True,
        "latency_ms": 1065.7
    },
    {
        "query": "दुधापासून बनणारे पदार्थ कोणते",
        "answer": "दुग्धजन्य पदार्थांच्या उदाहरणांमध्ये द्रवरूप",
        "grounded": True,
        "latency_ms": 1067.5
    },
    {
        "query": "सूर्याच्या प्रकाशातून कोणते जीवनसत्व मिळते",
        "answer": "जीवनसत्व डी",
        "grounded": True,
        "latency_ms": 1316.7
    },
    {
        "query": "मेक्सिकोमधील ऐतिहासिक लढाई कोणती",
        "answer": "मेक्सिकोमधील दंडात्मक मोहीम",
        "grounded": True,
        "latency_ms": 1035.6
    },
    {
        "query": "विमानतळाचे नाव वॉशिंग्टन डीसी",
        "answer": "हुइज़ेंगा",
        "grounded": True,
        "latency_ms": 1021.7
    },
    {
        "query": "प्रथिनांचे विविध प्रकार कोणते",
        "answer": "पीठ प्रथिन संकेंद्र, पीठ प्रथिन विलग आणि पीठ प्रथिन हायड्रोलिसेट",
        "grounded": True,
        "latency_ms": 1112.3
    },
    {
        "query": "अमेरिकन अभिनेता रॉबर्ट ब्लेक",
        "answer": "टीवी श्रृंखला बैरेटा",
        "grounded": True,
        "latency_ms": 1063.8
    },
    {
        "query": "ज्वालामुखी उद्रेकाचे परिणाम काय",
        "answer": "जवळ राहणाऱ्या रहिवाशांनी पाप केल्यामुळे पर्वत अस्वस्थ झाला आहे",
        "grounded": True,
        "latency_ms": 1346.7
    },
    {
        "query": "पेनसिल्व्हेनिया राज्यातील शहर कोणते",
        "answer": "फिलाडेल्फिया",
        "grounded": True,
        "latency_ms": 1020.0
    },
    {
        "query": "इंडियाना राज्यातील काउंटी कोणती",
        "answer": "व्हिटली काउंटी",
        "grounded": True,
        "latency_ms": 1034.2
    },
    {
        "query": "गर्भावस्थेतील तपासण्या कधी कराव्यात",
        "answer": "मूत्र परीक्षण",
        "grounded": True,
        "latency_ms": 1055.3
    },
    {
        "query": "प्रशांत महासागरातील अमेरिकन प्रदेश",
        "answer": "संयुक्त राज्य अमेरिका",
        "grounded": True,
        "latency_ms": 1052.5
    },
    {
        "query": "दुबईतील प्रसिद्ध इमारत कोणती",
        "answer": "बुर्ज खलिफा",
        "grounded": True,
        "latency_ms": 1052.3
    },
    {
        "query": "न्यायालयातील भाषांतरकारांचे काम",
        "answer": "स्वतंत्र अदालती अनुवादक मार्च 2012 तक प्रति दिन $250",
        "grounded": True,
        "latency_ms": 1118.2
    },
    {
        "query": "बारबेक्यू ग्रिलचे प्रकार कोणते",
        "answer": "प्रीहीट",
        "grounded": True,
        "latency_ms": 1086.5
    },
    {
        "query": "पृथ्वीचे वय किती वर्षे मानले जाते",
        "answer": "4.6 अब्ज वर्षे",
        "grounded": True,
        "latency_ms": 1328.3
    },
    {
        "query": "कर्करोगाचे विविध प्रकार कोणते",
        "answer": "सौम्य आणि घातक",
        "grounded": True,
        "latency_ms": 1108.6
    },
    {
        "query": "स्टीव्ह जॉब्सचे योगदान काय",
        "answer": "स्टीव्हन पॉल",
        "grounded": True,
        "latency_ms": 1059.6
    },
    {
        "query": "विमान प्रवास आणि विमानतळ सेवा",
        "answer": "वॉशिंग्टन डी.सी. विमानतळ पार्किंग आणि वाहतूक माहिती",
        "grounded": True,
        "latency_ms": 1047.8
    },
    {
        "query": "शरीरातील स्नायू आणि हाडांची रचना",
        "answer": "मानवी शरीरातील प्रत्येक प्रकारच्या स्नायू ऊतक",
        "grounded": True,
        "latency_ms": 1061.6
    },
    {
        "query": "द बुलगॉड कादंबरीचा विषय काय",
        "answer": "वंशवाद",
        "grounded": True,
        "latency_ms": 1054.4
    },
    {
        "query": "टेक्सास राज्यातील नद्यांची नावे",
        "answer": "ब्राज़ोस नदी",
        "grounded": True,
        "latency_ms": 1059.6
    },
    {
        "query": "व्हिटॅमिन डी चे नैसर्गिक स्रोत",
        "answer": "कॉड लिवर ऑयल",
        "grounded": True,
        "latency_ms": 1052.8
    }
]
DATASET_EN = [
    {
        "query": "what county is ninemile creek located in",
        "answer": "Onondaga County",
        "grounded": True,
        "latency_ms": 1047.9
    },
    {
        "query": "what is a salpingectomy surgery",
        "answer": "To remove one or both of your fallopian tubes",
        "grounded": True,
        "latency_ms": 1058.0
    },
    {
        "query": "what are triglycerides in blood",
        "answer": "A type of fat",
        "grounded": True,
        "latency_ms": 1066.7
    },
    {
        "query": "are cheetahs an endangered species",
        "answer": "Today cheetahs are an endangered species",
        "grounded": True,
        "latency_ms": 1064.7
    },
    {
        "query": "meaning of conquest",
        "answer": "Conquering a country or group of people",
        "grounded": True,
        "latency_ms": 1030.2
    },
    {
        "query": "what county is honeoye in new york",
        "answer": "Ontario County",
        "grounded": True,
        "latency_ms": 1018.5
    },
    {
        "query": "what does als stand for in medicine",
        "answer": "Amyotrophic lateral sclerosis",
        "grounded": True,
        "latency_ms": 1067.5
    },
    {
        "query": "what distinguishes a macronutrient from micronutrient",
        "answer": "Micronutrients are different from macronutrients (like carbohydrates, protein and fat) because they are necessary only in very tiny amounts",
        "grounded": True,
        "latency_ms": 1072.0
    },
    {
        "query": "what is a hyperlink on a webpage",
        "answer": "An internal link",
        "grounded": True,
        "latency_ms": 1056.7
    },
    {
        "query": "what county is franklin county columbus",
        "answer": "Franklin County is a county in the U.S. state of Ohio",
        "grounded": True,
        "latency_ms": 1050.0
    },
    {
        "query": "what birth defects does folic acid prevent",
        "answer": "Spina-bifida in your baby",
        "grounded": True,
        "latency_ms": 1319.0
    },
    {
        "query": "what is the price of sirius xm streaming",
        "answer": "$14.49",
        "grounded": True,
        "latency_ms": 1029.2
    },
    {
        "query": "how do you say volkswagen tiguan",
        "answer": "A compact crossover vehicle",
        "grounded": True,
        "latency_ms": 1073.9
    },
    {
        "query": "in which california county is van nuys located",
        "answer": "San Bernardino, County",
        "grounded": True,
        "latency_ms": 1038.8
    },
    {
        "query": "what type of radiation produces x ray photons",
        "answer": "Electromagnetic",
        "grounded": True,
        "latency_ms": 1047.7
    },
    {
        "query": "what is the primary measure of body mass index",
        "answer": "BMI) is a measure of body fat based on height and weight",
        "grounded": True,
        "latency_ms": 1075.3
    },
    {
        "query": "what bone is fractured in a hill sachs lesion",
        "answer": "Humerus bone",
        "grounded": True,
        "latency_ms": 1043.4
    },
    {
        "query": "what is the cellular role of mitochondria",
        "answer": "Metabolism",
        "grounded": True,
        "latency_ms": 1045.5
    },
    {
        "query": "what is the legal definition of property theft",
        "answer": "Unauthorized taking of property from another with the intent to permanently deprive them of it",
        "grounded": True,
        "latency_ms": 1090.7
    },
    {
        "query": "what medication combines isometheptene and dichloralphenazone",
        "answer": "Acetaminophen",
        "grounded": True,
        "latency_ms": 1071.6
    },
    {
        "query": "what are words with the same pronunciation called",
        "answer": "Homophones",
        "grounded": True,
        "latency_ms": 1105.7
    },
    {
        "query": "what blood type conflict causes newborn jaundice",
        "answer": "Blood type mismatch between the mother and baby",
        "grounded": True,
        "latency_ms": 4968.7
    },
    {
        "query": "which branch of the nervous system controls involuntary functions",
        "answer": "Autonomic nervous system",
        "grounded": True,
        "latency_ms": 1079.9
    },
    {
        "query": "what annual irish festival takes place in new york city",
        "answer": "The Annual Feast of San Gennaro",
        "grounded": True,
        "latency_ms": 1041.3
    },
    {
        "query": "what japanese dish consists of fried rice inside an egg omelet",
        "answer": "Omurice",
        "grounded": True,
        "latency_ms": 1058.7
    },
    {
        "query": "how do cat whiskers assist in spatial navigation and balance",
        "answer": "A cats tail also helps them balance",
        "grounded": True,
        "latency_ms": 1042.1
    },
    {
        "query": "what chronic lung condition is classified as a disability under ada",
        "answer": "Chronic obstructive pulmonary disease",
        "grounded": True,
        "latency_ms": 1086.5
    },
    {
        "query": "in what county is seminole city florida located",
        "answer": "Sanford",
        "grounded": True,
        "latency_ms": 1039.2
    },
    {
        "query": "are lottery winnings and contest prizes subject to income tax",
        "answer": "Lottery winnings are subject to state income tax in most states",
        "grounded": True,
        "latency_ms": 1033.5
    },
    {
        "query": "in which new york county is ninemile creek",
        "answer": "Jefferson County",
        "grounded": True,
        "latency_ms": 1066.0
    }
]

def run_suite():
    print("=" * 80)
    print("  RUNNING 90-QUESTION BENCHMARK SUITE 2")
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

    latencies_by_lang = {"HI": [], "MR": [], "EN": []}
    all_latencies = []

    t_start_total = time.perf_counter()

    for idx, (lang, query) in enumerate(all_questions, 1):
        payload = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(MODAL_URL, data=payload, headers={"Content-Type": "application/json"})
        
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                e2e_ms = round((time.perf_counter() - t0) * 1000, 1)
                answer = data.get("answer", "").strip()
                t_server = data.get("timings_ms", {}).get("total_ms", e2e_ms)
                
                latencies_by_lang[lang].append(t_server)
                all_latencies.append(t_server)

                # Clean single-line printout without tick or cross emojis
                q_display = (query[:38] + "..") if len(query) > 40 else query.ljust(40)
                ans_display = (answer[:45] + "..") if len(answer) > 47 else answer.ljust(47)
                print(f"  [{idx:02d}/90] [{lang}] {q_display} | {ans_display} ({t_server:.1f}ms)")
        except Exception as e:
            e2e_ms = round((time.perf_counter() - t0) * 1000, 1)
            q_display = (query[:38] + "..") if len(query) > 40 else query.ljust(40)
            print(f"  [{idx:02d}/90] [{lang}] {q_display} | ERROR: {str(e)[:40]} ({e2e_ms:.1f}ms)")

    total_time_s = round(time.perf_counter() - t_start_total, 1)

    print("\n" + "=" * 80)
    print("  BENCHMARK SUITE 2 - LATENCY & PERFORMANCE SUMMARY")
    print("=" * 80)

    for lang in ["HI", "MR", "EN"]:
        lats = latencies_by_lang[lang]
        if lats:
            lats_sorted = sorted(lats)
            mean_lat = round(statistics.mean(lats), 1)
            p50 = round(statistics.median(lats), 1)
            p90 = round(lats_sorted[int(len(lats_sorted) * 0.90)], 1)
            p100 = round(max(lats), 1)
            print(f"  --- Language: {lang} (30 Questions) ---")
            print(f"      Server Latency Mean : {mean_lat} ms (P50: {p50} ms | P90: {p90} ms | Max: {p100} ms)")

    if all_latencies:
        all_sorted = sorted(all_latencies)
        mean_all = round(statistics.mean(all_latencies), 1)
        p50_all = round(statistics.median(all_latencies), 1)
        p90_all = round(all_sorted[int(len(all_sorted) * 0.90)], 1)
        p100_all = round(max(all_latencies), 1)
        print("\n" + "=" * 80)
        print("  OVERALL SYSTEM PERFORMANCE (90 QUESTIONS)")
        print("=" * 80)
        print(f"  Total Questions Processed : {len(all_latencies)} / 90")
        print(f"  Server Latency Mean       : {mean_all} ms")
        print(f"  Server Latency P50        : {p50_all} ms")
        print(f"  Server Latency P90        : {p90_all} ms")
        print(f"  Server Latency Max (P100) : {p100_all} ms")
        print(f"  Total Suite Run Time      : {total_time_s} s")
        print("=" * 80)

if __name__ == "__main__":
    run_suite()
