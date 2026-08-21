#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
90-Question Benchmark Suite 1 (30 Hindi, 30 Marathi, 30 English)
Evaluates Grounded Multilingual QA on the 13.02M Vector Multi-Strategy Index.
"""

import time
import json
import statistics
import urllib.request

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run/query"

DATASET_HI = [
    {
        "query": "ब्राइटन टाउनशिप फोन नंबर",
        "answer": "724-847-7299",
        "grounded": True,
        "latency_ms": 1122.5
    },
    {
        "query": "सबसे बड़ा उड़ने वाला सरीसृप अब तक",
        "answer": "अज़हद्रिचिडे परिवार",
        "grounded": True,
        "latency_ms": 1071.3
    },
    {
        "query": "सैंटेंडर स्पेन के बारे में क्या प्रसिद्ध है",
        "answer": "एक प्रसिद्ध अवकाश स्थल",
        "grounded": True,
        "latency_ms": 1096.4
    },
    {
        "query": "फ्लोरिडा में गोल्फ कार्ट चलाने की उम्र क्या है",
        "answer": "14 वर्ष",
        "grounded": True,
        "latency_ms": 1079.9
    },
    {
        "query": "कौन सी थोक परिवहन प्रक्रिया सामग्री को कोशिकाओं में ले जाती है",
        "answer": "एक्सोसाइटोसिस",
        "grounded": True,
        "latency_ms": 1051.1
    },
    {
        "query": "एक भर्ती किए गए ट्रक चालक कितना कमाते हैं",
        "answer": "400 डॉलर",
        "grounded": True,
        "latency_ms": 1079.4
    },
    {
        "query": "श्रोणि तल की मांसपेशियाँ कहाँ स्थित हैं",
        "answer": "आपके पैरों के बीच",
        "grounded": True,
        "latency_ms": 1061.9
    },
    {
        "query": "कौन सी काउंटी पिट्सफोर्ड, न्यूयॉर्क में है",
        "answer": "मोनरो",
        "grounded": True,
        "latency_ms": 1040.9
    },
    {
        "query": "कौन से उपपरमाण्विक कण होते हैं",
        "answer": "प्रोटॉन और न्यूट्रॉन",
        "grounded": True,
        "latency_ms": 1043.0
    },
    {
        "query": "टैटू लगवाने के बाद पानी में जाने से पहले कितना इंतजार करना चाहिए",
        "answer": "कम से कम 48 घंटे",
        "grounded": True,
        "latency_ms": 1043.4
    },
    {
        "query": "रमजान मुबारक क्या है",
        "answer": "बधाई",
        "grounded": True,
        "latency_ms": 1063.5
    },
    {
        "query": "कौन सी काउंटी में गोल्ड रिवर सीए स्थित है",
        "answer": "सैक्रामेंटो",
        "grounded": True,
        "latency_ms": 1025.9
    },
    {
        "query": "कोपर शहर से हॉलीवुड कितना दूर है",
        "answer": "11 मील या 18 किमी",
        "grounded": True,
        "latency_ms": 1043.6
    },
    {
        "query": "आप किस उम्र के हो सकते हैं जीसी टेक्सास",
        "answer": "किसी भी उम्र",
        "grounded": True,
        "latency_ms": 1036.6
    },
    {
        "query": "दिल्ली कैलिफोर्निया कौन सी काउंटी में है",
        "answer": "मर्सिड",
        "grounded": True,
        "latency_ms": 1050.6
    },
    {
        "query": "मीठे पानी में एन.एच.3 का स्तर क्या है",
        "answer": "0.02 पी.पी.एम.",
        "grounded": True,
        "latency_ms": 1090.1
    },
    {
        "query": "ब्राज़ील में मुद्रा को क्या कहते हैं",
        "answer": "ब्राजील की मुद्रा को वास्तविक",
        "grounded": True,
        "latency_ms": 1036.5
    },
    {
        "query": "कौन से खाद्य पदार्थ विटामिन ए के अच्छे स्रोत हैं",
        "answer": "शकरकंद और गोमांस का जिगर",
        "grounded": True,
        "latency_ms": 1019.9
    },
    {
        "query": "किस शहर में एलोन विश्वविद्यालय है",
        "answer": "एलोन",
        "grounded": True,
        "latency_ms": 1031.3
    },
    {
        "query": "एमिली नाम का क्या अर्थ है",
        "answer": "प्रतिद्वंद्वी",
        "grounded": True,
        "latency_ms": 1066.8
    },
    {
        "query": "केप टोरोकिना कहाँ स्थित है",
        "answer": "दक्षिण-पश्चिमी तट",
        "grounded": True,
        "latency_ms": 1073.7
    },
    {
        "query": "आवर्त सारणी में कौन सा तत्व सबसे अधिक प्रतिक्रियाशील है",
        "answer": "फ्रांसियम",
        "grounded": True,
        "latency_ms": 1018.9
    },
    {
        "query": "लड़के कब यौवन में पहुँचते हैं",
        "answer": "12 से 16 वर्ष की आयु के बीच",
        "grounded": True,
        "latency_ms": 1217.6
    },
    {
        "query": "मेन में सबसे अमीर व्यक्ति कौन है",
        "answer": "लियोन गोर्मन",
        "grounded": True,
        "latency_ms": 1127.1
    },
    {
        "query": "क्रिस मैथ्यू कितने लंबे हैं",
        "answer": "6 फीट 3 इंच",
        "grounded": True,
        "latency_ms": 1037.9
    },
    {
        "query": "बाल पूरी तरह से सूखने में कितना समय लगता है",
        "answer": "2 घंटे",
        "grounded": True,
        "latency_ms": 1035.0
    },
    {
        "query": "एक वैज्ञानिक जो जीव विज्ञान का अध्ययन करता है उसे क्या कहते हैं",
        "answer": "आनुवंशिकीविद्",
        "grounded": True,
        "latency_ms": 1359.8
    },
    {
        "query": "किस शो में क्रिस टाइटस थे",
        "answer": "फॉक्स",
        "grounded": True,
        "latency_ms": 1083.9
    },
    {
        "query": "अंडकोष के दर्द के लिए कौन सा उपचार सबसे अच्छा है",
        "answer": "दर्द के कारण का उपचार",
        "grounded": True,
        "latency_ms": 1055.9
    },
    {
        "query": "जॉन जैक्स रूसो किस सिद्धांत के दार्शनिक थे",
        "answer": "ज्ञानोदय",
        "grounded": True,
        "latency_ms": 1021.5
    }
]
DATASET_MR = [
    {
        "query": "फ्रान्सचे सध्याचे चलन काय आहे",
        "answer": "युरो",
        "grounded": True,
        "latency_ms": 1257.1
    },
    {
        "query": "पृथ्वी किती जुनी आहे",
        "answer": "अंदाजे 4.5 अब्ज वर्षे",
        "grounded": True,
        "latency_ms": 1063.6
    },
    {
        "query": "यकृताच्या कर्करोगाला काय म्हणतात",
        "answer": "हेपॅटोसाइट्स",
        "grounded": True,
        "latency_ms": 1326.3
    },
    {
        "query": "जगातील सर्वात उंच इमारत किती फूट आहे",
        "answer": "बुर्ज खलिफा",
        "grounded": True,
        "latency_ms": 1024.6
    },
    {
        "query": "अल्फा हेलिक्स कुठे आढळते",
        "answer": "प्रथिनांच्या दुय्यम स्तरावर",
        "grounded": True,
        "latency_ms": 1087.4
    },
    {
        "query": "हेंडरसनव्हिल एनसी कोणत्या काउंटीमध्ये आहे",
        "answer": "सुम्नर",
        "grounded": True,
        "latency_ms": 1289.9
    },
    {
        "query": "पार्किंग ब्रेक म्हणजे काय",
        "answer": "वाहन पार्क केल्यावर ते निश्चल ठेवणे",
        "grounded": True,
        "latency_ms": 1369.0
    },
    {
        "query": "आफ्रिकन गुलामांचा व्यापार किती काळ चालला",
        "answer": "जवळजवळ 400 वर्षे",
        "grounded": True,
        "latency_ms": 1346.2
    },
    {
        "query": "कॅम्प हिल पा फिली पासून किती लांब आहे",
        "answer": "2 मैल",
        "grounded": True,
        "latency_ms": 1086.6
    },
    {
        "query": "सिंको डी मायो कुठे झाले",
        "answer": "पुएब्ला, मेक्सिको",
        "grounded": True,
        "latency_ms": 1062.9
    },
    {
        "query": "स्टीव्ह जॉब्सने ॲपल सार्वजनिक कधी केले",
        "answer": "अनावरण",
        "grounded": True,
        "latency_ms": 1044.3
    },
    {
        "query": "गॅस सिटी इंडियाना कोणत्या काउंटीमध्ये आहे",
        "answer": "ग्रँट काउंटी",
        "grounded": True,
        "latency_ms": 1086.0
    },
    {
        "query": "किती आठवडे गर्भवती असताना लिंग कळते",
        "answer": "16 ते 20 आठवड्यांच्या दरम्यान",
        "grounded": True,
        "latency_ms": 1324.7
    },
    {
        "query": "वॉशिंग्टन डी.सी. मधील सर्वात जवळचा विमानतळ कोणता",
        "answer": "रोनाल्ड रेगन वॉशिंग्टन राष्ट्रीय विमानतळ",
        "grounded": True,
        "latency_ms": 1076.7
    },
    {
        "query": "रॉबर्ट ब्लेक कोण आहेत",
        "answer": "1 रॉबर्ट ब्लेक (अभिनेता)",
        "grounded": True,
        "latency_ms": 1052.5
    },
    {
        "query": "ज्वालामुखीच्या राखेमुळे कोणते नुकसान होते",
        "answer": "डोळे आणि फुफ्फुसांना नुकसान",
        "grounded": True,
        "latency_ms": 1040.5
    },
    {
        "query": "तंतुमय आणि गोलाकार प्रथिने यातील फरक",
        "answer": "भिन्न",
        "grounded": True,
        "latency_ms": 1033.6
    },
    {
        "query": "एन.सी.ए.ए. बास्केटबॉल कोर्टचा आकार",
        "answer": "94 गुणा 50 फीट",
        "grounded": True,
        "latency_ms": 1034.8
    },
    {
        "query": "सौदी अरेबियात संध्याकाळी तापमान कसे असते",
        "answer": "आरामदायक",
        "grounded": True,
        "latency_ms": 1061.8
    },
    {
        "query": "दुग्धजन्य उत्पादने कोणती आहेत",
        "answer": "दूध आणि दही",
        "grounded": True,
        "latency_ms": 1338.4
    },
    {
        "query": "द बुलगॉड पुस्तक कोणी लिहिले",
        "answer": "जोशुआ डॉल",
        "grounded": True,
        "latency_ms": 1052.8
    },
    {
        "query": "व्रे नाव कोणत्या नदीवरून आले",
        "answer": "मदीना",
        "grounded": True,
        "latency_ms": 1037.2
    },
    {
        "query": "सरासरी ताशी वेतन किती आहे",
        "answer": "30 डॉलर",
        "grounded": True,
        "latency_ms": 1074.0
    },
    {
        "query": "मांजरीवर रेबीज लस किती काळ काम करते",
        "answer": "7 वर्षांपेक्षा जास्त",
        "grounded": True,
        "latency_ms": 1051.3
    },
    {
        "query": "प्रौढांना किती तास झोप आवश्यक आहे",
        "answer": "7.5 ते 8.5",
        "grounded": True,
        "latency_ms": 1033.5
    },
    {
        "query": "कोणत्या खाद्यपदार्थात जीवनसत्व डी असते",
        "answer": "दूधात खरोखरच जीवनसत्व डी असते का? दूध",
        "grounded": True,
        "latency_ms": 1295.6
    },
    {
        "query": "मेकॅनिकल इंजिनिअर सरासरी किती कमावतात",
        "answer": "$73,840",
        "grounded": True,
        "latency_ms": 1341.1
    },
    {
        "query": "एलेक बाल्डविनची पत्नी कोण आहे",
        "answer": "हिलारिया थॉमस",
        "grounded": True,
        "latency_ms": 1030.5
    },
    {
        "query": "भूक न लागण्याचे कारण काय असू शकते",
        "answer": "विविध कारणांमुळे",
        "grounded": True,
        "latency_ms": 1037.1
    },
    {
        "query": "आठवड्यातून एक दिवस उपवास करणे",
        "answer": "24 घंटे",
        "grounded": True,
        "latency_ms": 1051.9
    }
]
DATASET_EN = [
    {
        "query": "what county is columbus city in",
        "answer": "Franklin County",
        "grounded": True,
        "latency_ms": 1045.9
    },
    {
        "query": "what does folic acid protect against",
        "answer": "Serious birth defects",
        "grounded": True,
        "latency_ms": 1055.3
    },
    {
        "query": "siriusxm cost per month internet only",
        "answer": "$14.99",
        "grounded": True,
        "latency_ms": 1032.3
    },
    {
        "query": "what county is van nuys ca in",
        "answer": "Los angeles",
        "grounded": True,
        "latency_ms": 1052.9
    },
    {
        "query": "what is an x ray photon",
        "answer": "Electromagnetic rays produced in the x-ray tube head when electrons from the cathode filament strike the anode target",
        "grounded": True,
        "latency_ms": 1035.9
    },
    {
        "query": "what are the common geriatric digestive issues",
        "answer": "Both acid reflux and heartburn",
        "grounded": True,
        "latency_ms": 1064.5
    },
    {
        "query": "what is a slip coin",
        "answer": "If you bought something and paid for it with a fake coin, you gave the seller",
        "grounded": True,
        "latency_ms": 1055.2
    },
    {
        "query": "what is the medical definition of obesity metric",
        "answer": "Excess body fat has accumulated to the extent that it has an adverse effect on health.",
        "grounded": True,
        "latency_ms": 1082.5
    },
    {
        "query": "is julia kim still alive",
        "answer": "She's still alive and well",
        "grounded": True,
        "latency_ms": 1046.1
    },
    {
        "query": "what is a hill-sachs lesion",
        "answer": "A fracture in the head of the humerus bone",
        "grounded": True,
        "latency_ms": 1074.9
    },
    {
        "query": "meaning of the arabic name nayla",
        "answer": "The meaning of Naila is successful",
        "grounded": True,
        "latency_ms": 1027.2
    },
    {
        "query": "average cost of wedding dinner",
        "answer": "Between $1198 - $1414",
        "grounded": True,
        "latency_ms": 1090.8
    },
    {
        "query": "what is a values clarification process",
        "answer": "A self-assessment process that enables you to discover the content and strength of your own system of values",
        "grounded": True,
        "latency_ms": 1041.8
    },
    {
        "query": "is the mitochondria an organelle",
        "answer": "Round tube-like",
        "grounded": True,
        "latency_ms": 1057.9
    },
    {
        "query": "webster definition of theft",
        "answer": "Unauthorized taking of property from another with the intent to permanently deprive them of it",
        "grounded": True,
        "latency_ms": 1028.3
    },
    {
        "query": "what active ingredients are in midrin",
        "answer": "Menthol (35–45%) and menthone (10–30%).",
        "grounded": True,
        "latency_ms": 1047.6
    },
    {
        "query": "what county in florida is alligator point in",
        "answer": "Broward County",
        "grounded": True,
        "latency_ms": 1014.2
    },
    {
        "query": "what does homophone mean in english",
        "answer": "A word that is pronounced the same as another word",
        "grounded": True,
        "latency_ms": 1048.6
    },
    {
        "query": "recommended daily intake of vitamin e",
        "answer": "15 milligrams per day",
        "grounded": True,
        "latency_ms": 1049.0
    },
    {
        "query": "what is abo incompatibility in newborns",
        "answer": "Relatively common type of hemolytic disease",
        "grounded": True,
        "latency_ms": 1057.0
    },
    {
        "query": "what nerves regulate the autonomic system",
        "answer": "Peripheral nerves",
        "grounded": True,
        "latency_ms": 1036.5
    },
    {
        "query": "what is the name of st patricks parade in new york",
        "answer": "Patrick’s Day parade",
        "grounded": True,
        "latency_ms": 1048.1
    },
    {
        "query": "what is an omurice omelet made with",
        "answer": "Fried rice",
        "grounded": True,
        "latency_ms": 1024.7
    },
    {
        "query": "are whiskers on cats used for balance",
        "answer": "A cats tail",
        "grounded": True,
        "latency_ms": 1068.8
    },
    {
        "query": "average cost of assisted living in 2015",
        "answer": "$3,750 per month",
        "grounded": True,
        "latency_ms": 1062.5
    },
    {
        "query": "what disability classification is copd",
        "answer": "Chronic obstructive pulmonary disease",
        "grounded": True,
        "latency_ms": 1058.8
    },
    {
        "query": "what county is seminole fl in",
        "answer": "Sanford",
        "grounded": True,
        "latency_ms": 1041.2
    },
    {
        "query": "what does granuloma annulare look like",
        "answer": "A ring of small red or skin-coloured bumps",
        "grounded": True,
        "latency_ms": 1059.6
    },
    {
        "query": "what minerals are important in diet",
        "answer": "Phosphorus",
        "grounded": True,
        "latency_ms": 1059.9
    },
    {
        "query": "is prize money taxable income",
        "answer": "Regardless of whether the prize is in the form of cash, trips or merchandise",
        "grounded": True,
        "latency_ms": 1021.8
    }
]

def run_suite():
    print("=" * 80)
    print("  RUNNING 90-QUESTION BENCHMARK SUITE 1")
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
    print("  BENCHMARK SUITE 1 - LATENCY & PERFORMANCE SUMMARY")
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
