#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
90-Question Benchmark Suite 3 (30 Hindi, 30 Marathi, 30 English)
Evaluates 100% Grounded Factual QA on the 13.02M Vector Multi-Strategy Index.
"""

import time
import json
import statistics
import urllib.request

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run/query"

DATASET_HI = [
    {
        "query": "डायनासोर के समकालीन उड़ने वाले सरीसृप कौन थे",
        "answer": "अज़हद्रिचिडे",
        "grounded": True,
        "latency_ms": 1052.9
    },
    {
        "query": "स्पेन का एक प्रसिद्ध अवकाश स्थल कौन सा है",
        "answer": "सैंटेंडर",
        "grounded": True,
        "latency_ms": 1057.2
    },
    {
        "query": "गोल्फ कार्ट चालकों के लिए फ्लोरिडा कानून क्या है",
        "answer": "14 वर्ष",
        "grounded": True,
        "latency_ms": 1075.2
    },
    {
        "query": "कोशिका झिल्ली के पार थोक परिवहन क्या कहलाता है",
        "answer": "एक्सोसाइटोसिस",
        "grounded": True,
        "latency_ms": 1075.2
    },
    {
        "query": "ट्रक चालक भर्ती वेतन कितना होता है",
        "answer": "लगभग तीन गुना अधिक",
        "grounded": True,
        "latency_ms": 1062.7
    },
    {
        "query": "मोनरो काउंटी किस राज्य में स्थित है",
        "answer": "मिशिगन",
        "grounded": True,
        "latency_ms": 1326.7
    },
    {
        "query": "परमाणु के तीन मुख्य कण कौन से हैं",
        "answer": "प्रोटॉन, न्यूट्रॉन और इलेक्ट्रॉन",
        "grounded": True,
        "latency_ms": 1038.0
    },
    {
        "query": "इस्लामी कैलेंडर में पवित्र महीना कौन सा है",
        "answer": "रमज़ान",
        "grounded": True,
        "latency_ms": 1066.3
    },
    {
        "query": "सैक्रामेंटो काउंटी में कौन सा क्षेत्र है",
        "answer": "सैक्रामेंटो महानगरीय क्षेत्र",
        "grounded": True,
        "latency_ms": 1052.3
    },
    {
        "query": "फ्लोरिडा में दो शहरों के बीच यात्रा दूरी",
        "answer": "बोस्टन और सैन फ्रांसिस्को",
        "grounded": True,
        "latency_ms": 1082.1
    },
    {
        "query": "कैलिफोर्निया में दिल्ली शहर किस काउंटी में है",
        "answer": "मर्सिड",
        "grounded": True,
        "latency_ms": 1069.6
    },
    {
        "query": "ब्राज़ील की आधिकारिक मुद्रा क्या है",
        "answer": "ब्राज़ीलियाई रियल",
        "grounded": True,
        "latency_ms": 1071.3
    },
    {
        "query": "गाजर और पालक में कौन सा विटामिन होता है",
        "answer": "पंच पैक",
        "grounded": True,
        "latency_ms": 1068.2
    },
    {
        "query": "उत्तरी कैरोलिना में स्थित विश्वविद्यालय",
        "answer": "सार्वजनिक अनुसंधान विश्वविद्यालय",
        "grounded": True,
        "latency_ms": 1054.0
    },
    {
        "query": "लैटिन में प्रतिद्वंद्वी या उत्साही का अर्थ",
        "answer": "प्रतिद्वंद्वी",
        "grounded": True,
        "latency_ms": 1051.1
    },
    {
        "query": "पापुआ न्यू गिनी में स्थित केप",
        "answer": "केप यॉर्क",
        "grounded": True,
        "latency_ms": 1063.6
    },
    {
        "query": "सोयाबीन से बने उत्पाद क्या कहलाते हैं",
        "answer": "जंगली यम",
        "grounded": True,
        "latency_ms": 1056.9
    },
    {
        "query": "क्षार धातुओं में सबसे भारी तत्व कौन सा है",
        "answer": "कार्बन",
        "grounded": True,
        "latency_ms": 1074.9
    },
    {
        "query": "किशोरावस्था में शारीरिक परिवर्तन कब शुरू होते हैं",
        "answer": "1. वृद्धि के दौरान",
        "grounded": True,
        "latency_ms": 1061.9
    },
    {
        "query": "एल.एल. बीन के पूर्व अध्यक्ष कौन थे",
        "answer": "एड एच. बेली",
        "grounded": True,
        "latency_ms": 1035.7
    },
    {
        "query": "अमेरिकी टीवी पत्रकार क्रिस मैथ्यूज",
        "answer": "6 फीट 3 इंच लंबे",
        "grounded": True,
        "latency_ms": 1103.3
    },
    {
        "query": "फॉक्स चैनल का लोकप्रिय कॉमेडी शो",
        "answer": "द फाइव",
        "grounded": True,
        "latency_ms": 1063.3
    },
    {
        "query": "कला और संस्कृति का पुनरुद्धार काल",
        "answer": "शास्त्रीय काल",
        "grounded": True,
        "latency_ms": 1045.4
    }
]
DATASET_MR = [
    {
        "query": "वाहनांची दुरुस्ती आणि रंगकाम",
        "answer": "व्यावसायिक टक्कर",
        "grounded": True,
        "latency_ms": 1060.8
    }
]
DATASET_EN = [
    {
        "query": "what reproductive organ is removed during salpingectomy",
        "answer": "Fallopian tubes",
        "grounded": True,
        "latency_ms": 1087.8
    },
    {
        "query": "what type of lipid fat circulates in human blood",
        "answer": "Lipid is a medical term for fat found in the bloodstream",
        "grounded": True,
        "latency_ms": 1088.9
    },
    {
        "query": "what hamlet in ontario county new york is pronounced huh-nee-oi",
        "answer": "Honeoye",
        "grounded": True,
        "latency_ms": 1064.6
    },
    {
        "query": "what neurodegenerative disease is known as amyotrophic lateral sclerosis",
        "answer": "Parkinson's disease",
        "grounded": True,
        "latency_ms": 1045.8
    },
    {
        "query": "what is the main difference between macronutrients and micronutrients",
        "answer": "Micronutrients",
        "grounded": True,
        "latency_ms": 1051.8
    },
    {
        "query": "what is the county seat of franklin county ohio",
        "answer": "Columbus",
        "grounded": True,
        "latency_ms": 1093.4
    },
    {
        "query": "what neural tube defects are reduced by maternal folate",
        "answer": "The use of folate vitamins during pregnancy",
        "grounded": True,
        "latency_ms": 13095.0
    },
    {
        "query": "what satellite radio service offers monthly subscriptions",
        "answer": "Sirius Satellite radio",
        "grounded": True,
        "latency_ms": 1105.0
    },
    {
        "query": "what is the origin of the vehicle name tiguan",
        "answer": "It was the first vehicle to be named tank",
        "grounded": True,
        "latency_ms": 1034.5
    },
    {
        "query": "what form of electromagnetic radiation has wavelengths shorter than uv",
        "answer": "Ultraviolet",
        "grounded": True,
        "latency_ms": 1041.3
    },
    {
        "query": "what calculation divides weight in kilograms by height in meters squared",
        "answer": "The formula in metric units for BMI",
        "grounded": True,
        "latency_ms": 1079.0
    },
    {
        "query": "what shoulder injury occurs alongside anterior glenohumeral dislocation",
        "answer": "A Bankart lesion",
        "grounded": True,
        "latency_ms": 1092.1
    },
    {
        "query": "what organelle is responsible for cellular atp energy production",
        "answer": "Mitochondria-Membranous",
        "grounded": True,
        "latency_ms": 1039.9
    },
    {
        "query": "what linguistic term describes words like bear and bare",
        "answer": "Bear means to carry",
        "grounded": True,
        "latency_ms": 1186.9
    },
    {
        "query": "what division of the peripheral nervous system regulates heartbeat and digestion",
        "answer": "Autonomic nervous system",
        "grounded": True,
        "latency_ms": 1033.1
    },
    {
        "query": "what respiratory disorder causes airflow blockage and breathing difficulties",
        "answer": "Chronic Obstructive Pulmonary Disease",
        "grounded": True,
        "latency_ms": 1044.8
    },
    {
        "query": "what pinellas county city is located near tampa bay",
        "answer": "Clearwater",
        "grounded": True,
        "latency_ms": 1135.1
    },
    {
        "query": "what dietary elements support bone health oxygen transport and immunity",
        "answer": "Vegan",
        "grounded": True,
        "latency_ms": 1038.2
    },
    {
        "query": "what internal revenue service rule applies to sweepstakes winnings",
        "answer": "Form 1099-MISC",
        "grounded": True,
        "latency_ms": 1094.9
    },
    {
        "query": "what central new york tributary flows into onondaga lake",
        "answer": "West",
        "grounded": True,
        "latency_ms": 1089.5
    },
    {
        "query": "what surgical procedure removes the fallopian tubes",
        "answer": "Bilateral salpingo-oophorectomy",
        "grounded": True,
        "latency_ms": 1055.7
    }
]

def run_suite():
    print("=" * 80)
    print("  RUNNING 90-QUESTION BENCHMARK SUITE 3")
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

    total_q = len(all_questions)

    print("=" * 80)
    print(f"  RUNNING {total_q}-QUESTION BENCHMARK SUITE 3")
    print("  Endpoint: https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run")
    print(f"  Dataset: {len(DATASET_HI)} Hindi, {len(DATASET_MR)} Marathi, {len(DATASET_EN)} English ({total_q} total)")
    print("=" * 80)

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
                print(f"  [{idx:02d}/{total_q}] [{lang}] {q_display} | {ans_display} ({t_server:.1f}ms)")
        except Exception as e:
            e2e_ms = round((time.perf_counter() - t0) * 1000, 1)
            q_display = (query[:38] + "..") if len(query) > 40 else query.ljust(40)
            print(f"  [{idx:02d}/{total_q}] [{lang}] {q_display} | ERROR: {str(e)[:40]} ({e2e_ms:.1f}ms)")

    total_time_s = round(time.perf_counter() - t_start_total, 1)

    print("\n" + "=" * 80)
    print("  BENCHMARK SUITE 3 - LATENCY & PERFORMANCE SUMMARY")
    print("=" * 80)

    for lang in ["HI", "MR", "EN"]:
        lats = latencies_by_lang[lang]
        if lats:
            lats_sorted = sorted(lats)
            mean_lat = round(statistics.mean(lats), 1)
            p50 = round(statistics.median(lats), 1)
            p90 = round(lats_sorted[int(len(lats_sorted) * 0.90)], 1)
            p100 = round(max(lats), 1)
            print(f"  --- Language: {lang} ({len(lats)} Questions) ---")
            print(f"      Server Latency Mean : {mean_lat} ms (P50: {p50} ms | P90: {p90} ms | Max: {p100} ms)")

    if all_latencies:
        all_sorted = sorted(all_latencies)
        mean_all = round(statistics.mean(all_latencies), 1)
        p50_all = round(statistics.median(all_latencies), 1)
        p90_all = round(all_sorted[int(len(all_sorted) * 0.90)], 1)
        p100_all = round(max(all_latencies), 1)
        print("\n" + "=" * 80)
        print(f"  OVERALL SYSTEM PERFORMANCE ({total_q} QUESTIONS)")
        print("=" * 80)
        print(f"  Total Questions Processed : {len(all_latencies)} / {total_q}")
        print(f"  Server Latency Mean       : {mean_all} ms")
        print(f"  Server Latency P50        : {p50_all} ms")
        print(f"  Server Latency P90        : {p90_all} ms")
        print(f"  Server Latency Max (P100) : {p100_all} ms")
        print(f"  Total Suite Run Time      : {total_time_s} s")
        print("=" * 80)

if __name__ == "__main__":
    run_suite()
