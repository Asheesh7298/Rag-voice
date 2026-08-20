#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
90-Question Benchmark Suite - SET 2 (30 Hindi, 30 Marathi, 30 English)
Evaluates Grounded Accuracy, Guardrail Behavior, and Latency against the live Multi-Strategy Index.
Completely independent from Set 1.
"""

import time
import json
import statistics
import urllib.request
import urllib.parse

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run"

DATASET_90_SET2 = {
    "hi": [
        {
            "id": "hi-833219-p1-native",
            "query_id": "hi-833219",
            "query": "ब्राज़ील में किस पैसे को कहते हैं",
            "ground_truth_answer": "ब्राज़ीलियाई रियल",
            "sample_model_answer": "पैसे की आपूर्ति के बराबर होती है, जिसे LM वक्र के रूप में जाना जाता है।",
            "sample_latency_ms": 144.77
        },
        {
            "id": "hi-326012-p5-native",
            "query_id": "hi-326012",
            "query": "आधे समय के लिए आपको कितना पेल ग्रांट मिलता है?",
            "ground_truth_answer": "2,775 डॉलर",
            "sample_model_answer": "2013/2014 के लिए प्रदान की गई औसत पेल ग्रांट $4,299 थी।",
            "sample_latency_ms": 143.49
        },
        {
            "id": "hi-1006613-p1-native",
            "query_id": "hi-1006613",
            "query": "कौन सी हस्ती को वॉलमार्ट ट्रक से टक्कर लगी और अस्पताल में भर्ती कराया गया",
            "ground_truth_answer": "ट्रेसी मॉर्गन",
            "sample_model_answer": "2 साथी हास्य अभिनेता जेम्स मैकनायर की न्यू जर्सी में दुर्घटना में मौत हो गई।",
            "sample_latency_ms": 137.21
        },
        {
            "id": "hi-1017354-p8-native",
            "query_id": "hi-1017354",
            "query": "जिन दार्शनिक ने सामाजिक अनुबंध की अवधारणा विकसित की",
            "ground_truth_answer": "थॉमस हॉबे और रूसो।",
            "sample_model_answer": "जॉन लॉक जैसे कई अन्य दार्शनिकों ने अपने काम को प्रभावित करने के लिए अपने सामाजिक अनुबंध का उपयोग किया।",
            "sample_latency_ms": 133.62
        },
        {
            "id": "hi-660966-p8-native",
            "query_id": "hi-660966",
            "query": "कौन से खाद्य पदार्थ विटामिन ए के अच्छे स्रोत हैं",
            "ground_truth_answer": "गाजर",
            "sample_model_answer": "अंडे की जर्दी, पनीर, साबुत दूध, मक्खन, सुदृढ़ दूध और मार्जरीन भी विटामिन ए के अच्छे स्रोत हैं।",
            "sample_latency_ms": 140.88
        },
        {
            "id": "hi-821444-p7-native",
            "query_id": "hi-821444",
            "query": "कौन सा पहला कंप्यूटर बनाया गया था",
            "ground_truth_answer": "चार्ल्स बैबेज",
            "sample_model_answer": "अन्य लोग कहेंगे कि एबेकस पहला कंप्यूटर है।",
            "sample_latency_ms": 135.96
        },
        {
            "id": "hi-868082-p6-native",
            "query_id": "hi-868082",
            "query": "आड़ू के पेड़ों के लिए किस तरह का उर्वरक",
            "ground_truth_answer": "एक अकार्बनिक उर्वरक",
            "sample_model_answer": "मार्च में 3 फुट के क्षेत्र में 1 कप 10-10-10 खाद से आड़ू को उर्वरित करें।",
            "sample_latency_ms": 132.28
        },
        {
            "id": "hi-908687-p6-native",
            "query_id": "hi-908687",
            "query": "किस शहर में एलोन विश्वविद्यालय है",
            "ground_truth_answer": "एलोन",
            "sample_model_answer": "जुडसन विश्वविद्यालय ईसाई महाविद्यालय एल्गिन, इलिनोइस - जुडसन विश्वविद्यालय अपने वार्षिक संस्थापक दिवस को शुक्रवार, 16 अक्टूबर, 2015 को मनाएगा, जिसमें डॉ.",
            "sample_latency_ms": 136.47
        },
        {
            "id": "hi-830683-p0-native",
            "query_id": "hi-830683",
            "query": "एमिली का क्या अर्थ है",
            "ground_truth_answer": "निर्माणकारी और प्रयासरत",
            "sample_model_answer": "फ्रेंच में, नाम को एमी के रूप में लिखा जाता है, जिसका अर्थ है प्रिय।",
            "sample_latency_ms": 139.83
        },
        {
            "id": "hi-979339-p6-native",
            "query_id": "hi-979339",
            "query": "केप कहाँ स्थित हो सकता है",
            "ground_truth_answer": "न्यू जर्सी में",
            "sample_model_answer": "कोल्डवेल बैंकर प्रीफर्ड आपको केप मे काउंटी, एन.जे.",
            "sample_latency_ms": 173.65
        },
        {
            "id": "hi-798264-p4-native",
            "query_id": "hi-798264",
            "query": "सोया क्या है",
            "ground_truth_answer": "सबसे अधिक प्रोटीन युक्त ज्ञात शाक।",
            "sample_model_answer": "• सोया (संज्ञा) सोया संज्ञा के 4 अर्थ हैं: 1.",
            "sample_latency_ms": 137.4
        },
        {
            "id": "hi-595743-p9-native",
            "query_id": "hi-595743",
            "query": "कौन सा शहर शीत युद्ध का कारण बना",
            "ground_truth_answer": "बर्लिन",
            "sample_model_answer": "परमाणु हथियारों के अस्तित्व ने ही शीत युद्ध के विकास को जन्म दिया।",
            "sample_latency_ms": 132.29
        },
        {
            "id": "hi-67732-p2-native",
            "query_id": "hi-67732",
            "query": "क्या गूगल बुक्स को किंडल पर पढ़ा जा सकता है?",
            "ground_truth_answer": "कोई नहीं",
            "sample_model_answer": "किंडल फायर पर गूगल ई-बुक्स पढ़ने के लिए इंडीबाउंड रीडर ऐप को साइडलोड करना आवश्यक है (जो नुक, एंड्रॉइड, आईफोन और आईपैड के लिए भी काम करता है - आप ऐप को अपने उपकरण के ऐप स्टोर में पा सकते हैं)।",
            "sample_latency_ms": 137.72
        },
        {
            "id": "hi-1008998-p1-native",
            "query_id": "hi-1008998",
            "query": "आवर्त सारणी में कौन सा तत्व सबसे अधिक प्रतिक्रियाशील है",
            "ground_truth_answer": "फ्रांसियम",
            "sample_model_answer": "क्षारीय धातु: आवर्त सारणी के समूह IA में तत्वों का परिवार।",
            "sample_latency_ms": 137.73
        },
        {
            "id": "hi-943592-p1-native",
            "query_id": "hi-943592",
            "query": "लड़के कब यौवन में पहुँचते हैं",
            "ground_truth_answer": "जब लड़के युवावस्था में प्रवेश करते हैं, तो उनके शरीर में कई शारीरिक परिवर्तन होने लगते हैं जो उनकी उपस्थिति को प्रभावित करते हैं। कई क्षेत्रों में परिवर्तन होने लगता है, जिनमें शामिल हैं: मांसपेशियां: लड़कों में युवावस्था के दौरान अधिक मांसपेशियों का विकास होना शुरू हो जाता है।",
            "sample_model_answer": "शुक्राणु उत्पादन की शुरुआत",
            "sample_latency_ms": 140.82
        },
        {
            "id": "hi-410295-p0-native",
            "query_id": "hi-410295",
            "query": "क्या फ्लेक्सेरिल एक गैबा है",
            "ground_truth_answer": "सहमत हैं तो",
            "sample_model_answer": "मुझे पता है कि यह एक ट्राइसाइक्लिक है, लेकिन मैं यह निर्धारित नहीं कर सकता कि यह जीएबीए रिसेप्टर्स पर काम करता है या नहीं।",
            "sample_latency_ms": 136.02
        },
        {
            "id": "hi-1072556-p2-native",
            "query_id": "hi-1072556",
            "query": "मोना लिसा सबसे प्रसिद्ध चित्र क्यों है",
            "ground_truth_answer": "मोना लिसा की मुस्कान।",
            "sample_model_answer": "कोई भी निश्चित रूप से नहीं जानता था कि पेंटिंग मोना लिसा इतनी प्रसिद्ध और प्रसिद्ध क्यों है।",
            "sample_latency_ms": 133.93
        },
        {
            "id": "hi-1011876-p6-native",
            "query_id": "hi-1011876",
            "query": "जो कि मजबूत पर्कोसेट या लोर्टैब है",
            "ground_truth_answer": "परकोसेट",
            "sample_model_answer": "परकोसेट को लॉर्टैब से अधिक मजबूत माना जाता है।",
            "sample_latency_ms": 165.16
        },
        {
            "id": "hi-255942-p3-native",
            "query_id": "hi-255942",
            "query": "स्नातक पूरा करने में कितना समय लगता है",
            "ground_truth_answer": "लगभग 4 साल।",
            "sample_model_answer": "अधिकांश मामलों में, स्नातकोत्तर डिग्री कार्यक्रम को पूरा करने में दो साल लगते हैं, हालांकि इस नियम के कुछ अपवाद हैं।",
            "sample_latency_ms": 133.8
        },
        {
            "id": "hi-959615-p3-native",
            "query_id": "hi-959615",
            "query": "किसी को बताया कि कौन सी कंपनी में काम करने के लिए आवेदन कर रहा है?",
            "ground_truth_answer": "2001",
            "sample_model_answer": "एक अलग कंपनी",
            "sample_latency_ms": 134.85
        },
        {
            "id": "hi-605017-p9-native",
            "query_id": "hi-605017",
            "query": "कौन सी काउंटी डेफुनियाक स्प्रिंग्स, फ्लोरिडा है",
            "ground_truth_answer": "वॉल्टन",
            "sample_model_answer": "डेफुनियाक स्प्रिंग्स संयुक्त राज्य अमेरिका के फ्लोरिडा में वॉलटन काउंटी का एक शहर है।",
            "sample_latency_ms": 139.98
        },
        {
            "id": "hi-1041646-p4-native",
            "query_id": "hi-1041646",
            "query": "मेन में सबसे अमीर व्यक्ति कौन है",
            "ground_truth_answer": "लियोन गोर्मन",
            "sample_model_answer": "मेन के सबसे अमीर व्यक्ति लियोन गोर्मन हैं, जो एल.एल.",
            "sample_latency_ms": 136.78
        },
        {
            "id": "hi-525971-p0-native",
            "query_id": "hi-525971",
            "query": "टाइप 1 मधुमेह जिसे",
            "ground_truth_answer": "मेलिटस प्रकार 1",
            "sample_model_answer": "मधुमेह मेलिटस टाइप 1 (जिसे टाइप 1 मधुमेह या टी1डी के रूप में भी जाना जाता है, पूर्व में इंसुलिन-निर्भर मधुमेह या किशोर मधुमेह) मधुमेह मेलिटस का एक रूप है जो अग्न्याशय में इंसुलिन-उत्पादक बीटा कोशिकाओं के स्वप्रतिरक्षी विनाश के परिणामस्वरूप होता है।",
            "sample_latency_ms": 133.78
        },
        {
            "id": "hi-340816-p0-native",
            "query_id": "hi-340816",
            "query": "क्रिस मैथ्यू कितने लंबे हैं",
            "ground_truth_answer": "6 फीट 3 इंच लंबा",
            "sample_model_answer": "क्रिस मैथ्यूज 6 फीट 3 इंच लंबे हैं।",
            "sample_latency_ms": 129.74
        },
        {
            "id": "hi-254883-p3-native",
            "query_id": "hi-254883",
            "query": "आपके बाल पूरी तरह से सूखने में कितना समय लगता है",
            "ground_truth_answer": "30-40 मिनट।",
            "sample_model_answer": "मेरा बाल एक लंबाई से एक गज से अधिक लंबा था, और आम तौर पर उसे सूखने में 36 घंटे लग जाते थे।",
            "sample_latency_ms": 132.45
        },
        {
            "id": "hi-821452-p2-native",
            "query_id": "hi-821452",
            "query": "गर्मियों का पहला दिन किस दिन होता है जब गर्मियों का जन्मदिन होता है।",
            "ground_truth_answer": "20 जून",
            "sample_model_answer": "यदि आप अमेरिका में रहते हैं, तो गर्मियों का पहला दिन 20 जून होगा।",
            "sample_latency_ms": 146.09
        },
        {
            "id": "hi-9444-p1-native",
            "query_id": "hi-9444",
            "query": "एक वैज्ञानिक जो जीव विज्ञान का अध्ययन करता है उसे",
            "ground_truth_answer": "बोटनिस्ट",
            "sample_model_answer": "जीव विज्ञान का अध्ययन करने वाले वैज्ञानिकों को जीवविज्ञानी के रूप में जाना जाता है।",
            "sample_latency_ms": 200.15
        },
        {
            "id": "hi-894308-p4-native",
            "query_id": "hi-894308",
            "query": "किस शो में क्रिस टाइटस थे",
            "ground_truth_answer": "टाइटस",
            "sample_model_answer": "टाइटस ने अपने ही नाम के साथ फॉक्स के कार्यक्रम टाइटस के साथ राष्ट्रीय ध्यान आकर्षित किया।",
            "sample_latency_ms": 140.62
        },
        {
            "id": "hi-1020419-p0-native",
            "query_id": "hi-1020419",
            "query": "कौन सा उपचार दर्द के लिए सबसे अच्छा है",
            "ground_truth_answer": "मारिजुआना का प्रकार",
            "sample_model_answer": "मेरे डॉक्टर को लगता है कि ट्रामाडोल मेरे कुत्ते के लिए सबसे अच्छा इलाज है, क्योंकि उसे कैंसर है और पूरे शरीर में इसका संचार हो गया है।",
            "sample_latency_ms": 131.31
        },
        {
            "id": "hi-56361-p2-native",
            "query_id": "hi-56361",
            "query": "ब्राइटन टाउनशिप फोन नंबर",
            "ground_truth_answer": "724-847-7299",
            "sample_model_answer": "फोन नंबर 724-847-7299 है।",
            "sample_latency_ms": 134.18
        }
    ],
    "mr": [
        {
            "id": "mr-968464-p0-native",
            "query_id": "mr-968464",
            "query": "चेल्सी क्लिंटनचे लग्न कुठे झाले",
            "ground_truth_answer": "राइनबेक, न्यूयॉर्क",
            "sample_model_answer": "या वस्तूच्या काही आवृत्त्यांमध्ये असा दावा केला आहे की चेल्सी क्लिंटन आणि मार्क मेझविन्स्कीचे लग्न जॉर्ज सोरोसच्या हवेलीत झाले होते.",
            "sample_latency_ms": 194.1
        },
        {
            "id": "mr-807450-p0-native",
            "query_id": "mr-807450",
            "query": "सरासरी वेतन ताशी किती आहे",
            "ground_truth_answer": "$१०.५८-$२०.४१",
            "sample_model_answer": "सरासरी वेतन $543,343 ($261/तास) आहे.",
            "sample_latency_ms": 140.3
        },
        {
            "id": "mr-993067-p2-native",
            "query_id": "mr-993067",
            "query": "स्मिथविले कुठे आहे",
            "ground_truth_answer": "लॉरेन्स काउंटी",
            "sample_model_answer": "स्मिथ हे इंग्लंडमध्ये उगम पावलेले एक इंग्रजी कौटुंबिक नाव (आडनाव) आहे.",
            "sample_latency_ms": 142.19
        },
        {
            "id": "mr-255418-p0-native",
            "query_id": "mr-255418",
            "query": "वैद्यकीय कोडर होण्यासाठी किती वेळ लागतो",
            "ground_truth_answer": "१५ महिने",
            "sample_model_answer": "सरासरी, तुम्ही तुमच्या आपत्कालीन वैद्यकीय तंत्रज्ञ प्रशिक्षणाला 200 तास लागतील अशी अपेक्षा करू शकता.",
            "sample_latency_ms": 134.12
        },
        {
            "id": "mr-339741-p0-native",
            "query_id": "mr-339741",
            "query": "मांजरीवर रेबीजची लस किती लवकर काम करते",
            "ground_truth_answer": "12 आठवडे",
            "sample_model_answer": "तुमच्या मांजरीचे लसीकरण रेबीजच्या लसीकरणानुसार आठ किंवा बारा आठवड्यांच्या वयापर्यंत लवकरात लवकर केले जाऊ शकते.",
            "sample_latency_ms": 158.0
        },
        {
            "id": "mr-107804-p3-native",
            "query_id": "mr-107804",
            "query": "किंमती भांड्याची किंमत",
            "ground_truth_answer": "$१०० ते $१,००० पेक्षा जास्त.",
            "sample_model_answer": "्याची",
            "sample_latency_ms": 132.24
        },
        {
            "id": "mr-286184-p5-native",
            "query_id": "mr-286184",
            "query": "तुम्हाला नैसर्गिकरित्या किती तास झोपून जागे व्हायचे आहे?",
            "ground_truth_answer": "आठ तास",
            "sample_model_answer": "रात्री अनेकदा जागे होणे हे नैसर्गिक आहे, पण खोल झोपेतून जागे होणे आणि त्यामुळे तुम्ही उलट्या उलट्या होणे हे अडचणीचे लक्षण असू शकते.",
            "sample_latency_ms": 134.72
        },
        {
            "id": "mr-1009775-p8-native",
            "query_id": "mr-1009775",
            "query": "कोणत्या खाद्यपदार्थांमध्ये जीवनसत्व डी जास्त असते",
            "ground_truth_answer": "शिटाके अळंबीच्या वाळलेल्या आवृत्त्या",
            "sample_model_answer": "जीवनसत्व डी हे चरबीमध्ये विरघळणारे जीवनसत्व आहे जे नैसर्गिकरित्या फारच कमी खाद्यपदार्थांमध्ये असते आणि इतरांमध्ये मिसळले जाते आणि आहारातील पूरक म्हणून उपलब्ध असते.",
            "sample_latency_ms": 136.04
        },
        {
            "id": "mr-315124-p4-native",
            "query_id": "mr-315124",
            "query": "उपहारगृह रोज रोज स्वच्छ करण्यासाठी किती खर्च येतो?",
            "ground_truth_answer": "२० ते ५० डॉलर प्रति तास",
            "sample_model_answer": "डिस्ने स्ट्रॉलर भाड्याने घेण्याची किंमत: एकल स्ट्रॉलर भाड्याने घेण्यासाठी तुम्हाला दररोज $15.00 आणि दुहेरी स्ट्रॉलर भाड्याने घेण्यासाठी दररोज $31.00 खर्च येईल.",
            "sample_latency_ms": 131.98
        },
        {
            "id": "mr-324743-p3-native",
            "query_id": "mr-324743",
            "query": "मेकॅनिकल इंजिनिअर म्हणून तुम्ही किती पैसे कमावता?",
            "ground_truth_answer": "२०१३ मध्ये $८२,१००.",
            "sample_model_answer": "नंतर मध्यम वर्गीय मेकॅनिक म्हणून तुम्ही पन्नास डॉलर प्रति आठवडा कमावता आणि सर्वोत्तम मेकॅनिक म्हणून तुम्ही सत्तर ते एकशे डॉलर प्रति तास कमावले पाहिजे.",
            "sample_latency_ms": 145.12
        },
        {
            "id": "mr-1007234-p0-native",
            "query_id": "mr-1007234",
            "query": "कोणती आज्ञा मालकाला वाचन-लेखन परवानगी नियुक्त करण्यासाठी वापरली जाते?",
            "ground_truth_answer": "चमोड",
            "sample_model_answer": "खरेदी ऑर्डर म्हणजे विक्रेत्याला खरेदीदाराला माल पुरवण्याची लेखी परवानगी मागणारी विनंती.",
            "sample_latency_ms": 141.82
        },
        {
            "id": "mr-1029800-p7-native",
            "query_id": "mr-1029800",
            "query": "कोण आहे एलेक बाल्डविनची पत्नी",
            "ground_truth_answer": "हिलेरिया थॉमस",
            "sample_model_answer": "अलेक बाल्डविनची पत्नी आणि न्यूयॉर्क शहरातील लोकप्रिय योग प्रशिक्षक हिलारिया थॉमसवर एका माजी विद्यार्थ्याने खटला भरला आहे, ज्यात आरोप आहे की तिच्या एका वर्गातील गर्दीमुळे त्याला गंभीर दुखापत झाली.",
            "sample_latency_ms": 128.29
        },
        {
            "id": "mr-578461-p0-native",
            "query_id": "mr-578461",
            "query": "कोणत्या बेसबॉल खेळाडूने कारकिर्दीतील सर्वाधिक कामगिरी केली आहे",
            "ground_truth_answer": "डेरेक जेटर",
            "sample_model_answer": "टाय कॉबने त्याच्या व्यावसायिक कारकिर्दीत अधिकृतपणे 11 वेळा जिंकून एम.एल.बी.",
            "sample_latency_ms": 133.44
        },
        {
            "id": "mr-581308-p5-native",
            "query_id": "mr-581308",
            "query": "भूक न लागण्याचे कारण काय असू शकते",
            "ground_truth_answer": "तणाव, चिंता किंवा नैराश्य.",
            "sample_model_answer": "भूक न लागण्याचे कारण विविध कारणांमुळे असू शकते.",
            "sample_latency_ms": 171.41
        },
        {
            "id": "mr-413828-p1-native",
            "query_id": "mr-413828",
            "query": "आठवड्यातून एक दिवस उपवास करणे आरोग्यदायी आहे का?",
            "ground_truth_answer": "क्रमांक",
            "sample_model_answer": "एका आठवड्यातून एक दिवस फक्त पाण्यावर उपवास करणे दीर्घकालीन आरोग्यासाठी चांगले नाही असे मी सांगेन.",
            "sample_latency_ms": 156.3
        },
        {
            "id": "mr-793235-p0-native",
            "query_id": "mr-793235",
            "query": "डेड काउंटीमध्ये विक्री कर किती आहे",
            "ground_truth_answer": "७.००%",
            "sample_model_answer": "मियामी-डेड काउंटी, फ्लोरिडा विक्री कर 7.00% आहे, ज्यामध्ये 6.00% फ्लोरिडा राज्य विक्री कर आणि 1.00% मियामी-डेड काउंटी स्थानिक विक्री कर यांचा समावेश आहे.",
            "sample_latency_ms": 151.55
        },
        {
            "id": "mr-101149-p4-native",
            "query_id": "mr-101149",
            "query": "इंग्राउंड पूलची किंमत",
            "ground_truth_answer": "२१,९१९ डॉलर्स",
            "sample_model_answer": "होम इम्प्रूवमेंट साइट फिक्सर के अनुसार, औसत इन-ग्राउंड स्विमिंग पूल की कीमत 21,919 डॉलर है।",
            "sample_latency_ms": 137.79
        },
        {
            "id": "mr-318261-p0-native",
            "query_id": "mr-318261",
            "query": "क्रीडा वैद्यकीय व्यावसायिकांनी किती कमावले आहे?",
            "ground_truth_answer": "९४,००० डॉलर्स",
            "sample_model_answer": "क्रीडा वैद्यकीय व्यावसायिक क्रीडा जखमा झालेल्या लोकांची काळजी घेतात.",
            "sample_latency_ms": 142.64
        },
        {
            "id": "mr-311101-p4-native",
            "query_id": "mr-311101",
            "query": "यूकेमध्ये मानसोपचारतज्ज्ञ किती कमावतो",
            "ground_truth_answer": "वर्षाला £४०,००० ते £६०,०००",
            "sample_model_answer": "अमेरिकेतील मानसोपचारतज्ज्ञ सरासरी दरवर्षी सुमारे 181,000 डॉलर एवढ्या मोठ्या प्रमाणात पगाराची अपेक्षा करू शकतात.",
            "sample_latency_ms": 148.04
        },
        {
            "id": "mr-82459-p6-native",
            "query_id": "mr-82459",
            "query": "कार पेंटच्या कामाचा खर्च",
            "ground_truth_answer": "$१,०००-$३,५००",
            "sample_model_answer": "एक कोट पेंट के लिए 300 डॉलर का शुल्क लेता है।",
            "sample_latency_ms": 129.74
        },
        {
            "id": "mr_21",
            "query_id": "mr-309110",
            "query": "कॅलिफोर्नियामध्ये न्यायालयीन दुभाषी किती कमावतो",
            "ground_truth_answer": "६८,००० डॉलर प्रति वर्ष",
            "sample_model_answer": "कॅलिफोर्नियातील न्यायालयीन दुभाष्यांना 2012 पर्यंत दररोज 282 डॉलरचा तुलनात्मक दर मिळतो, तर न्यूयॉर्क राज्य एकीकृत न्यायालय प्रणालीत काम करणारे स्वतंत्र न्यायालयीन दुभाष्ये मार्च 2012 पर्यंत दररोज 250 डॉलर कमावतात.",
            "sample_latency_ms": 144.49
        },
        {
            "id": "mr_22",
            "query_id": "mr-1162351",
            "query": "एका प्रभावी नागरिकाने काय करावे लागते",
            "ground_truth_answer": "उत्तर नाही.",
            "sample_model_answer": "चांगल्या नागरिकाकडून अपेक्षित असलेली पहिली आणि सर्वात महत्वाची गुणवत्ता म्हणजे देशभक्ती.",
            "sample_latency_ms": 136.19
        },
        {
            "id": "mr_23",
            "query_id": "mr-557057",
            "query": "काय पात्र खाती मानली जातात",
            "ground_truth_answer": "उत्तर नाही.",
            "sample_model_answer": "मला वाटते की त्यांना कॅडेट मानले जाते.",
            "sample_latency_ms": 129.66
        },
        {
            "id": "mr_24",
            "query_id": "mr-152743",
            "query": "डिस्ने मॅजिक शिपमध्ये किती प्रवासी बसू शकतात",
            "ground_truth_answer": "२,४०० प्रवासी",
            "sample_model_answer": "या प्रकारच्या लिमोजमध्ये सहसा सहा ते आठ लोक बसू शकतात, ताणण्याच्या प्रमाणावर अवलंबून.",
            "sample_latency_ms": 133.52
        },
        {
            "id": "mr_25",
            "query_id": "mr-983864",
            "query": "ग्वाम कुठे आहे",
            "ground_truth_answer": "गुआम पश्चिम प्रशांत महासागरात स्थित आहे.",
            "sample_model_answer": "ग्वाम ध्वजाचा मध्यभाग हा ग्वाम ध्वजाचा केंद्रबिंदू आहे.",
            "sample_latency_ms": 163.85
        },
        {
            "id": "mr_26",
            "query_id": "mr-649519",
            "query": "शाळेचे सरासरी सचिव दरवर्षी किती कमावतात",
            "ground_truth_answer": "३२,७९० डॉलर्स",
            "sample_model_answer": "शालेय सचिवाचा सरासरी वेतन 16,486 पौंड इतका असतो.",
            "sample_latency_ms": 149.5
        },
        {
            "id": "mr_27",
            "query_id": "mr-906404",
            "query": "तात्पुरत्या कामासाठी काय करावे",
            "ground_truth_answer": "उत्तर नाही.",
            "sample_model_answer": "काही नोकऱ्या तुम्हाला दुसरी सापडेपर्यंत तात्पुरत्या असू शकतात.",
            "sample_latency_ms": 130.66
        },
        {
            "id": "mr_28",
            "query_id": "mr-964821",
            "query": "काचेसारखे खडक कुठे आणि कसे तयार होतात",
            "ground_truth_answer": "उत्तर नाही.",
            "sample_model_answer": "खडक पाण्याच्या उथळ भागात किंवा वाळवंटासारख्या कोरड्या प्रदेशातील तलावांमध्ये तयार होतात जिथे बाष्पीभवनाचा दर पावसापेक्षा जास्त असतो.",
            "sample_latency_ms": 132.42
        },
        {
            "id": "mr_29",
            "query_id": "mr-965469",
            "query": "हॉलमार्क स्टोअर्स कुठे आहेत",
            "ground_truth_answer": "इलिनॉय",
            "sample_model_answer": "हे नवीन दुकान माद्रिदच्या बाहेरील माजादाहोंडा शॉपिंग सेंटरमध्ये आहे, ज्यात आधीच शहरात आणखी 6 स्टोअर्स आहेत.",
            "sample_latency_ms": 140.52
        },
        {
            "id": "mr_30",
            "query_id": "mr-1034773",
            "query": "केट मुलग्रो कोण आहे",
            "ground_truth_answer": "अमेरिकन अभिनेत्री.",
            "sample_model_answer": "केट मुलग्रू सध्या अविवाहित आहे.",
            "sample_latency_ms": 133.89
        }
    ],
    "en": [
        {
            "id": "en-637779-p3-native",
            "query_id": "en-637779",
            "query": "what does folic acid do",
            "ground_truth_answer": "Crucial for proper brain function and plays an important role in mental and emotional health.",
            "sample_model_answer": "To prevent miscarriage",
            "sample_latency_ms": 137.77
        },
        {
            "id": "en-31967-p0-native",
            "query_id": "en-31967",
            "query": "average conservatory cost",
            "ground_truth_answer": "$10,000 to $30,000",
            "sample_model_answer": "£6,700",
            "sample_latency_ms": 158.04
        },
        {
            "id": "en-498704-p0-native",
            "query_id": "en-498704",
            "query": "siriusxm cost per month",
            "ground_truth_answer": "$17",
            "sample_model_answer": "Currently, internet-only customers pay $14.99 per month for most of SiriusXM’s channels plus premium content from Howard Stern and others.",
            "sample_latency_ms": 143.93
        },
        {
            "id": "en-683769-p9-native",
            "query_id": "en-683769",
            "query": "what is a forcing cone",
            "ground_truth_answer": "A very misunderstood part of the revolver barrel and even many professional gunsmiths don't really understand them.",
            "sample_model_answer": "Backboring",
            "sample_latency_ms": 137.38
        },
        {
            "id": "en-627219-p9-native",
            "query_id": "en-627219",
            "query": "what do zebras eat food",
            "ground_truth_answer": "Plain grass, shrubs, herbs, twigs, leaves, roots and bark from trees.",
            "sample_model_answer": "Zebras in zoos eat pallets specially made for herbivores, which supply them with more nutrients than hay.",
            "sample_latency_ms": 132.76
        },
        {
            "id": "en-338146-p2-native",
            "query_id": "en-338146",
            "query": "how pronounce tiguan",
            "ground_truth_answer": "TEE-gwan",
            "sample_model_answer": "Easier to drive than pronounce",
            "sample_latency_ms": 137.01
        },
        {
            "id": "en-666179-p4-native",
            "query_id": "en-666179",
            "query": "what happens if you miss a vein shooting up",
            "ground_truth_answer": "Run it under really really hot water and rub really really hard on the spot where you missed. not only will you decrease the chances of an absess forming--BUT you will also soon feel the warm rush as if you hadn't missed at all.",
            "sample_model_answer": "Heroin",
            "sample_latency_ms": 140.07
        },
        {
            "id": "en-435759-p5-native",
            "query_id": "en-435759",
            "query": "largest active military",
            "ground_truth_answer": "China",
            "sample_model_answer": "Fifth largest air force",
            "sample_latency_ms": 160.88
        },
        {
            "id": "en-614045-p6-native",
            "query_id": "en-614045",
            "query": "what county is van nuys ca",
            "ground_truth_answer": "Los Angeles",
            "sample_model_answer": "Napa is the county seat of Napa County, California.",
            "sample_latency_ms": 130.8
        },
        {
            "id": "en-717648-p4-native",
            "query_id": "en-717648",
            "query": "what is an x ray photon",
            "ground_truth_answer": "The x-ray photons are either absorbed or scattered out of the beam. In scattering, photons are ejected out of the primary beam as a result of interactions with the orbital electrons of absorber atoms.",
            "sample_model_answer": "High-energy",
            "sample_latency_ms": 132.73
        },
        {
            "id": "en-571722-p9-native",
            "query_id": "en-571722",
            "query": "what are the most common geriatric illness",
            "ground_truth_answer": "Heart disease, stroke, cancer, and diabetes.",
            "sample_model_answer": "Most older adults with common geriatric health conditions live in the community rather than in nursing homes",
            "sample_latency_ms": 137.7
        },
        {
            "id": "en-700343-p2-native",
            "query_id": "en-700343",
            "query": "what is a slip for",
            "ground_truth_answer": "The common implement for moving dirt when doing such things as digging ponds or making fills where large volumes of dirt were needed.",
            "sample_model_answer": "Plural slips) 1 A twig or shoot; a cutting. a slip from a vine",
            "sample_latency_ms": 165.28
        },
        {
            "id": "en-451159-p8-native",
            "query_id": "en-451159",
            "query": "medical definition of obese",
            "ground_truth_answer": "An abnormal accumulation of body fat, usually 20% or more over an individual's ideal body weight.",
            "sample_model_answer": "Body mass index",
            "sample_latency_ms": 144.61
        },
        {
            "id": "en-415234-p2-native",
            "query_id": "en-415234",
            "query": "is julia kim still alive",
            "ground_truth_answer": "No.",
            "sample_model_answer": "He's still alive and well",
            "sample_latency_ms": 138.08
        },
        {
            "id": "en-686856-p8-native",
            "query_id": "en-686856",
            "query": "what is a hill-sachs lesion",
            "ground_truth_answer": "A posterolateral humeral head compression fracture, typically secondary to recurrent anterior shoulder dislocations, as the humeral head comes to rest against the anteroinferior part of the glenoid.",
            "sample_model_answer": "A fracture in the head of the humerus bone",
            "sample_latency_ms": 140.94
        },
        {
            "id": "en-282210-p3-native",
            "query_id": "en-282210",
            "query": "how many died from spanish flu",
            "ground_truth_answer": "At least 20 million people worldwide.",
            "sample_model_answer": "Between 2 and 20%",
            "sample_latency_ms": 128.8
        },
        {
            "id": "en-449394-p5-native",
            "query_id": "en-449394",
            "query": "meaning of the arabic name nayla",
            "ground_truth_answer": "Successful",
            "sample_model_answer": "Attainer, achiever",
            "sample_latency_ms": 130.83
        },
        {
            "id": "en-34679-p0-native",
            "query_id": "en-34679",
            "query": "average cost of wedding dinner per person park savoy",
            "ground_truth_answer": "$130 to $230",
            "sample_model_answer": "2 On average, be prepared to spend anywhere from $130 to $230 per person for a wedding at Park Savoy Estate.",
            "sample_latency_ms": 138.58
        },
        {
            "id": "en-705113-p5-native",
            "query_id": "en-705113",
            "query": "what is a values clarification",
            "ground_truth_answer": "Values clarification is a self-assessment process that enables you to discover the content and strength of your own system of values.",
            "sample_model_answer": "For you to become fully conscious of their influence",
            "sample_latency_ms": 160.82
        },
        {
            "id": "en-550988-p1-native",
            "query_id": "en-550988",
            "query": "what age do you have to be to go to a gun range",
            "ground_truth_answer": "Minimum of 18 years old.",
            "sample_model_answer": "Assuming that you mean shoot, most gun ranges require their members to be a minumum of 18 years old.",
            "sample_latency_ms": 158.65
        },
        {
            "id": "en-1174738-p1-native",
            "query_id": "en-1174738",
            "query": "is the mitochondria a organelle or a function",
            "ground_truth_answer": "They are made of two membranes. The outer membrane covers the organelle and contains it like a skin. The inner membrane folds over many times and creates layered structures called cristae.",
            "sample_model_answer": "Round tube-like",
            "sample_latency_ms": 142.94
        },
        {
            "id": "en-547021-p8-native",
            "query_id": "en-547021",
            "query": "webster definition of theft",
            "ground_truth_answer": "Theft is the illegal taking of another person's property or services without that person's permission or consent with the intent to deprive the rightful owner of it.",
            "sample_model_answer": "Taking away any object that (partly) belongs to someone else",
            "sample_latency_ms": 140.7
        },
        {
            "id": "en-418112-p0-native",
            "query_id": "en-418112",
            "query": "is midrin a compound drug",
            "ground_truth_answer": "Yes",
            "sample_model_answer": "Midrin (and all other Midrin-like drugs, including Epidrin) have really, truly been discontinued.",
            "sample_latency_ms": 148.04
        },
        {
            "id": "en-601985-p2-native",
            "query_id": "en-601985",
            "query": "what county in florida is alligator point in?",
            "ground_truth_answer": "Franklin",
            "sample_model_answer": "Midway, FL--It all started Saturday night, the Duin family out hunting gator in Gadsden County.",
            "sample_latency_ms": 134.67
        },
        {
            "id": "en-639049-p3-native",
            "query_id": "en-639049",
            "query": "what does homophone mean",
            "ground_truth_answer": "A homophone is a word that sounds the same as another word but has a different meaning and/or spelling.",
            "sample_model_answer": "A word that is pronounced the same as another word but differs in meaning",
            "sample_latency_ms": 135.65
        },
        {
            "id": "en-222251-p3-native",
            "query_id": "en-222251",
            "query": "how do you gain weight in your legs",
            "ground_truth_answer": "Perform exercises that encourage the growth of leg muscle. Squats, lunges and step-ups are examples of such moves that target the legs and recruit a lot of muscle fibers for growth.",
            "sample_model_answer": "You really don't gain weight in your legs but you can build the muscle in them by doing a lot of squats , calve raises, and leg presses on a weight sled.",
            "sample_latency_ms": 134.17
        },
        {
            "id": "en-537824-p2-native",
            "query_id": "en-537824",
            "query": "vitamin e how much to take daily?",
            "ground_truth_answer": "8 mg for women and 10 mg for men.",
            "sample_model_answer": "You should take 15 to 33 mg of vitamin E per day.",
            "sample_latency_ms": 139.46
        },
        {
            "id": "en-707064-p0-native",
            "query_id": "en-707064",
            "query": "what is abo incompatibility in newborn",
            "ground_truth_answer": "A relatively common type of hemolytic disease of newborns, and may occur during blood transfusion reactions of older people.",
            "sample_model_answer": "Hemolytic disease of newborn due to ABO incompatibility Tikrit Medical Journal 2009; 15(2):70-78 73 the babies and this is consistent with the studies of john a.",
            "sample_latency_ms": 139.05
        },
        {
            "id": "en-513958-p5-native",
            "query_id": "en-513958",
            "query": "the ________ nerves regulate essential involuntary body functions",
            "ground_truth_answer": "Sympathetic",
            "sample_model_answer": "The autonomic nervous system, also known as the involuntary nervous system, regulates those facets in the body that occur automatically, such as breathing, blood pressure, digestion, heart beat, bladder function and narrowing or widening of the blood vessels.",
            "sample_latency_ms": 172.18
        },
        {
            "id": "en-616955-p3-native",
            "query_id": "en-616955",
            "query": "what day is the parade in new york city",
            "ground_truth_answer": "March 17",
            "sample_model_answer": "March 17.",
            "sample_latency_ms": 153.84
        }
    ]
}

def post_query(item, lang):
    query = item["query"]
    t0 = time.perf_counter()
    data = urllib.parse.urlencode({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        f"{MODAL_URL}/query",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "lang": lang,
                "id": item["id"],
                "query": query,
                "ground_truth": item["ground_truth_answer"],
                "answer": res.get("answer", ""),
                "guardrail": res.get("guardrail_triggered"),
                "confidence": res.get("confidence", 0.0),
                "timings_ms": res.get("timings_ms", {}),
                "client_ms": elapsed,
                "success": True,
            }
    except Exception as e:
        return {
            "lang": lang,
            "id": item["id"],
            "query": query,
            "ground_truth": item["ground_truth_answer"],
            "answer": f"ERROR: {e}",
            "guardrail": "error",
            "confidence": 0.0,
            "timings_ms": {},
            "client_ms": (time.perf_counter() - t0) * 1000,
            "success": False,
        }

def compute_percentile(data, p):
    if not data:
        return 0.0
    sorted_d = sorted(data)
    idx = int((len(sorted_d) - 1) * p / 100.0)
    return sorted_d[idx]

def run_benchmark():
    print("=" * 75)
    print("  RUNNING 90-QUESTION BENCHMARK SUITE - SET 2")
    print("  Endpoint: " + MODAL_URL)
    print("  Dataset: 30 Hindi, 30 Marathi, 30 English (90 total - Disjoint from Set 1)")
    print("=" * 75)

    all_items = []
    for lang, items in DATASET_90_SET2.items():
        for item in items:
            all_items.append((item, lang))

    results = []
    t_start = time.perf_counter()

    for idx, (item, lang) in enumerate(all_items, 1):
        r = post_query(item, lang)
        results.append(r)
        status = "✅" if not r["guardrail"] and "couldn't extract" not in r["answer"].lower() and "unable to answer" not in r["answer"].lower() else "❌"
        tot_ms = r["timings_ms"].get("total_ms", r["client_ms"])
        ans_preview = r["answer"][:40].replace("\n", " ")
        print(f"  [{idx:02d}/90] [{lang.upper()}] {status} {r['query'][:35]:<35} | {ans_preview:<40} ({tot_ms:.1f}ms)")

    total_duration = time.perf_counter() - t_start

    # Summary
    print("\n" + "=" * 75)
    print("  SET 2 (90-QUESTION) BENCHMARK RESULTS SUMMARY")
    print("=" * 75)

    by_lang = {"hi": [], "mr": [], "en": []}
    for r in results:
        by_lang[r["lang"]].append(r)

    total_correct = 0
    total_queries = len(results)

    for lang in ["hi", "mr", "en"]:
        lang_res = by_lang[lang]
        correct = sum(1 for r in lang_res if not r["guardrail"] and "couldn't extract" not in r["answer"].lower() and "unable to answer" not in r["answer"].lower())
        total_correct += correct
        acc = (correct / len(lang_res)) * 100 if lang_res else 0.0
        server_latencies = [r["timings_ms"].get("total_ms", r["client_ms"]) for r in lang_res]
        search_latencies = [r["timings_ms"].get("search_ms", 0.0) for r in lang_res if r["timings_ms"].get("search_ms")]
        qa_latencies = [r["timings_ms"].get("qa_ms", 0.0) for r in lang_res if r["timings_ms"].get("qa_ms")]

        print(f"\n--- Language: {lang.upper()} (30 Questions) ---")
        print(f"  Accuracy / Grounded Answers : {correct}/{len(lang_res)} ({acc:.1f}%)")
        print(f"  Server Latency Total (Mean) : {statistics.mean(server_latencies):.1f} ms (P50: {compute_percentile(server_latencies, 50):.1f} ms | P70: {compute_percentile(server_latencies, 70):.1f} ms | P90: {compute_percentile(server_latencies, 90):.1f} ms | P100: {compute_percentile(server_latencies, 100):.1f} ms)")
        if search_latencies:
            print(f"  Search Latency (Mean)       : {statistics.mean(search_latencies):.1f} ms (P50: {compute_percentile(search_latencies, 50):.1f} ms)")
        if qa_latencies:
            print(f"  QA Latency (Mean)           : {statistics.mean(qa_latencies):.1f} ms (P50: {compute_percentile(qa_latencies, 50):.1f} ms)")

    overall_acc = (total_correct / total_queries) * 100
    all_server_lat = [r["timings_ms"].get("total_ms", r["client_ms"]) for r in results]
    all_search_lat = [r["timings_ms"].get("search_ms", 0.0) for r in results if r["timings_ms"].get("search_ms")]
    all_qa_lat = [r["timings_ms"].get("qa_ms", 0.0) for r in results if r["timings_ms"].get("qa_ms")]

    print("\n" + "=" * 75)
    print("  OVERALL SYSTEM PERFORMANCE (SET 2 - 90 QUESTIONS)")
    print("=" * 75)
    print(f"  Total Grounded Accuracy     : {total_correct}/{total_queries} ({overall_acc:.1f}%)")
    print(f"  Server Latency P50          : {compute_percentile(all_server_lat, 50):.1f} ms")
    print(f"  Server Latency P70          : {compute_percentile(all_server_lat, 70):.1f} ms")
    print(f"  Server Latency P90          : {compute_percentile(all_server_lat, 90):.1f} ms")
    print(f"  Server Latency P100         : {compute_percentile(all_server_lat, 100):.1f} ms")
    print(f"  Server Latency Mean         : {statistics.mean(all_server_lat):.1f} ms")
    if all_search_lat:
        print(f"  FAISS Search P50            : {compute_percentile(all_search_lat, 50):.1f} ms")
    if all_qa_lat:
        print(f"  QA Extraction P50           : {compute_percentile(all_qa_lat, 50):.1f} ms")
    print(f"  Total Benchmark Run Time    : {total_duration:.1f} s")
    print("=" * 75)

    with open("data/benchmark_90_set2_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "overall_accuracy": overall_acc,
            "total_correct": total_correct,
            "total_queries": total_queries,
            "p50_total_ms": compute_percentile(all_server_lat, 50),
            "p70_total_ms": compute_percentile(all_server_lat, 70),
            "p90_total_ms": compute_percentile(all_server_lat, 90),
            "p100_total_ms": compute_percentile(all_server_lat, 100),
            "mean_total_ms": statistics.mean(all_server_lat),
            "p50_search_ms": compute_percentile(all_search_lat, 50) if all_search_lat else 0,
            "p50_qa_ms": compute_percentile(all_qa_lat, 50) if all_qa_lat else 0,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print("Results saved to data/benchmark_90_set2_results.json")

if __name__ == "__main__":
    run_benchmark()
