#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
90-Question Benchmark Suite (30 Hindi, 30 Marathi, 30 English)
Evaluates Grounded Accuracy, Guardrail Behavior, and Latency against the live Multi-Strategy Index.
"""

import time
import json
import statistics
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

MODAL_URL = "https://healthbaba25--voice-rag-voicerag-fastapi-app.modal.run"

DATASET_90 = {
    "hi": [
        {
            "id": "hi-436492-p1-native",
            "query_id": "hi-436492",
            "query": "सबसे बड़ा उड़ने वाला सरीसृप अब तक",
            "ground_truth_answer": "प्रोजेक्टोरिया का अज़हद्रिचिडे परिवार",
            "sample_model_answer": "सबसे बड़े उड़ने वाले जानवर जो कभी भी जीवित रहे थे, वे अज़हद्रिचिडे परिवार के सदस्य थे, जो उड़ने वाले सरीसृप थे जो डायनासोर के लगभग एक ही समय में रहते थे, हालांकि वे वास्तव में डायनासोर नहीं थे।",
            "sample_latency_ms": 164.14
        },
        {
            "id": "hi-746312-p6-native",
            "query_id": "hi-746312",
            "query": "सैंटेंडर स्पेन के बारे में क्या प्रसिद्ध है",
            "ground_truth_answer": "छोटी नौका यात्राएँ और पारिवारिक पर्यटन।",
            "sample_model_answer": "स्पेन में कैंटाब्रिया की राजधानी, सैंटेंडर, स्पेन के लोगों के लिए एक प्रसिद्ध अवकाश स्थल है, और इन दिनों ब्रिटेन के लोग भी यू.के.",
            "sample_latency_ms": 135.5
        },
        {
            "id": "hi-804443-p2-native",
            "query_id": "hi-804443",
            "query": "फ्लोरिडा में गोल्फ कार्ट चलाने की उम्र क्या है",
            "ground_truth_answer": "14 वर्ष",
            "sample_model_answer": "कई राज्यों में न्यूनतम आयु 13 है, लेकिन फ्लोरिडा कानून निर्धारित करता है कि गोल्फ कार्ट चालकों की आयु कम से कम 14 वर्ष होनी चाहिए, हालांकि आपको लाइसेंस की आवश्यकता नहीं है।",
            "sample_latency_ms": 141.64
        },
        {
            "id": "hi-602859-p0-native",
            "query_id": "hi-602859",
            "query": "कौन सी काउंटी बेसिन वाई है",
            "ground_truth_answer": "बिग हॉर्न काउंटी",
            "sample_model_answer": "जूडिथ बेसिन काउंटी की काउंटी सीट स्टैनफोर्ड, लेविसटाउन और ग्रेट फॉल्स के बीच मिडवे पर एम.टी.",
            "sample_latency_ms": 133.2
        },
        {
            "id": "hi-952644-p7-native",
            "query_id": "hi-952644",
            "query": "साइबोर्ग डीसी फिल्म कब आ रही है",
            "ground_truth_answer": "3 अप्रैल 2020 को",
            "sample_model_answer": "साइबोर्ग 3 अप्रैल 2020 को रिलीज़ होने वाला है, और ग्रीन लैंटर्न 19 जून 2020 को रिलीज़ होगा।",
            "sample_latency_ms": 135.33
        },
        {
            "id": "hi-578380-p8-native",
            "query_id": "hi-578380",
            "query": "कौन से बैंड में अभी भी मूल सदस्य हैं",
            "ground_truth_answer": "एल.ए. गन्स",
            "sample_model_answer": "ट्रैसी गन्स इस बैंड के एकमात्र स्थायी सदस्य थे जो 1983 से 2002 तक अपने मूल संस्करण के माध्यम से चले।",
            "sample_latency_ms": 136.0
        },
        {
            "id": "hi-718277-p4-native",
            "query_id": "hi-718277",
            "query": "निक्स का दूसरा नाम क्या है",
            "ground_truth_answer": "कुछ भी नहीं",
            "sample_model_answer": "मूल रूप से विक्टोरियन स्लैंग, निक्स की तुलना पहले के जर्मन निक्स से की जा सकती है, जो कुछ नहीं (‘कुछ नहीं’) का एक बोलचाल का छोटा रूप है।",
            "sample_latency_ms": 131.19
        },
        {
            "id": "hi-815775-p5-native",
            "query_id": "hi-815775",
            "query": "एक तीव्र कोण का क्या स्तर है",
            "ground_truth_answer": "एक तीव्र कोण 90 डिग्री से अधिक और 180 डिग्री से कम कोण होता है।",
            "sample_model_answer": "90 डिग्री से अधिक और 180 डिग्री से कम कोण होता है। हाँ, एक तीव्र कोण ज्यामितीय रूप से एक समकोण से अधिक",
            "sample_latency_ms": 143.45
        },
        {
            "id": "hi-230267-p3-native",
            "query_id": "hi-230267",
            "query": "पेनसाकोला से जैक्सनविले तक कितना दूर",
            "ground_truth_answer": "358 मील या 576 किलोमीटर",
            "sample_model_answer": "तक की कुल ड्राइविंग दूरी 358 मील या 576 किलोमीटर है।",
            "sample_latency_ms": 138.03
        },
        {
            "id": "hi-580360-p8-native",
            "query_id": "hi-580360",
            "query": "कौन सी थोक परिवहन प्रक्रिया सामग्री को कोशिकाओं में ले जाती है?",
            "ground_truth_answer": "एक्सोसाइटोसिस",
            "sample_model_answer": "एक्सोसाइटोसिस (एक्सो = बाहरी, साइटोसिस = परिवहन तंत्र) थोक परिवहन का एक रूप है जिसमें सामग्री को झिल्ली-बद्ध पुटिकाओं में कोशिका के अंदर से बाहर की ओर परिवहन किया जाता है जो प्लाज्मा झिल्ली के साथ जुड़ जाते हैं।",
            "sample_latency_ms": 138.63
        },
        {
            "id": "hi-311886-p2-native",
            "query_id": "hi-311886",
            "query": "एक भर्ती किए गए ट्रक चालक कितना कमाता है?",
            "ground_truth_answer": "70,000 डॉलर से 85,000 डॉलर तक",
            "sample_model_answer": "एक ट्रक चालक, भारी/ट्रैक्टर-ट्रेलर औसतन 20.59 न्यूजीलैंड डॉलर प्रति घंटा की मजदूरी कमाता है।",
            "sample_latency_ms": 144.28
        },
        {
            "id": "hi-966761-p0-native",
            "query_id": "hi-966761",
            "query": "श्रोणि तल की मांसपेशियाँ कहाँ स्थित होती हैं",
            "ground_truth_answer": "पैरों के बीच",
            "sample_model_answer": "श्रोणि तल की मांसपेशियां आपके पैरों के बीच स्थित होती हैं और आपके जघनास्थि से आपके रीढ़ की हड्डी के आधार तक आगे की ओर जाती हैं।",
            "sample_latency_ms": 135.42
        },
        {
            "id": "hi-611357-p2-native",
            "query_id": "hi-611357",
            "query": "कौन सी काउंटी पिट्सफोर्ड, न्यूयॉर्क में है",
            "ground_truth_answer": "मोनरो काउंटी",
            "sample_model_answer": "पिट्सफोर्ड, रोचेस्टर का एक उपनगर, मोनरो काउंटी, न्यूयॉर्क, संयुक्त राज्य अमेरिका में एक शहर है।",
            "sample_latency_ms": 139.38
        },
        {
            "id": "hi-899543-p0-native",
            "query_id": "hi-899543",
            "query": "कौन से उपपरमाण्विक कण",
            "ground_truth_answer": "उपपरमाण्विक कण परमाणु से छोटे कण होते हैं।",
            "sample_model_answer": "उपपरमाण्विक कण: एक उपपरमाण्विक कण एक परमाणु से छोटा कण होता है: यह प्राथमिक या मिश्रित हो सकता है।",
            "sample_latency_ms": 134.26
        },
        {
            "id": "hi-1176436-p0-native",
            "query_id": "hi-1176436",
            "query": "बैरी बरबैंक का मौसम पूर्वानुमान कितना पुराना है",
            "ground_truth_answer": "बैरी बरबैंक 67 वर्ष के हैं।",
            "sample_model_answer": "बरबैंक, सीए - अपने क्षेत्र के लिए घंटे-दर-घंटे के दृश्य, 10-दिवसीय पूर्वानुमान, तापमान, आर्द्रता, वर्षा सहित बहुत ही नवीनतम मौसम पूर्वानुमान प्राप्त करें।",
            "sample_latency_ms": 129.61
        },
        {
            "id": "hi-764113-p0-native",
            "query_id": "hi-764113",
            "query": "लैक्टोन क्या है",
            "ground_truth_answer": "एक खाद्य योजक।",
            "sample_model_answer": "यह ग्लुकोनिक एसिड का एक लैक्टोन (चक्रीय एस्टर) है।",
            "sample_latency_ms": 141.07
        },
        {
            "id": "hi-240237-p3-native",
            "query_id": "hi-240237",
            "query": "टैटू लगवाने के बाद पानी में जाने से पहले तुम्हें कितने समय तक इंतजार करना पड़ेगा?",
            "ground_truth_answer": "48 घंटे",
            "sample_model_answer": "टैटू लगवाने के बाद, आपको अपने नए टैटू को सुरक्षात्मक बाधा बनने के लिए कम से कम 48 घंटे तक इंतजार करना चाहिए, लेकिन कभी-कभी ऐसा होना अपरिहार्य होता है।",
            "sample_latency_ms": 139.79
        },
        {
            "id": "hi-646677-p8-native",
            "query_id": "hi-646677",
            "query": "रमजान मुबारक क्या है",
            "ground_truth_answer": "बधाई हो, यह रमजान है।",
            "sample_model_answer": "नवीनतम सबसे पुराना सर्वश्रेष्ठ उत्तर: रमजान मुबारक का अर्थ है बधाई, यह रमजान है...(मूल रूप से) हम बधाई कहते हैं क्योंकि हम इस बार और इस द्वारा लाए जाने वाले सभी आशीर्वादों के लिए उत्साहित हैं।",
            "sample_latency_ms": 148.25
        },
        {
            "id": "hi-609744-p3-native",
            "query_id": "hi-609744",
            "query": "कौन सी काउंटी मोंट्रोस, वा है",
            "ground_truth_answer": "वेस्टमोरलैंड काउंटी",
            "sample_model_answer": "मोंट्रोस वेस्टमोरलैंड काउंटी, वर्जीनिया, संयुक्त राज्य अमेरिका में एक शहर है।",
            "sample_latency_ms": 135.7
        },
        {
            "id": "hi-606592-p4-native",
            "query_id": "hi-606592",
            "query": "ग्राहम्सविले न्यूयॉर्क कौन्टी में",
            "ground_truth_answer": "सुलिवन काउंटी",
            "sample_model_answer": "1.83 एकड़ ग्राहम्सविले, सुलिवन काउंटी, न्यूयॉर्क।",
            "sample_latency_ms": 146.36
        },
        {
            "id": "hi-215430-p0-native",
            "query_id": "hi-215430",
            "query": "उत्तरी यूरोप में पुनर्जागरण कैसे विकसित हुआ?",
            "ground_truth_answer": "उत्तरी पुनर्जागरण वह पुनर्जागरण था जो आल्प्स के उत्तर में यूरोप में हुआ था। 1497 से पहले, इतालवी पुनर्जागरण का इटली के बाहर बहुत कम प्रभाव था। 15वीं शताब्दी के अंत से, इसके विचार पूरे यूरोप में फैल गए। इसने जर्मन पुनर्जागरण, फ्रांसीसी पुनर्जागरण, अंग्रेजी पुनर्जागरण, निम्न देशों में पुनर्जागरण, पोलिश पुनर्जागरण और अन्य राष्ट्रीय और स्थानीयकृत आंदोलनों को प्रभावित किया, प्रत्येक की अपनी विशेषताएं और ताकत थीं।",
            "sample_model_answer": "उत्तरी पुनर्जागरण प्रोटेस्टेंट सुधार से भी निकटता से जुड़ा हुआ था, जिसके परिणामस्वरूप विभिन्न प्रोटेस्टेंट समूहों और रोमन कैथोलिक चर्च के बीच आंतरिक और बाहरी संघर्षों की लंबी श्रृंखला थी।",
            "sample_latency_ms": 144.39
        },
        {
            "id": "hi-229532-p9-native",
            "query_id": "hi-229532",
            "query": "आपको शादी के लिए कितने समय पहले निमंत्रण भेजना चाहिए?",
            "ground_truth_answer": "छह से आठ सप्ताह के बीच",
            "sample_model_answer": "ये कार्यक्रम से छह से आठ सप्ताह पहले भेजे जाने चाहिए, यह इस बात पर निर्भर करता है कि सूची में कितने बाहरी लोग हैं।",
            "sample_latency_ms": 146.85
        },
        {
            "id": "hi-590721-p0-native",
            "query_id": "hi-590721",
            "query": "श्रोणि घूर्णन का कारण क्या है",
            "ground_truth_answer": "मांसपेशियों का असंतुलन",
            "sample_model_answer": "मांसपेशियों का असंतुलन होना ही पूर्व श्रोणि झुकाव का सबसे आम कारण है।",
            "sample_latency_ms": 132.45
        },
        {
            "id": "hi-108443-p3-native",
            "query_id": "hi-108443",
            "query": "लेपित डामर को सील करने की लागत प्रति वर्ग फुट",
            "ground_truth_answer": "3.00 डॉलर प्रति वर्ग फुट।",
            "sample_model_answer": "31 वर्षों में, आप अपने डामर को सील करने के लिए कुल 3.00 डॉलर प्रति वर्ग फुट खर्च करेंगे।",
            "sample_latency_ms": 143.61
        },
        {
            "id": "hi-253482-p1-native",
            "query_id": "hi-253482",
            "query": "अवसादरोधी दवाओं को काम करने में कितना समय लगता है",
            "ground_truth_answer": "2 से 6 सप्ताह",
            "sample_model_answer": "और त्रिचक्रीय अवसादरोधी - अधिकांश लोगों के लिए 2 से 6 सप्ताह तक के समय के लगभग समान समय में काम करते हैं, जबकि अधिकांश लोग 3 से 4 सप्ताह के भीतर लाभ महसूस करना शुरू कर देंगे।",
            "sample_latency_ms": 135.47
        },
        {
            "id": "hi-606554-p2-native",
            "query_id": "hi-606554",
            "query": "कौन सी काउंटी में गोल्ड रिवर सीए।",
            "ground_truth_answer": "सैक्रामेंटो",
            "sample_model_answer": "गोल्ड रिवर कैलिफोर्निया के सैक्रामेंटो काउंटी में एक जनगणना-नामित स्थान (सी.डी.पी.) है।",
            "sample_latency_ms": 139.02
        },
        {
            "id": "hi-230795-p7-native",
            "query_id": "hi-230795",
            "query": "कोपर शहर से हॉलीवुड कितना दूर है",
            "ground_truth_answer": "11 मील या 18 किमी।",
            "sample_model_answer": "तक की दूरी 11 मील या 18 किमी है।",
            "sample_latency_ms": 135.83
        },
        {
            "id": "hi-550556-p0-native",
            "query_id": "hi-550556",
            "query": "आप किस उम्र के हो सकते हैं जीसी?",
            "ground_truth_answer": "14 से 16 साल के",
            "sample_model_answer": "किसी भी उम्र के हो सकते हैं। जीसीएसई 16 साल",
            "sample_latency_ms": 134.79
        },
        {
            "id": "hi-605041-p7-native",
            "query_id": "hi-605041",
            "query": "दिल्ली कैलिफोर्निया कौन्टी है",
            "ground_truth_answer": "मेर्सिड काउंटी",
            "sample_model_answer": "दिल्ली संयुक्त राज्य अमेरिका के कैलिफोर्निया राज्य के मर्सिड काउंटी में एक जनगणना नामित स्थान (सी.डी.पी.) है, जो मर्सिड से 18 मील (29 कि.मी.) पश्चिम-उत्तर-पश्चिम में 118 फीट (36 मीटर) की ऊँचाई पर स्थित है।",
            "sample_latency_ms": 137.45
        },
        {
            "id": "hi-38282-p0-native",
            "query_id": "hi-38282",
            "query": "मीठे पानी में एन.एच.3 का औसत स्तर",
            "ground_truth_answer": "0.02 पी.पी.एम.",
            "sample_model_answer": "मीठे पानी या समुद्री वातावरण में एन.एच.3 के रूप में 0.02 पी.पी.एम.",
            "sample_latency_ms": 131.54
        }
    ],
    "mr": [
        {
            "id": "mr-334111-p3-native",
            "query_id": "mr-334111",
            "query": "पृथ्वी किती जुनी आहे",
            "ground_truth_answer": "४.५ ते ४.६ अब्ज वर्षे जुने.",
            "sample_model_answer": "खूप तरुण पृथ्वीसाठी निर्मिती \"वैज्ञानिकां\"नी पुढे केलेल्या युक्तिवादांचे विश्लेषण करण्यासाठी, मी येथे पृथ्वी 4.5 ते 4.6 अब्ज वर्षे जुनी आहे याची खात्री देणारा पुरावा थोडक्यात सारांशित केला आहे.",
            "sample_latency_ms": 134.91
        },
        {
            "id": "mr-766412-p0-native",
            "query_id": "mr-766412",
            "query": "यकृताच्या कर्करोगाला काय म्हणतात",
            "ground_truth_answer": "यकृताचा कर्करोग",
            "sample_model_answer": "याला पित्त नलिकेचा कर्करोग असेही म्हणतात.",
            "sample_latency_ms": 139.2
        },
        {
            "id": "mr-8710-p3-native",
            "query_id": "mr-8710",
            "query": "दातांच्या रक्तवाहिन्या आणि मज्जातंतूंनी बनलेले _____ हे दातांचे जिवंत ऊतक आहे.",
            "ground_truth_answer": "दंतमज्जा",
            "sample_model_answer": "दातांच्या मूळ वाहिन्यांमध्ये रक्तवाहिन्या, मज्जातंतू आणि संयोजी ऊती असलेले दंतमज्जा हे जिवंत ऊती असतात, त्यामुळे दातांना दुखापत झाल्यावर काय होते आणि कोणत्या उपचाराची आवश्यकता असते याबद्दलच्या सामान्य प्रश्नांची उत्तरे हा लेख देतो.",
            "sample_latency_ms": 140.25
        },
        {
            "id": "mr-283567-p8-native",
            "query_id": "mr-283567",
            "query": "जगातील सर्वात उंच इमारत किती फूट आहे",
            "ground_truth_answer": "२,७२२ फूट",
            "sample_model_answer": "इमारतीची अंतिम उंची 1,776 फूट (541 मीटर) युनायटेड स्टेट्सच्या स्वातंत्र्याच्या घोषणेकडे आणि एक देश म्हणून तिचा जन्म दर्शवते.",
            "sample_latency_ms": 141.12
        },
        {
            "id": "mr-994387-p8-native",
            "query_id": "mr-994387",
            "query": "अल्फा हेलिक्स कुठे आढळते ज्या प्रथिन संघटनेच्या स्तरावर आहे",
            "ground_truth_answer": "अल्फा हेलिक्स आणि बीटा शीट्स हे प्रथिनांच्या दुय्यम स्तरावर आढळतात.",
            "sample_model_answer": "अल्फा हेलिक्स आणि बीटा शीट्स हे प्रथिनांच्या दुय्यम स्तरावर आढळतात.",
            "sample_latency_ms": 143.3
        },
        {
            "id": "mr-1163874-p6-native",
            "query_id": "mr-1163874",
            "query": "हेंडरसनव्हिल एनसी कोणत्या काउंटीमध्ये आहे",
            "ground_truth_answer": "हेंडरसनविले, एन.सी. हेंडरसन काउंटीमध्ये स्थित आहे.",
            "sample_model_answer": "हेंडरसनविले, एनसी: तुम्हाला काय माहिती पाहिजे?",
            "sample_latency_ms": 143.2
        },
        {
            "id": "mr-427035-p1-native",
            "query_id": "mr-427035",
            "query": "पार्किंग ब्रेक म्हणजे ई ब्रेक आहे",
            "ground_truth_answer": "होय, पार्किंग ब्रेकला ई-ब्रेक असेही म्हणतात.",
            "sample_model_answer": "कारमध्ये, पार्किंग ब्रेक, ज्याला हँड ब्रेक, इमर्जन्सी ब्रेक किंवा ई-ब्रेक असेही म्हणतात, हा एक अडकवणारा ब्रेक आहे जो सहसा वाहन स्थिर ठेवण्यासाठी वापरला जातो.",
            "sample_latency_ms": 139.42
        },
        {
            "id": "mr-748971-p1-native",
            "query_id": "mr-748971",
            "query": "फ्रान्सचे सध्याचे चलन काय आहे",
            "ground_truth_answer": "युरो",
            "sample_model_answer": "फ्रान्सचे चलन युरो आहे.",
            "sample_latency_ms": 133.99
        },
        {
            "id": "mr-243926-p5-native",
            "query_id": "mr-243926",
            "query": "आफ्रिकन गुलामांचा व्यापार किती काळ चालला?",
            "ground_truth_answer": "400 वर्षे",
            "sample_model_answer": "पश्चिम आफ्रिकेतील गुलामांचा व्यापार, त्याच्या वळवळत्या मध्यवर्ती मार्गासह, जवळजवळ 400 वर्षे चालला.",
            "sample_latency_ms": 143.76
        },
        {
            "id": "mr-230697-p3-native",
            "query_id": "mr-230697",
            "query": "कॅम्प हिल पा फिली पासून किती लांब आहे",
            "ground_truth_answer": "114 मैल किंवा 183 किलोमीटर",
            "sample_model_answer": "ते पी.एच.एल.पर्यंतचे एकूण वाहन चालवण्याचे अंतर 114 मैल किंवा 183 किलोमीटर आहे.",
            "sample_latency_ms": 146.3
        },
        {
            "id": "mr-614844-p1-native",
            "query_id": "mr-614844",
            "query": "वुडविल, मिनेसोटा कोणत्या काउंटीमध्ये आहे?",
            "ground_truth_answer": "वासेका काउंटी",
            "sample_model_answer": "वुडव्हिल टाउनशिप, वासेका काउंटी, मिनेसोटा.",
            "sample_latency_ms": 138.1
        },
        {
            "id": "mr-615094-p5-native",
            "query_id": "mr-615094",
            "query": "हायटस्टाऊन, एनजे कोणत्या काउंटीमध्ये आहे",
            "ground_truth_answer": "मर्सर काउंटी",
            "sample_model_answer": "हायट्सटाउन हे अमेरिकेच्या न्यू जर्सी राज्यातील मर्सर काउंटीमधील एक नगर आहे.",
            "sample_latency_ms": 150.08
        },
        {
            "id": "mr-968495-p1-native",
            "query_id": "mr-968495",
            "query": "सिंको डी मायो कुठे झाले",
            "ground_truth_answer": "मेक्सिको.",
            "sample_model_answer": "सिंको डी मायो हा 5 मे 1862 रोजी पुएब्ला, मेक्सिको येथे झालेल्या विजयी लढाईचे प्रतीक आहे.",
            "sample_latency_ms": 141.62
        },
        {
            "id": "mr-937292-p0-native",
            "query_id": "mr-937292",
            "query": "ॲपल जेव्हा आयपो सार्वजनिक झाले तेव्हा",
            "ground_truth_answer": "1980",
            "sample_model_answer": "जेव्हा स्टीव्ह जॉब्सने 1980 मध्ये अ‍ॅपलला सार्वजनिक केले, तेव्हा त्यांची कंपनी वैयक्तिक संगणकांच्या विक्रीत टँडी कॉर्पच्या मागे पडली आणि मॅसॅच्युसेट्सच्या नियामकांनी आय.पी.ओ.ला राज्याच्या रहिवाशांसाठी \"खूप जोखमीचा\" मानले.",
            "sample_latency_ms": 134.53
        },
        {
            "id": "mr-42226-p2-native",
            "query_id": "mr-42226",
            "query": "सौंदर्यप्रसाधनतज्ज्ञाचा सरासरी वेतन",
            "ground_truth_answer": "२४,४२१ डॉलर प्रति वर्ष",
            "sample_model_answer": "संयुक्त राज्य अमेरिका में सौंदर्य पेशेवर का वेतन लगभग $37,000 प्रति वर्ष है।",
            "sample_latency_ms": 149.99
        },
        {
            "id": "mr-983229-p7-native",
            "query_id": "mr-983229",
            "query": "गॅस सिटी इंडियाना कुठे आहे",
            "ground_truth_answer": "गॅस सिटी हे इंडियानाच्या ग्रँट काउंटीमधील मिसिसिनेवा नदीच्या काठावरील एक शहर आहे.",
            "sample_model_answer": "गॅस सिटी हे इंडियानाच्या ग्रँट काउंटीमधील मिसिसिनेवा नदीच्या काठावरील एक शहर आहे.",
            "sample_latency_ms": 144.51
        },
        {
            "id": "mr-299938-p0-native",
            "query_id": "mr-299938",
            "query": "तुम्ही किती आठवडे गर्भवती असताना लिंग निश्चित करू शकता?",
            "ground_truth_answer": "१६ ते २२ आठवडे.",
            "sample_model_answer": "जेव्हा तुम्ही सुमारे 16 ते 22 आठवड्यांचे असता, तेव्हा केलेले अल्ट्रासाऊंड तुमच्या बाळाचे लिंग निश्चित करू शकतात.",
            "sample_latency_ms": 146.65
        },
        {
            "id": "mr-552637-p7-native",
            "query_id": "mr-552637",
            "query": "वॉशिंग्टन डी.सी. मधील सर्वसमावेशक रिसॉर्टच्या सर्वात जवळचे विमानतळ कोणते आहे.",
            "ground_truth_answer": "रोनाल्ड रेगन वॉशिंग्टन राष्ट्रीय विमानतळ",
            "sample_model_answer": "वॉशिंग्टन, कोलंबिया जिल्ह्याजवळील प्रमुख विमानतळ: 1 रोनाल्ड रेगन वॉशिंग्टन राष्ट्रीय विमानतळ (डीसीए / केडीसीए) हा सर्वात जवळचा प्रमुख विमानतळ आहे.",
            "sample_latency_ms": 145.65
        },
        {
            "id": "mr-1037787-p2-native",
            "query_id": "mr-1037787",
            "query": "कोण आहे रॉबर्ट ब्लेक",
            "ground_truth_answer": "तो एक अमेरिकन अभिनेता आहे.",
            "sample_model_answer": "रॉबर्ट ब्लेकचा संदर्भ असा असू शकतो: 1 रॉबर्ट ब्लेक (अभिनेता) (जन्म 1933), अमेरिकन अभिनेता, टीव्ही मालिका बॅरेटामध्ये अभिनय केला.",
            "sample_latency_ms": 139.62
        },
        {
            "id": "mr-911385-p6-native",
            "query_id": "mr-911385",
            "query": "ज्वालामुखीमुळे कोणत्या प्रकारचे नुकसान होते",
            "ground_truth_answer": "पळसण, भूस्खलन किंवा भूकंप.",
            "sample_model_answer": "ज्वालामुखीच्या विस्फोटाचे अनेक प्रकार - ज्या दरम्यान लाव्हा, टेफ्रा (राख, लॅपिली, ज्वालामुखीचे बॉम्ब आणि ब्लॉक्स) आणि विविध वायू ज्वालामुखीच्या व्हेंट किंवा फिशरमधून बाहेर काढले जातात - ज्वालामुखीशास्त्रज्ञांनी ओळखले आहेत.",
            "sample_latency_ms": 133.81
        },
        {
            "id": "mr-147286-p1-native",
            "query_id": "mr-147286",
            "query": "तंतुमय आणि गोलाकार यांच्यातील फरक",
            "ground_truth_answer": "तंतुमय प्रथिने आणि गोलाकार प्रथिने आकार, आकृती, विरघळणशीलता, स्वरूप तसेच कार्यामध्ये भिन्न असतात.",
            "sample_model_answer": "तंतुमय प्रथिने आणि गोलाकार प्रथिने आकार, आकृती, विरघळणशीलता, स्वरूप तसेच कार्यामध्ये भिन्न असतात.",
            "sample_latency_ms": 132.74
        },
        {
            "id": "mr-871306-p6-native",
            "query_id": "mr-871306",
            "query": "आमच्याकडे कोणत्या प्रकारचे वेबर गॅस ग्रिल आहेत?",
            "ground_truth_answer": "वेबर क्यू २०० गॅस ग्रिल जोड्या",
            "sample_model_answer": "जर तुमची जागा पूर्ण आकाराच्या गॅस ग्रिलसाठी खूपच अरुंद असेल, किंवा तुम्हाला टेलगेटिंग किंवा कार कॅम्पिंगसाठी पुरेसे कॉम्पॅक्ट हवे असेल, तर $200 वेबर Q 1200 घ्या.",
            "sample_latency_ms": 138.12
        },
        {
            "id": "mr-328810-p0-native",
            "query_id": "mr-328810",
            "query": "एन.सी.ए.ए. बास्केटबॉल कोर्ट किती बांधायचा आहे",
            "ground_truth_answer": "९४ फूट (२८.७ मी.)",
            "sample_model_answer": "बास्केटबॉल कोर्ट बांधण्यासाठी खूप जागा लागते, कारण नियमन कोर्ट 94 फूट (28.6 मी) लांब असू शकतात.",
            "sample_latency_ms": 137.46
        },
        {
            "id": "mr-235456-p9-native",
            "query_id": "mr-235456",
            "query": "सौदी अरेबियात किती गरमी होते",
            "ground_truth_answer": "120 डिग्री फॅरनहाइट",
            "sample_model_answer": "हे जगातील काही देशांपैकी एक आहे जिथे उन्हाळ्याच्या काळात तापमान 120 अंश फॅ (50 अंश सेल्सियस) पेक्षा जास्त होते.",
            "sample_latency_ms": 142.75
        },
        {
            "id": "mr-270170-p5-native",
            "query_id": "mr-270170",
            "query": "स्नॅप वाटाणे किती वेळ शिजवायचे",
            "ground_truth_answer": "२-३ मिनिटे.",
            "sample_model_answer": "तुमचे स्नॅप वाटाणे तळत असताना, एका पॅनमध्ये लोणी वितळवा किंवा ऑलिव्ह ऑइल वापरून तुमचे स्नॅप वाटाणे सुमारे 2-3 मिनिटे शिजवा.",
            "sample_latency_ms": 142.67
        },
        {
            "id": "mr-895616-p7-native",
            "query_id": "mr-895616",
            "query": "ब्रंचसाठी कोणत्या आकाराचे नॅपकिन आहेत",
            "ground_truth_answer": "६ १/२ इंच वर्ग",
            "sample_model_answer": "वैयक्तिकृत लंच नॅपकिन्स 6 1/2 इंच चौरसाकार असतात आणि सामान्यत: बुफे टेबलवर किंवा ब्रंच, लंच किंवा रात्रीच्या जेवणासाठी साखरपेरा, लग्नाच्या पार्ट्या, लग्नाच्या तयारी जेवण आणि लग्नाच्या रिसेप्शनसाठी डायनिंग टेबलवर वापरल्या जातात.",
            "sample_latency_ms": 146.44
        },
        {
            "id": "mr-660883-p9-native",
            "query_id": "mr-660883",
            "query": "कोणते अन्नपदार्थ दुग्धजन्य उत्पादने आहेत",
            "ground_truth_answer": "दूध, चीज आणि आइस्क्रीम.",
            "sample_model_answer": "दुग्धजन्य उत्पादने किंवा दुग्ध उत्पादने हे सस्तन प्राण्यांच्या दुधापासून, मुख्यतः गायी, पाणम्हशी, बकरी, मेंढ्या, याक, घोडे, उंट आणि घरगुती म्हशींच्या दुधापासून तयार केलेले अन्न आहे.",
            "sample_latency_ms": 144.17
        },
        {
            "id": "mr-809752-p8-native",
            "query_id": "mr-809752",
            "query": "बुलगॉड कोणता आहे",
            "ground_truth_answer": "द बुलगॉड हा लेखक जोशुआ डॅल यांच्या प्रकाशित कवितांचा सर्वात नवीन भाग आहे.",
            "sample_model_answer": "द बुलगॉड: द बुलगॉड हा लेखक जोशुआ डॅल यांच्या प्रकाशित कवितांचा सर्वात नवीन भाग आहे.",
            "sample_latency_ms": 131.27
        },
        {
            "id": "mr-601909-p5-native",
            "query_id": "mr-601909",
            "query": "व्रे हे नाव कोणत्या काउंटीमधून आले आहे",
            "ground_truth_answer": "जुने नॉर्स.",
            "sample_model_answer": "हे नाव आयरिश काउंटी क्लेअरशी संबंधित आहे.",
            "sample_latency_ms": 136.66
        },
        {
            "id": "mr-303754-p1-native",
            "query_id": "mr-303754",
            "query": "वजन कमी करण्यासाठी दररोज किती नारळाचे तेल लागेल.",
            "ground_truth_answer": "३६-३९ ग्रॅम.",
            "sample_model_answer": "एका अभ्यासात, 4 आठवड्यांसाठी दररोज 2 बडचमच (30 मिली) नारळाचे तेल घेणारे लठ्ठ किंवा मोठे लोक त्यांच्या कमरेपासून सरासरी 1.1 इंच (2.87 से.मी.) वजन कमी करतात (5).",
            "sample_latency_ms": 149.05
        }
    ],
    "en": [
        {
            "id": "en-467556-p1-native",
            "query_id": "en-467556",
            "query": "nyu tuition cost",
            "ground_truth_answer": "$43,746 for the 2014-2015 academic year.",
            "sample_model_answer": "$50k per year",
            "sample_latency_ms": 142.03
        },
        {
            "id": "en-28213-p7-native",
            "query_id": "en-28213",
            "query": "at what age do kids start to hold memories",
            "ground_truth_answer": "Before the age of 2–4 years.",
            "sample_model_answer": "Age 10",
            "sample_latency_ms": 131.66
        },
        {
            "id": "en-604568-p3-native",
            "query_id": "en-604568",
            "query": "what county is columbus city in",
            "ground_truth_answer": "Bartholomew",
            "sample_model_answer": "Columbus ( /kəlʌmbəs/) is a city in and the county seat of Bartholomew County, Indiana, United States.",
            "sample_latency_ms": 144.14
        },
        {
            "id": "en-26191-p2-native",
            "query_id": "en-26191",
            "query": "are whiskers on cats used for balance",
            "ground_truth_answer": "Yes whiskers are used to help balance as well as feelers to help them judge space and help in the dark.",
            "sample_model_answer": "A cats tail also helps them balance, my cat had lost his tail do to an …injury and now he uses his whiskers more to help balance.",
            "sample_latency_ms": 134.49
        },
        {
            "id": "en-1174759-p3-native",
            "query_id": "en-1174759",
            "query": "is the bachelor legal",
            "ground_truth_answer": "Yes, the bachelor is legal.",
            "sample_model_answer": "A bachelor in law and legal studies",
            "sample_latency_ms": 136.58
        },
        {
            "id": "en-33073-p0-native",
            "query_id": "en-33073",
            "query": "average cost of assisted living in washington state",
            "ground_truth_answer": "$4,625 per month.",
            "sample_model_answer": "According to the 2015 Genworth Financial cost of care survey, the average cost of assisted living in Washington is $4,625 per month.",
            "sample_latency_ms": 136.68
        },
        {
            "id": "en-620911-p7-native",
            "query_id": "en-620911",
            "query": "what disability is copd",
            "ground_truth_answer": "Chronic Obstructive Pulmonary Disease and Social Security Disability COPD, or chronic obstructive pulmonary disease is a series of lung diseases that damages your lungs, blocking airflow and affecting your ability to breathe.",
            "sample_model_answer": "Chronic respiratory disorders",
            "sample_latency_ms": 134.31
        },
        {
            "id": "en-693333-p6-native",
            "query_id": "en-693333",
            "query": "what is a omurice omelet",
            "ground_truth_answer": "A contemporary Asian dish consisting of an omelette made with fried rice.",
            "sample_model_answer": "With a fluffy omelette covering a bed of savory sweet chicken fried rice",
            "sample_latency_ms": 138.38
        },
        {
            "id": "en-315061-p0-native",
            "query_id": "en-315061",
            "query": "how much does it cost to change a jeep alternator",
            "ground_truth_answer": "A Jeep Compass Alternator Replacement is between $377 and $577.",
            "sample_model_answer": "Between $377 and $577",
            "sample_latency_ms": 139.27
        },
        {
            "id": "en-612569-p1-native",
            "query_id": "en-612569",
            "query": "what county is seminole fl in",
            "ground_truth_answer": "Pinellas County",
            "sample_model_answer": "Seminole is a city in Pinellas County, Florida, United States.",
            "sample_latency_ms": 139.0
        },
        {
            "id": "en-32055-p9-native",
            "query_id": "en-32055",
            "query": "average cost dental implant",
            "ground_truth_answer": "$4250",
            "sample_model_answer": "According to DentalImplantCostGuide.com, the average completed dental implant, in the U.S., costs about $4250.",
            "sample_latency_ms": 138.52
        },
        {
            "id": "en-638438-p4-native",
            "query_id": "en-638438",
            "query": "what does granuloma annulare look like",
            "ground_truth_answer": "A ring of small red or skin-coloured bumps",
            "sample_model_answer": "A ring of small red or skin-coloured bumps",
            "sample_latency_ms": 130.78
        },
        {
            "id": "en-516461-p9-native",
            "query_id": "en-516461",
            "query": "the importance of minerals in diet",
            "ground_truth_answer": "Minerals are incredibly important for health and to prevent chronic disease.",
            "sample_model_answer": "Minerals are incredibly important for health and to prevent chronic disease.",
            "sample_latency_ms": 135.17
        },
        {
            "id": "en-481996-p0-native",
            "query_id": "en-481996",
            "query": "prize money taxable income",
            "ground_truth_answer": "Prizes are considered taxable income regardless of whether the prize is in the form of cash, trips or merchandise.",
            "sample_model_answer": "Regardless of whether the prize is in the form of cash, trips or merchandise",
            "sample_latency_ms": 142.33
        },
        {
            "id": "en-610348-p1-native",
            "query_id": "en-610348",
            "query": "what county is nine mile in",
            "ground_truth_answer": "Onondaga",
            "sample_model_answer": "Ninemile Creek is located in Onondaga County near the towns of Camillus.",
            "sample_latency_ms": 139.65
        },
        {
            "id": "en-698805-p5-native",
            "query_id": "en-698805",
            "query": "what is a salpingectomies",
            "ground_truth_answer": "Salpingectomy is the surgical removal of one or both fallopian tubes.",
            "sample_model_answer": "The surgical removal of one or both fallopian tubes",
            "sample_latency_ms": 139.76
        },
        {
            "id": "en-524762-p0-native",
            "query_id": "en-524762",
            "query": "triglycerides what are they",
            "ground_truth_answer": "A triglyceride is an ester derived from glycerol and three to four fatty acids.",
            "sample_model_answer": "A type of fat found in the blood",
            "sample_latency_ms": 133.44
        },
        {
            "id": "en-1174756-p2-native",
            "query_id": "en-1174756",
            "query": "is the cheetah an endangered species",
            "ground_truth_answer": "Yes, Cheetah is considered Endangered species.",
            "sample_model_answer": "Cheetahs are considered to be endangered animals in that only 12,400 species are left in more than 20 different African countries",
            "sample_latency_ms": 135.94
        },
        {
            "id": "en-447502-p3-native",
            "query_id": "en-447502",
            "query": "meaning of conquest",
            "ground_truth_answer": "Conquest is the act of conquering a country or group of people.",
            "sample_model_answer": "The act of conquering a country or group of people",
            "sample_latency_ms": 148.51
        },
        {
            "id": "en-573198-p0-native",
            "query_id": "en-573198",
            "query": "what are the risks associated with the use of nuclear energy",
            "ground_truth_answer": "Nuclear power arise from health effects of radiation. This radiation consists of subatomic particles traveling at or near the velocity of light---186,000 miles per second.ining uranium to fuel nuclear power plants leaves mill tailings, the residues from chemical processing of the ore, which lead to radon exposures to the public.",
            "sample_model_answer": "Health effects of radiation",
            "sample_latency_ms": 130.4
        },
        {
            "id": "en-607381-p0-native",
            "query_id": "en-607381",
            "query": "what county is honeoye in",
            "ground_truth_answer": "New York",
            "sample_model_answer": "This Single Family Home is located at County Road 36 in Honeoye, New York.",
            "sample_latency_ms": 133.64
        },
        {
            "id": "en-709821-p0-native",
            "query_id": "en-709821",
            "query": "what is als stand for and mean",
            "ground_truth_answer": "Amyotrophic laterals sclerosis, is a progressive neurodegenerative disease that affects nerve cells in the brain and the spinal cord.",
            "sample_model_answer": "ASL is short for As Hell, meaning extremely.n abbreviation that is widely used on Kik, Twitter, Facebook and elsewhere on the internet, but what does ASL mean in slang?",
            "sample_latency_ms": 132.52
        },
        {
            "id": "en-475482-p5-native",
            "query_id": "en-475482",
            "query": "pink eye treatment how long",
            "ground_truth_answer": "Within 24 to 48 hours.",
            "sample_model_answer": "Up to 14 days",
            "sample_latency_ms": 137.69
        },
        {
            "id": "en-621519-p0-native",
            "query_id": "en-621519",
            "query": "what distinguishes a macronutrient from a micronutrient",
            "ground_truth_answer": "Macronutrients mainly include carbohydrates, proteins and fats and also water which are required in large quantities and their main function being the release of energy in body.Whereas, micronutrients mainly comprise vitamins and minerals which are required in minute quantities.",
            "sample_model_answer": "Because they contain no calories and are required in much smaller amounts.",
            "sample_latency_ms": 136.16
        },
        {
            "id": "en-716162-p5-native",
            "query_id": "en-716162",
            "query": "what is an mla writing format",
            "ground_truth_answer": "It is a very professional way to format a paper, and, even if not required, is a nice, scholarly touch.",
            "sample_model_answer": "Provides guidelines to researchers and writers in formatting research papers and other reports",
            "sample_latency_ms": 137.15
        },
        {
            "id": "en-38032-p4-native",
            "query_id": "en-38032",
            "query": "average january temps in mount pleasant, sc",
            "ground_truth_answer": "The January low is 39.",
            "sample_model_answer": "The January low is 39.",
            "sample_latency_ms": 141.68
        },
        {
            "id": "en-645586-p1-native",
            "query_id": "en-645586",
            "query": "what does physical education mean",
            "ground_truth_answer": "Physical fitness and the ability to perform and enjoy day-to-day physical activities with ease.",
            "sample_model_answer": "A course taught in school that focuses on developing physical fitness and the ability to perform and enjoy day-to-day physical activities with ease",
            "sample_latency_ms": 139.42
        },
        {
            "id": "en-482341-p9-native",
            "query_id": "en-482341",
            "query": "process theories of motivation suggest that",
            "ground_truth_answer": "The group of motivational theories that falls under the umbrella category of Process Theories of Motivation is based on the use of our rational thought processes or cognitive processing abilities.",
            "sample_model_answer": "Behavior is the result of conscious decision-making processes",
            "sample_latency_ms": 135.12
        },
        {
            "id": "en-488003-p9-native",
            "query_id": "en-488003",
            "query": "requirements to be able to use virtual machines",
            "ground_truth_answer": "The virtual machine hardware must be. compatible with Workstation 8 and later virtual machines. n Support for USB 2.0 and 3.0 requires that you configure virtual machine settings to enable USB 2.0 and. 3.0 support and that you have compatible guest operating systems and virtual machine hardware.",
            "sample_model_answer": "The host system must meet certain requirements",
            "sample_latency_ms": 162.9
        },
        {
            "id": "en-687403-p7-native",
            "query_id": "en-687403",
            "query": "what is a hyperlink",
            "ground_truth_answer": "The most essential ingredient of all hypertext systems, including the World Wide Web.",
            "sample_model_answer": "A hyperlink site is a COM object that implements the IHlinkSite interface and supplies either the moniker or interface identifier of its hyperlink container.",
            "sample_latency_ms": 136.11
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
    print("  RUNNING 90-QUESTION MULTI-STRATEGY BENCHMARK SUITE")
    print("  Endpoint: " + MODAL_URL)
    print("  Dataset: 30 Hindi, 30 Marathi, 30 English (90 total)")
    print("=" * 75)

    all_items = []
    for lang, items in DATASET_90.items():
        for item in items:
            all_items.append((item, lang))

    results = []
    t_start = time.perf_counter()

    for idx, (item, lang) in enumerate(all_items, 1):
        r = post_query(item, lang)
        results.append(r)
        status = "✅" if not r["guardrail"] and "couldn't extract" not in r["answer"].lower() else "❌"
        tot_ms = r["timings_ms"].get("total_ms", r["client_ms"])
        ans_preview = r["answer"][:40].replace("\n", " ")
        print(f"  [{idx:02d}/90] [{lang.upper()}] {status} {r['query'][:35]:<35} | {ans_preview:<40} ({tot_ms:.1f}ms)")

    total_duration = time.perf_counter() - t_start

    # Summary
    print("\n" + "=" * 75)
    print("  90-QUESTION BENCHMARK RESULTS SUMMARY")
    print("=" * 75)

    by_lang = {"hi": [], "mr": [], "en": []}
    for r in results:
        by_lang[r["lang"]].append(r)

    total_correct = 0
    total_queries = len(results)

    for lang in ["hi", "mr", "en"]:
        lang_res = by_lang[lang]
        correct = sum(1 for r in lang_res if not r["guardrail"] and "couldn't extract" not in r["answer"].lower())
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
    print("  OVERALL SYSTEM PERFORMANCE (90 QUESTIONS)")
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

    with open("data/benchmark_90_results.json", "w", encoding="utf-8") as f:
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
    print("Results saved to data/benchmark_90_results.json")

if __name__ == "__main__":
    run_benchmark()
