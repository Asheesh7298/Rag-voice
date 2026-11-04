// Example questions shown on the start screen and as quick chips (all answerable from the index).

export interface Example {
  lang: "en" | "hi" | "mr";
  label: string;
  text: string;
}

export const STARTER_QUESTIONS: Example[] = [
  { lang: "en", label: "ENGLISH", text: "what county is columbus city in" },
  { lang: "en", label: "ENGLISH", text: "what is providing customer service" },
  { lang: "hi", label: "HINDI", text: "ब्राइटन टाउनशिप फोन नंबर" },
  { lang: "hi", label: "HINDI", text: "मेन में सबसे अमीर व्यक्ति कौन है" },
  { lang: "mr", label: "MARATHI", text: "महाराष्ट्राची राजधानी कोणती" },
  { lang: "mr", label: "MARATHI", text: "संगणक म्हणजे काय" },
];

export const QUICK_CHIPS: (Example & { title: string })[] = [
  { lang: "en", label: "ENGLISH", title: "Columbus County", text: "what county is columbus city in" },
  { lang: "en", label: "ENGLISH", title: "Jeep Alternator", text: "how much does it cost to change a jeep alternator" },
  { lang: "hi", label: "HINDI", title: "ब्राइटन फोन", text: "ब्राइटन टाउनशिप फोन नंबर" },
  { lang: "hi", label: "HINDI", title: "मेन अमीर व्यक्ति", text: "मेन में सबसे अमीर व्यक्ति कौन है" },
  { lang: "mr", label: "MARATHI", title: "फ्रान्सचे चलन", text: "फ्रान्सचे सध्याचे चलन काय आहे" },
  { lang: "mr", label: "MARATHI", title: "पार्किंग ब्रेक", text: "पार्किंग ब्रेक म्हणजे ई ब्रेक आहे" },
];

export const DEFAULT_HINT = "🎙️ Press mic to speak or type in English, Hindi, or Marathi";
