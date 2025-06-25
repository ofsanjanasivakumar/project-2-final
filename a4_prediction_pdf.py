import re
import unicodedata
import torch
import fitz
from a1_model_5 import HybridHateSpeechModel
from transformers import DistilBertTokenizer
import nltk
from fpdf import FPDF
nltk.download("punkt")
LEET_MAPPING={'4': 'a', '@': 'a', '8': 'b', '(': 'c', '3': 'e', '6': 'g', '1': 'i', '!': 'i', '0': 'o', '$': 's', '5': 's', '7': 't', '9': 'g', '+': 't', '2': 'z'}
HOMOGLYPH_MAPPING={'ɑ': 'a', 'ɒ': 'a', '℮': 'e', 'ʃ': 's', 'і': 'i', 'ᴍ': 'm', '𝗇': 'n', '𝗀': 'g', '𝗋': 'r'}
SPACED_WORDS=["nigger", "faggot", "kike", "chink", "tranny", "spic", "dyke", "raghead", "whore", "bitch", "fag", "nigga", "hoe"]
def normalize_text(text):
    text=''.join(HOMOGLYPH_MAPPING.get(c,c) for c in text)
    text=unicodedata.normalize('NFKC',text)
    def replace_leetspeak(match):
        return ''.join(LEET_MAPPING.get(c, c) for c in match.group())
    text=re.sub(r'\b[a-zA-Z0-9@!$+]+\b',replace_leetspeak,text)
    for word in SPACED_WORDS:
        spaced_pattern=r'\b' + r'\s*'.join(list(word)) + r'\b'
        text=re.sub(spaced_pattern,word,text,flags=re.IGNORECASE)
    return text
victim_keywords={
    "Gender": {
        "Women": ["bitch", "cunt", "ho", "hoes", "hoe", "cunts", "bitches", "women", "female"],
        "Men": ["male", "boy"],
        "Homosexual": ["fag", "faggot", "dyke", "gay", "lesbian"],
        "Bisexual": ["bisexual"],
        "Pansexual": ["pan", "pansexual"],
        "Transgender": ["tranny", "transgender", "trans"]
    },
    "Race": {
        "African": ["nigger", "niggas", "nigga", "niggers", "black"],
        "Arab": ["arab", "camel jockey", "raghead"],
        "Caucasian": ["white", "honky", "karen"],
        "Hispanic": ["beaner", "spic", "mexican"],
        "Indian": ["indian", "desi"],
        "Chinese": ["chink", "ching chong", "chinese"],
        "Korean": ["korean"],
        "Japanese": ["jap", "japanese"],
        "Filipino": ["gook", "philippines"],
        "Pakistani": ["paki", "pakistani"]
    },
    "Religion": {
        "Islam": ["muslim", "terrorist", "muzzie", "moslem", "muzrat"],
        "Christianity": ["christian", "church"],
        "Jewish": ["jew", "jewish", "hitler", "kike", "yid"],
        "Hindu": ["hindu", "cow worship"]
    },
    "Other": {
        "Economic": ["trailer trash", "poor"],
        "Age": ["old", "young", "boomer"],
        "Refugee": ["refugee", "immigrant"]
    }
}
victim_resources={
    "Women": "https://www.unwomen.org/en",
    "Men": "https://ukmensday.org.uk/resources/",
    "Homosexual": "https://www.thetrevorproject.org/",
    "Bisexual": "https://biresource.org/",
    "Pansexual": "https://www.glaad.org/",
    "Transgender": "https://transequality.org/",
    "African": "https://naacp.org/",
    "Arab": "https://www.aaiusa.org/",
    "Caucasian": "https://www.adl.org/",
    "Hispanic": "https://www.unidosus.org/",
    "Indian": "https://standagainsthate.com/",
    "Chinese": "https://www.stopaapihate.org/",
    "Korean": "https://www.kacla.org/",
    "Japanese": "https://www.jacl.org/",
    "Filipino": "https://naffaa.org/",
    "Pakistani": "https://www.mpac.org/",
    "Islam": "https://www.cair.com/",
    "Christianity": "https://www.christianity.com/",
    "Jewish": "https://www.adl.org/",
    "Hindu": "https://www.hafsite.org/",
    "Economic": "https://www.feedingamerica.org/",
    "Age": "https://www.aarp.org/",
    "Refugee": "https://www.unhcr.org/"
}
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=HybridHateSpeechModel(tfidf_vocab_size=0,num_classes=2,num_victim_groups=4).to(device)
model.load_state_dict(torch.load("model_5_victim.pth",map_location=device))
model.eval()
tokenizer=DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
def refine_victim_groups(text,victim_labels):
    text_lower=text.lower()
    refined_labels=set()
    for broad_group,subgroups in victim_keywords.items():
        if broad_group in victim_labels:
            for subgroup,keywords in subgroups.items():
                if any(keyword in text_lower for keyword in keywords):
                    refined_labels.add(subgroup)
            if not refined_labels:
                refined_labels.add(broad_group)
    return list(refined_labels)
def predict(text):
    text=normalize_text(text)
    input_ids=tokenizer(text,padding="max_length",truncation=True,max_length=128,return_tensors="pt")["input_ids"].to(device)
    attention_mask=tokenizer(text,padding="max_length",truncation=True,max_length=128,return_tensors="pt")["attention_mask"].to(device)
    edge_index=torch.zeros((2,1),dtype=torch.long).to(device)
    with torch.no_grad():
        toxicity_logits,victim_logits=model(input_ids,attention_mask,edge_index)
    toxicity_pred=torch.argmax(toxicity_logits,dim=1).item()
    victim_pred=(torch.sigmoid(victim_logits)>0.4).cpu().numpy()
    victim_groups=["Race","Religion","Gender","Other"]
    targeted_groups=[victim_groups[i] for i, is_targeted in enumerate(victim_pred[0]) if is_targeted]
    if toxicity_pred==1:
        targeted_groups=refine_victim_groups(text,targeted_groups)
        return toxicity_pred,targeted_groups
    return toxicity_pred,[]
def process_pdf(input_pdf, output_pdf):
    doc=fitz.open(input_pdf)
    text="\n".join([page.get_text("text") for page in doc])
    sentences=nltk.sent_tokenize(text)
    pdf=FPDF()
    pdf.add_page()
    pdf.set_font("Arial",size=12)
    for i, sentence in enumerate(sentences):
        toxicity,victims=predict(sentence)
        label="Hate Speech" if toxicity == 1 else "Not Hate Speech"
        output_text=f"Sentence {i+1}: {sentence}\nPrediction: {label}\n"
        if toxicity==1 and victims:
            output_text+=f"Targeted Groups: {', '.join(victims)}\n"
            for victim in victims:
                if victim in victim_resources:
                    output_text+=f"Resource: {victim_resources[victim]}\n"
        pdf.multi_cell(0,10,output_text+"\n")
    pdf.output(output_pdf)
    print(f"Predictions saved to {output_pdf}!")
process_pdf("input.pdf","predictions.pdf")
