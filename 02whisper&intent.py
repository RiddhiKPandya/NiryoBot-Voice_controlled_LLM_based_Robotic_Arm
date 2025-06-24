import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import spacy
import json  

# Load the small English spaCy model
nlp = spacy.load("en_core_web_sm")

colors = ["red", "green", "pink", "yellow", "lime_green"]  # Extended colors

def get_compound_noun(token):
    parts = [tok.text for tok in token.lefts if tok.dep_ == "compound"]
    parts.append(token.text)
    return " ".join(parts)

def get_objects(verb_token):
    objects = []
    for child in verb_token.children:
        if child.dep_ in ("dobj", "attr", "oprd", "pobj"):
            objects.append(get_compound_noun(child))
        elif child.dep_ == "prep":
            for subchild in child.children:
                if subchild.dep_ == "pobj":
                    objects.append(get_compound_noun(subchild))
    return objects

def extract_intents_and_objects(sentence):
    doc = nlp(sentence)
    results = []
    seen_verbs = set()

    for token in doc:
        if token.pos_ == "VERB":
            if (token.lemma_, token.i) in seen_verbs:
                continue
            seen_verbs.add((token.lemma_, token.i))

            objs = get_objects(token)
            results.append((token.lemma_, objs))

            for conj in token.conjuncts:
                if conj.pos_ == "VERB" and (conj.lemma_, conj.i) not in seen_verbs:
                    seen_verbs.add((conj.lemma_, conj.i))
                    conj_objs = get_objects(conj)
                    results.append((conj.lemma_, conj_objs))

    # Patch: manually detect 'point at <color> card' style
    if not results:
        for i, token in enumerate(doc):
            if token.text.lower() == "point":
                for child in token.children:
                    if child.dep_ == "prep" and child.text.lower() == "at":
                        for subchild in child.children:
                            if subchild.dep_ == "pobj" and subchild.text.lower() == "card":
                                color = None
                                for grandchild in subchild.lefts:
                                    if grandchild.text.lower() in colors:
                                        color = grandchild.text.lower()
                                if color:
                                    results.append(("point", [f"{color} card"]))

    return results

def extract_color_from_objects(objects):
    for obj in objects:
        for color in colors:
            if color in obj.lower():
                return color.upper()  # Match with YOLO keys
    return None

if __name__ == "__main__":
    fs = 44100
    seconds = 5

    print("  Recording...")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write("mic_input.wav", fs, recording)
    print(" Recording complete. Translating...")

    model = whisper.load_model("small")
    result = model.transcribe("mic_input.wav", task="translate")

    print("\n Translated Text:")
    print(result["text"])

    print("\n Extracted Intents and Objects:")
    output = extract_intents_and_objects(result["text"])
    for intent, objs in output:
        print(f" → Intent: {intent}, Objects: {objs}")

        if intent == "point":
            color = extract_color_from_objects(objs)
            if color:
                # ✅ Save to JSON
                with open("spoken_color.json", "w") as f:
                    json.dump({"color": color}, f)
                print(f"\nSaved spoken color to 'spoken_color.json': {color}")
            else:
                print(" No valid color found in spoken command.")
