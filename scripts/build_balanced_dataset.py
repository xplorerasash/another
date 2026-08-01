"""Build a balanced training dataset for SafeChat-AI.

The raw data (dataset/labeled.csv) is a Twitter cyberbullying set with a
heavy class imbalance (20,620 harmful vs 4,163 safe) and almost no normal
conversational text. A model trained on it alone over-predicts "harmful"
(e.g. it flags simple greetings like "Hello!") while under-predicting clean
(non-profanity) insults and threats.

This script:
  1. Loads dataset/labeled.csv
  2. Keeps the original safe tweets (minus obvious profanity)
  3. Adds thousands of clean, diverse conversational samples (greetings,
     questions, small talk, chatbot queries)
  4. Adds clean (non-profanity) insults, bullying, and threats so the
     classifier learns those patterns without relying on profanity
  5. Balances both classes and writes dataset/balanced.csv
"""
import random
import re
from pathlib import Path

import pandas as pd

SRC = Path("dataset/labeled.csv")
OUT = Path("dataset/balanced.csv")
TARGET_PER_CLASS = 9000
SEED = 42

PROFANITY = re.compile(
    r"\b("
    r"fuck|shit|bitch|ass(?:hole)?|dick|cunt|whore|hoe|slut|pussy|fag|"
    r"retard|bastard|nigg|nazi|loser|dumb|idiot|stupid|ugly|pathetic|"
    r"worthless|kill\s+(you|yourself|urself|my)|die|rape|murder|terrorist|"
    r"gay|tranny|pedo"
    r")\b",
    re.IGNORECASE,
)

GREETINGS = [
    "hi", "hello", "hey", "hey there", "hi there", "good morning",
    "good afternoon", "good evening", "nice to meet you", "howdy",
    "what's up", "whats up", "how are you", "how's it going", "how are you doing",
    "how are things", "long time no see", "greetings", "hi everyone",
    "hello there", "yo", "good to see you", "great to see you", "how do you do",
]

SMALL_TALK = [
    "the weather is nice today",
    "it's raining outside",
    "i'm feeling good today",
    "i had a long day at work",
    "what a beautiful day",
    "have a nice day",
    "see you later",
    "talk to you soon",
    "have a good weekend",
    "take care",
    "i'm a bit tired today",
    "today was a busy day",
    "i'm looking forward to the weekend",
    "it's a lovely morning",
    "the coffee here is really good",
    "i slept well last night",
    "i have a meeting later",
    "i'm going for a walk",
    "the sun is shining",
    "i feel relaxed today",
    "how was your trip",
    "did you watch the game last night",
    "i read a good book this week",
    "i cooked dinner at home",
    "the traffic was bad this morning",
    "i finished my homework early",
]

TOPICS = [
    "machine learning", "artificial intelligence", "python", "programming",
    "the weather", "this movie", "that book", "cooking", "football",
    "cricket", "tennis", "music", "photography", "science", "history",
    "geography", "the internet", "computers", "robots", "space", "cars",
    "animals", "food", "health", "exercise", "meditation", "reading",
    "writing", "mathematics", "physics", "chemistry", "biology",
    "economics", "culture", "travel", "languages", "video games", "art",
    "design", "business", "education", "gardening", "hiking", "swimming",
    "painting", "dancing", "cyber security", "data science", "web design",
    "coffee", "tea", "chocolate", "pizza", "basketball", "baseball",
    "chess", "puzzles", "the news", "the economy", "climate change",
    "the ocean", "mountains", "beaches", "forests", "birds", "dogs", "cats",
    "dinosaurs", "planets", "the moon", "the sun", "stars", "galaxies",
    "airplanes", "trains", "bicycles", "electric cars", "smartphones",
    "tablets", "laptops", "software", "apps", "websites", "databases",
    "cloud computing", "blockchain", "virtual reality", "augmented reality",
    "the stock market", "investing", "saving money", "budgeting", "cooking",
    "baking", "grilling", "vegetables", "fruits", "rice", "bread", "cheese",
    "yoga", "running", "jogging", "weight training", "stretching",
    "swimming lessons", "public speaking", "negotiation", "leadership",
    "teamwork", "problem solving", "critical thinking", "creativity",
    "innovation", "entrepreneurship", "marketing", "advertising", "sales",
    "customer service", "project management", "time management",
    "note taking", "homework", "exams", "university", "college", "school",
    "libraries", "museums", "theater", "cinema", "concerts", "festivals",
    "holidays", "birthdays", "weddings", "family", "friends", "neighbors",
    "community", "volunteering", "charity", "the environment", "recycling",
    "solar power", "wind energy", "electricity", "water conservation",
    "gardens", "parks", "playgrounds", "restaurants", "cafes", "markets",
    "shopping", "fashion", "clothes", "shoes", "jewelry", "watches",
    "comics", "novels", "poetry", "essays", "letters", "diaries",
    "languages learning", "grammar", "vocabulary", "pronunciation",
    "tourism", "hotels", "airports", "railways", "roads", "bridges",
    "the circus", "magic tricks", "card games", "board games", "puzzles",
    "origami", "pottery", "woodworking", "sewing", "knitting", "embroidery",
]

ACTIONS = [
    "cook pasta", "learn python", "fix my computer", "write an essay",
    "start a business", "lose weight", "play guitar", "install software",
    "bake a cake", "clean my room", "plan a trip", "build a website",
    "study for exams", "save money", "learn a new language", "go hiking",
    "read a book", "watch a movie", "listen to music", "take photos",
    "practice yoga", "go running", "prepare dinner", "buy groceries",
    "call my family", "organize my desk", "water the plants", "take a nap",
    "draw a picture", "write a poem", "solve a puzzle", "repair my bike",
]

ADJECTIVES = ["great", "good", "interesting", "fascinating", "helpful", "useful", "fun", "exciting"]

QUESTIONS = [
    "what is {t}", "what is a {t}", "how does {t} work",
    "can you explain {t}", "tell me about {t}", "what do you think about {t}",
    "how do i {a}", "can you help me with {t}", "why is {t} important",
    "do you like {t}", "what is the best way to learn {t}",
    "is {t} difficult to learn", "how long does it take to learn {t}",
    "can you recommend a good {t}",
]

STATEMENTS = [
    "i like {t}", "i enjoy {t}", "i love {t}", "i think {t} is {adj}",
    "i find {t} interesting", "i usually {a} on weekends",
    "i am learning {t}", "i just finished {a}", "today i will {a}",
    "i agree with you", "that sounds great", "that makes sense",
    "i see what you mean", "i would like to learn more about {t}",
    "i have a question about {t}", "this is a nice day for {a}",
]

COMPLIMENTS = [
    "thank you", "thanks a lot", "thanks for your help",
    "i appreciate your help", "you are great", "you are helpful",
    "you are awesome", "nice job", "well done", "goodbye", "bye",
    "good night", "thanks for everything", "i appreciate you",
    "you did a great job", "that was very helpful",
]

CHATBOT_QUERIES = [
    "what is machine learning", "who are you", "what can you do",
    "tell me about yourself", "what is your name", "are you a robot",
    "what is bert", "teach me python", "explain artificial intelligence",
    "how do chatbots work", "what can a chatbot do", "tell me a fun fact",
    "what is your purpose", "how were you made", "do you understand me",
]

# ---- Clean harmful augmentation (insults/bullying/threats, no profanity) ----

ADJ_INSULTS = [
    "stupid", "dumb", "worthless", "pathetic", "useless", "hopeless",
    "incompetent", "disgusting", "repulsive", "brainless", "clueless",
    "ignorant", "annoying", "despicable", "contemptible", "foolish",
]

NOUN_INSULTS = [
    "an idiot", "a moron", "a loser", "a joke", "a waste of space",
    "a burden", "an embarrassment", "a failure", "a disappointment",
    "a freak", "a disgrace", "a nobody", "a clown", "a weakling",
    "a parasite", "a stain on society", "a mess", "a laughing stock",
    "a total disaster", "an absolute joke",
]

INSULT_TEMPLATES = [
    "you are {i}", "you are so {i}", "you are really {i}", "you're {i}",
    "you're so {i}", "you are truly {i}", "you are just {i}",
    "you are nothing but {i}", "you're nothing but {i}", "i think you are {i}",
    "you will always be {i}", "everyone thinks you are {i}",
    "stop being {i}", "you are clearly {i}",
]

AMPLIFIERS = ["so", "really", "very", "extremely", "absolutely", "totally", "completely"]

INSULT_SUFFIXES = [
    "and everyone hates you", "and nobody likes you", "and i hate you",
    "and you know it", "and everyone knows it",
]

PERSONAL_ATTACKS = [
    "nobody likes you", "everyone hates you", "no one wants you here",
    "you don't belong here", "you should not exist", "go away forever",
    "why do you even bother", "you ruin everything", "you are the worst",
    "i hate you", "i hate everything about you", "i can't stand you",
    "you disgust me", "you make me sick", "shut up", "shut your mouth",
    "get lost", "you're pathetic and everyone knows it",
    "i wish you were never born", "you are a burden to everyone",
]

THREATS = [
    "you better watch your back", "watch your back", "watch out or else",
    "i'm going to get you", "you will regret this", "you will pay for this",
    "i'm going to hurt you", "i'm going to beat you up", "i will find you",
    "i know where you live", "you're finished", "you're dead",
    "i will destroy you", "i am going to destroy you", "you better run",
    "you better not show your face again", "i will make you pay",
    "you won't get away with this", "i'm coming for you",
    "i am going to kill you", "i will end you", "you are a dead man",
]

BULLYING_PHRASES = [
    "go back to where you came from", "you don't deserve to be here",
    "nobody wants you around", "you're not welcome here",
    "why don't you just disappear", "no one cares about you",
    "you drag everyone down", "everything about you is wrong",
    "you ruin every conversation", "people laugh at you behind your back",
    "you're an embarrassment to everyone", "why are you even here",
    "you add nothing to this group", "leave and never come back",
]

SPECIFIC_HARMFUL = [
    "you are stupid", "you are so stupid", "you are an idiot",
    "you are a dumb idiot", "you are worthless", "you are worthless and pathetic",
    "you are pathetic", "you are useless", "you are a loser",
    "i hate you", "i really hate you", "i hate you so much",
    "you better watch your back", "watch your back", "i am going to get you",
    "you will regret this", "i am going to hurt you", "i am going to kill you",
    "shut up", "shut your mouth", "nobody likes you", "everyone hates you",
    "you don't belong here", "go away and never come back",
]


def _build_augmented_harmful() -> list:
    """Generate clean, non-profanity insults/bullying/threats so the model
    learns these patterns independently of profanity."""
    samples: list = []

    def add(text: str):
        t = text.strip().lower()
        if t:
            samples.append(t)

    for tpl in INSULT_TEMPLATES:
        for i in ADJ_INSULTS + NOUN_INSULTS:
            add(tpl.format(i=i))

    for amp in AMPLIFIERS:
        for i in ADJ_INSULTS:
            add(f"you are {amp} {i}")

    for i in ADJ_INSULTS:
        for suffix in INSULT_SUFFIXES:
            add(f"you are {i} {suffix}")

    for i1 in ADJ_INSULTS[:10]:
        for i2 in ADJ_INSULTS[:10]:
            if i1 != i2:
                add(f"you are {i1} and {i2}")

    for p in PERSONAL_ATTACKS:
        add(p)
    for t in THREATS:
        add(t)
    for b in BULLYING_PHRASES:
        add(b)
    for s in SPECIFIC_HARMFUL:
        add(s)

    return list(dict.fromkeys(samples))


def _build_augmented(rng: random.Random) -> list:
    samples: list = []

    def add(text: str):
        t = text.strip().lower()
        if t and not PROFANITY.search(t):
            samples.append(t)

    for g in GREETINGS:
        add(g)
    for s in SMALL_TALK:
        add(s)
    for c in COMPLIMENTS:
        add(c)
    for q in CHATBOT_QUERIES:
        add(q)

    # Enumerate every combination deterministically for maximum diversity.
    for t in TOPICS:
        for a in ACTIONS:
            for adj in ADJECTIVES:
                for q in QUESTIONS:
                    add(q.format(t=t, a=a))
                for s in STATEMENTS:
                    add(s.format(t=t, a=a, adj=adj))

    return samples


def main():
    rng = random.Random(SEED)
    df = pd.read_csv(SRC, dtype={"label": int}).dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)

    safe = df[df["label"] == 0]["text"].tolist()
    harmful = df[df["label"] == 1]["text"].tolist()

    safe_clean = [t.strip() for t in safe if t.strip() and not PROFANITY.search(t) and len(t.strip()) > 5]

    augmented = _build_augmented(rng)
    combined_safe = list(dict.fromkeys(safe_clean + augmented))
    print(f"Original safe: {len(safe)} -> clean: {len(safe_clean)}")
    print(f"Augmented safe generated: {len(augmented)}")
    print(f"Combined unique safe: {len(combined_safe)}")

    if len(combined_safe) < TARGET_PER_CLASS:
        raise ValueError(
            f"Only {len(combined_safe)} safe samples after augmentation; "
            f"target is {TARGET_PER_CLASS}. Increase augmentation."
        )
    rng.shuffle(combined_safe)
    safe_sample = combined_safe[:TARGET_PER_CLASS]

    augmented_harmful = _build_augmented_harmful()
    print(f"Augmented harmful generated: {len(augmented_harmful)}")
    if len(augmented_harmful) >= TARGET_PER_CLASS:
        raise ValueError(
            f"Curated harmful examples ({len(augmented_harmful)}) must stay "
            f"below the per-class target ({TARGET_PER_CLASS})."
        )

    rng.shuffle(harmful)
    harmful_sample = augmented_harmful + harmful[:TARGET_PER_CLASS - len(augmented_harmful)]
    print(f"Final safe: {len(safe_sample)}, harmful: {len(harmful_sample)} "
          f"({len(augmented_harmful)} curated + {len(harmful_sample) - len(augmented_harmful)} tweets)")

    out_df = pd.DataFrame(
        {"text": safe_sample + harmful_sample, "label": [0] * len(safe_sample) + [1] * len(harmful_sample)}
    )
    out_df = out_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    out_df.to_csv(OUT, index=False)
    print(f"Wrote {len(out_df)} balanced rows to {OUT}")


if __name__ == "__main__":
    main()
