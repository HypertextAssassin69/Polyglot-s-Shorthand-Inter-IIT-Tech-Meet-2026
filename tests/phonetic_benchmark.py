import pandas as pd
import re
from collections import defaultdict

from src.polyglot.normalization.phonetic import phonetic_tokenize
from src.polyglot.normalization.pragmatic import extract_pragmatics


DATA_PATH = "data/raw/comi_lingua/TN_train.csv"


# --------------------------------------------------
# Basic helpers
# --------------------------------------------------

def clean_word(word):
    return re.sub(r"[^\w]", "", word.lower())


def representation(word):
    return tuple(phonetic_tokenize(word))


# --------------------------------------------------
# 1. Sanity checks
# --------------------------------------------------

def sanity_checks():

    pairs = [
        ("mama", "mamma"),
        ("kam", "kaam"),
        ("sath", "saath"),
        ("pani", "paani"),
        ("muje", "mujhe"),
        ("kese", "kaise"),
    ]

    print("\n=== SANITY CHECKS ===")

    for a, b in pairs:

        ra = representation(a)
        rb = representation(b)

        print(f"{a:10} -> {list(ra)}")
        print(f"{b:10} -> {list(rb)}")
        print(f"same      -> {ra == rb}")
        print()


# --------------------------------------------------
# 2. Dataset coverage
# --------------------------------------------------

def dataset_coverage():

    df = pd.read_csv(DATA_PATH)

    total_changes = 0
    matches = 0

    for raw_sentence, normalized_sentence in zip(
        df["Sentences"],
        df["Annotated by: Annotator 1"]
    ):

        raw_words = raw_sentence.split()
        normalized_words = normalized_sentence.split()

        for raw_word, normalized_word in zip(
            raw_words,
            normalized_words
        ):

            raw_word = clean_word(raw_word)
            normalized_word = clean_word(normalized_word)

            if not raw_word or not normalized_word:
                continue

            if raw_word == normalized_word:
                continue

            total_changes += 1

            if representation(raw_word) == representation(normalized_word):
                matches += 1

    print("\n=== DATASET COVERAGE ===")

    print(f"Total observed changes : {total_changes:,}")
    print(f"Phonetic matches       : {matches:,}")
    print(f"Coverage               : {matches / total_changes:.2%}")


# --------------------------------------------------
# 3. Rule coverage
# --------------------------------------------------

RULES = [
    ("a", "AA"),
    ("i", "EE"),
    ("u", "OO"),
    ("j", "JH"),
    ("e", "AI"),
    ("a", "A"),
]


def equivalent_with_rule(tokens1, tokens2, rule):

    a, b = rule

    def convert(tokens, source, target):
        return tuple(
            target if token == source else token
            for token in tokens
        )

    return (
        convert(tokens1, a, b) == tokens2
        or
        convert(tokens2, a, b) == tokens1
    )


def rule_coverage():

    df = pd.read_csv(DATA_PATH)

    total_changes = 0
    matches = defaultdict(int)

    for raw_sentence, normalized_sentence in zip(
        df["Sentences"],
        df["Annotated by: Annotator 1"]
    ):

        raw_words = raw_sentence.split()
        normalized_words = normalized_sentence.split()

        for raw_word, normalized_word in zip(
            raw_words,
            normalized_words
        ):

            raw_word = clean_word(raw_word)
            normalized_word = clean_word(normalized_word)

            if not raw_word or not normalized_word:
                continue

            if raw_word == normalized_word:
                continue

            total_changes += 1

            raw_tokens = representation(raw_word)
            normalized_tokens = representation(normalized_word)

            for rule in RULES:

                if equivalent_with_rule(
                    raw_tokens,
                    normalized_tokens,
                    rule
                ):
                    matches[rule] += 1

    print("\n=== RULE COVERAGE ===")

    for rule in RULES:

        count = matches[rule]

        print(
            f"{rule[0]:>3} <-> {rule[1]:<3} | "
            f"{count:>7,} matches | "
            f"{count / total_changes:.2%}"
        )


# --------------------------------------------------
# 4. Vocabulary collision analysis
# --------------------------------------------------

def build_vocabulary():

    df = pd.read_csv(DATA_PATH)

    vocabulary = set()

    for sentence in df["Sentences"]:

        for word in sentence.split():

            word = clean_word(word)

            if word:
                vocabulary.add(word)

    return vocabulary


def collision_analysis():

    vocabulary = build_vocabulary()

    print("\n=== COLLISION ANALYSIS ===")
    print(f"Vocabulary size: {len(vocabulary):,}")

    for rule in RULES:

        groups = defaultdict(list)

        for word in vocabulary:

            tokens = representation(word)

            a, b = rule

            normalized = tuple(
                b if token == a else token
                for token in tokens
            )

            groups[normalized].append(word)

        collisions = {
            key: words
            for key, words in groups.items()
            if len(words) > 1
        }

        print(
            f"\n{rule[0]} <-> {rule[1]} "
            f"| collision groups: {len(collisions):,}"
        )

        shown = 0

        for words in collisions.values():

            print("   ", ", ".join(sorted(words)))

            shown += 1

            if shown >= 10:
                break


# --------------------------------------------------
# 5. Candidate generation
# --------------------------------------------------

def build_phonetic_index():

    vocabulary = build_vocabulary()

    index = defaultdict(set)

    for word in vocabulary:

        tokens = representation(word)

        index[tokens].add(word)

    return index


def generate_candidates(word, index):

    tokens = representation(word)

    candidates = set()

    # Original representation
    candidates.update(index.get(tokens, set()))

    # Try each experimental rule in both directions
    for rule in RULES:

        a, b = rule

        # a -> b
        converted = tuple(
            b if token == a else token
            for token in tokens
        )

        candidates.update(
            index.get(converted, set())
        )

        # b -> a
        converted = tuple(
            a if token == b else token
            for token in tokens
        )

        candidates.update(
            index.get(converted, set())
        )

    return candidates


def candidate_evaluation():

    df = pd.read_csv(DATA_PATH)

    index = build_phonetic_index()

    total = 0

    recall_1 = 0
    recall_2 = 0
    recall_5 = 0

    candidate_counts = []

    examples = []

    for raw_sentence, normalized_sentence in zip(
        df["Sentences"],
        df["Annotated by: Annotator 1"]
    ):

        raw_words = raw_sentence.split()
        normalized_words = normalized_sentence.split()

        for raw_word, normalized_word in zip(
            raw_words,
            normalized_words
        ):

            raw_word = clean_word(raw_word)
            normalized_word = clean_word(normalized_word)

            if not raw_word or not normalized_word:
                continue

            if raw_word == normalized_word:
                continue

            total += 1

            candidates = generate_candidates(
                raw_word,
                index
            )

            candidate_list = sorted(candidates)

            candidate_counts.append(len(candidate_list))

            if normalized_word in candidate_list[:1]:
                recall_1 += 1

            if normalized_word in candidate_list[:2]:
                recall_2 += 1

            if normalized_word in candidate_list[:5]:
                recall_5 += 1

            if (
                normalized_word in candidates
                and len(examples) < 30
            ):
                examples.append(
                    (
                        raw_word,
                        normalized_word,
                        candidate_list[:10]
                    )
                )

    print("\n=== CANDIDATE GENERATION ===")

    print(f"Changed word pairs : {total:,}")

    print(
        f"Recall@1           : "
        f"{recall_1 / total:.2%}"
    )

    print(
        f"Recall@2           : "
        f"{recall_2 / total:.2%}"
    )

    print(
        f"Recall@5           : "
        f"{recall_5 / total:.2%}"
    )

    print(
        f"Average candidates : "
        f"{sum(candidate_counts) / len(candidate_counts):.2f}"
    )

    print("\nExamples where target was generated:")
    print("-" * 60)

    for raw, target, candidates in examples:

        print(
            f"{raw:12} -> "
            f"{target:12} | "
            f"{candidates}"
        )
# --------------------------------------------------
# 6. Edit-distance baseline
# --------------------------------------------------

def edit_distance(a, b):

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):

        current = [i]

        for j, char_b in enumerate(b, start=1):

            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char_a != char_b)

            current.append(
                min(insertion, deletion, substitution)
            )

        previous = current

    return previous[-1]


def generate_edit_candidates(word, vocabulary, max_distance=2):

    candidates = []

    for candidate in vocabulary:

        # Cheap length filter
        if abs(len(word) - len(candidate)) > max_distance:
            continue

        distance = edit_distance(word, candidate)

        if distance <= max_distance:
            candidates.append(candidate)

    return candidates


def edit_distance_evaluation():

    df = pd.read_csv(DATA_PATH)
    vocabulary = build_vocabulary()

    changed_pairs = []

    for raw_sentence, normalized_sentence in zip(
        df["Sentences"],
        df["Annotated by: Annotator 1"]
    ):
        raw_words = raw_sentence.split()
        normalized_words = normalized_sentence.split()

        for raw_word, normalized_word in zip(
            raw_words,
            normalized_words
        ):
            raw_word = clean_word(raw_word)
            normalized_word = clean_word(normalized_word)

            if not raw_word or not normalized_word:
                continue

            if raw_word == normalized_word:
                continue

            changed_pairs.append((raw_word, normalized_word))

    import random
    random.seed(42)
    sample_size = 2000

    if len(changed_pairs) > sample_size:
        sampled_pairs = random.sample(changed_pairs, sample_size)
    else:
        sampled_pairs = changed_pairs

    total = 0
    recall_1 = 0
    recall_5 = 0
    candidate_counts = []
    examples = []

    for raw_word, normalized_word in sampled_pairs:

        total += 1

        candidates = generate_edit_candidates(
            raw_word,
            vocabulary
        )

        # Rank by edit distance
        candidates.sort(
            key=lambda x: edit_distance(raw_word, x)
        )

        candidate_counts.append(len(candidates))

        if normalized_word in candidates[:1]:
            recall_1 += 1

        if normalized_word in candidates[:5]:
            recall_5 += 1

        if normalized_word in candidates and len(examples) < 20:
            examples.append(
                (
                    raw_word,
                    normalized_word,
                    candidates[:10]
                )
            )

    print("\n=== EDIT-DISTANCE BASELINE ===")
    print(f"Sample size        : {len(sampled_pairs):,}")
    print(f"Random seed        : 42")
    if total > 0:
        print(f"Recall@1           : {recall_1 / total:.2%}")
        print(f"Recall@5           : {recall_5 / total:.2%}")
        print(
            f"Average candidates : "
            f"{sum(candidate_counts) / len(candidate_counts):.2f}"
        )
    else:
        print("No candidates found.")

    print("\nExamples:")
    print("-" * 60)

    for raw, target, candidates in examples:

        print(
            f"{raw:12} -> "
            f"{target:12} | "
            f"{candidates}"
        )


def pragmatic_baseline_sample():

    df = pd.read_csv(DATA_PATH)
    sentences = df["Sentences"].dropna().tolist()
    
    import random
    random.seed(42)
    sample_size = 50
    if len(sentences) > sample_size:
        sampled_sentences = random.sample(sentences, sample_size)
    else:
        sampled_sentences = sentences
        
    print("\n=== PRAGMATIC EXTRACTION BASELINE ===")
    
    for sentence in sampled_sentences:
        result = extract_pragmatics(sentence)
        print(f"\nRaw Input    : {sentence}")
        print(f"Base Content : {result['content_after_pragmatic_extraction']}")
        print(f"Annotations  : {result['pragmatic_annotations']}")
        print(f"Trace        : {result['trace']}")


if __name__ == "__main__":

    sanity_checks()
    dataset_coverage()
    rule_coverage()
    collision_analysis()
    candidate_evaluation()
    edit_distance_evaluation()
    pragmatic_baseline_sample()