# Romanized Hindi -> phonetic units
#
# This is an experimental representation.
# It is NOT the final normalization system.

PHONETIC_UNITS = {
    # Aspirated consonants
    "chh": "CHH",
    "jh": "JH",
    "kh": "KH",
    "gh": "GH",
    "th": "TH",
    "dh": "DH",
    "ph": "PH",
    "bh": "BH",

    # Other common consonant units
    "sh": "SH",

    # Long vowels
    "aa": "AA",
    "ee": "EE",
    "ii": "EE",
    "oo": "OO",
    "uu": "OO",

    # Diphthongs
    "ai": "AI",
    "ei": "AI",
    "au": "AU",
    "ou": "AU",
}


def phonetic_tokenize(word):
    """
    Convert a Romanized Hindi word into approximate phonetic units.
    """

    word = word.lower()

    units = []
    i = 0

    # Check longer patterns first.
    patterns = sorted(PHONETIC_UNITS, key=len, reverse=True)

    while i < len(word):
        matched = False

        for pattern in patterns:
            if word.startswith(pattern, i):
                units.append(PHONETIC_UNITS[pattern])
                i += len(pattern)
                matched = True
                break

        if not matched:
            units.append(word[i])
            i += 1

    return units


def equivalent_units(a, b, soft_equivalences=None):
    """
    Compare two phonetic-unit sequences.

    soft_equivalences lets us experimentally test whether
    two units should be treated as equivalent.

    IMPORTANT:
    This does NOT modify either representation.
    """

    if soft_equivalences is None:
        soft_equivalences = set()

    if len(a) != len(b):
        return False

    for x, y in zip(a, b):

        # Already identical
        if x == y:
            continue

        # Experimentally equivalent
        if (x, y) in soft_equivalences:
            continue

        return False

    return True