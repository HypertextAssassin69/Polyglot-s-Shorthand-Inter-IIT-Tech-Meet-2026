def score_candidate(candidate, context, word_frequency):
    """
    Simple baseline candidate scorer.

    Higher-frequency words receive a higher score.
    Context is accepted now but not used yet.
    """

    return word_frequency.get(candidate, 0)


def rank_candidates(candidates, context, word_frequency):
    """
    Rank candidates from most to least frequent.
    """

    scored = [
        (candidate, score_candidate(candidate, context, word_frequency))
        for candidate in candidates
    ]

    scored.sort(key=lambda x: x[1], reverse=True)

    return scored