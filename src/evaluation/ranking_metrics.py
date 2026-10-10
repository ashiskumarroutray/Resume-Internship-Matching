def precision_at_k(ranked_items, relevant_items, k):
    """
    Fraction of the top-k ranked items that are relevant.

    ranked_items: IDs in predicted ranking order.
    relevant_items: Set of relevant IDs.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not ranked_items:
        return 0.0

    top_k = ranked_items[:k]

    relevant_count = sum(
        1 for item in top_k if item in relevant_items
    )

    return relevant_count / len(top_k)


def recall_at_k(ranked_items, relevant_items, k):
    """
    Fraction of all known relevant items found in the top k.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_items:
        return 0.0

    top_k = ranked_items[:k]

    relevant_count = sum(
        1 for item in top_k if item in relevant_items
    )

    return relevant_count / len(relevant_items)


def reciprocal_rank(ranked_items, relevant_items):
    """
    Reciprocal rank of the first relevant result.
    """

    for rank, item in enumerate(ranked_items, start=1):
        if item in relevant_items:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(rankings, relevance_sets):
    """
    Mean Reciprocal Rank across multiple queries.

    rankings: List of ranked ID lists.
    relevance_sets: Corresponding sets of relevant IDs.
    """

    if len(rankings) != len(relevance_sets):
        raise ValueError(
            "Each ranking must have a corresponding relevance set"
        )

    if not rankings:
        return 0.0

    scores = [
        reciprocal_rank(ranking, relevant)
        for ranking, relevant in zip(rankings, relevance_sets)
    ]

    return sum(scores) / len(scores)


if __name__ == "__main__":

    # Illustrative example only—not measured project performance.
    ranked_ids = ["job_3", "job_1", "job_4", "job_2"]
    relevant_ids = {"job_1", "job_2"}

    print("Precision@3:", precision_at_k(
        ranked_ids, relevant_ids, 3
    ))

    print("Recall@3:", recall_at_k(
        ranked_ids, relevant_ids, 3
    ))

    print("Reciprocal Rank:", reciprocal_rank(
        ranked_ids, relevant_ids
    ))

    print("MRR:", mean_reciprocal_rank(
        [ranked_ids],
        [relevant_ids],
    ))