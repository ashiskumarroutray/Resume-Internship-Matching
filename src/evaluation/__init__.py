def precision_at_k(relevant_items, predicted_items, k):
    """
    Precision@K
    """

    predicted = predicted_items[:k]

    if not predicted:
        return 0.0

    relevant = set(relevant_items)
    predicted = set(predicted)

    return len(relevant.intersection(predicted)) / len(predicted)


def recall_at_k(relevant_items, predicted_items, k):
    """
    Recall@K
    """

    predicted = predicted_items[:k]

    if not relevant_items:
        return 0.0

    relevant = set(relevant_items)
    predicted = set(predicted)

    return len(relevant.intersection(predicted)) / len(relevant)


def reciprocal_rank(relevant_items, predicted_items):
    """
    Reciprocal Rank
    """

    relevant = set(relevant_items)

    for rank, item in enumerate(predicted_items, start=1):

        if item in relevant:
            return 1 / rank

    return 0.0


def display_ranking_metrics(
    relevant_items,
    predicted_items
):
    print("\n" + "=" * 70)
    print("RANKING EVALUATION")
    print("=" * 70)

    for k in [1, 3, 5]:

        precision = precision_at_k(
            relevant_items,
            predicted_items,
            k
        )

        recall = recall_at_k(
            relevant_items,
            predicted_items,
            k
        )

        print(
            f"\nPrecision@{k}: "
            f"{precision * 100:.2f}%"
        )

        print(
            f"Recall@{k}: "
            f"{recall * 100:.2f}%"
        )

    rr = reciprocal_rank(
        relevant_items,
        predicted_items
    )

    print(
        f"\nReciprocal Rank: "
        f"{rr:.4f}"
    )