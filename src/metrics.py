def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def f05_single(true_set: set, pred_set: set) -> float:
    # Both empty → perfect singleton
    if not true_set and not pred_set:
        return 1.0
    # One empty and other not → worst case
    if not true_set or not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    return f_beta(precision, recall, beta=0.5)


def macro_f05(truth: dict, preds: dict, all_s1_ids) -> float:
    """
    truth: {s1_id: set(matched_ids)}  (only matched entities; missing = empty)
    preds: {s1_id: set(predicted_ids)}
    all_s1_ids: iterable of ALL S1 IDs to include in the average
    """
    scores = []
    for s1 in all_s1_ids:
        t = truth.get(s1, set())
        p = preds.get(s1, set())
        scores.append(f05_single(t, p))
    return sum(scores) / len(scores) if scores else 0.0