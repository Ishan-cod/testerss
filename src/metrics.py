def f05_single(true_set, pred_set):
    if not true_set and not pred_set: return 1.0
    if not true_set or not pred_set:  return 0.0
    tp = len(true_set & pred_set)
    if tp == 0: return 0.0
    p = tp / len(pred_set)
    r = tp / len(true_set)
    return (1.25 * p * r) / (0.25 * p + r)


def macro_f05(truth_map, pred_map, all_s1_ids):
    scores = []
    for s1 in all_s1_ids:
        scores.append(f05_single(truth_map.get(s1, set()), pred_map.get(s1, set())))
    return sum(scores) / len(scores) if scores else 0.0