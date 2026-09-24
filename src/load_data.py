import pandas as pd
from . import config


def _read(path):
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    return df


def load_train():
    s1 = _read(config.TRAIN_S1)
    s2 = _read(config.TRAIN_S2)
    s3 = _read(config.TRAIN_S3)
    gt = _read(config.TRAIN_GT)
    return s1, s2, s3, gt


def load_test():
    s1 = _read(config.TEST_S1)
    s2 = _read(config.TEST_S2)
    s3 = _read(config.TEST_S3)
    return s1, s2, s3


def parse_ground_truth(gt: pd.DataFrame):
    """
    Returns dict: {source1_entity_id: set(matched_ids)}
    Empty string means no matches.
    """
    out = {}
    for _, row in gt.iterrows():
        s1 = row["source1_entity_id"]
        raw = row.get("matched_entity_ids", "") or ""
        ids = {x.strip() for x in raw.split(",") if x.strip()}
        out[s1] = ids
    return out