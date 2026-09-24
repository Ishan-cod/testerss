import os
import numpy as np

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use DATASET_DIR from environment if provided,
# otherwise use the local dataset location.
DATASET_DIR = os.getenv(
    "DATASET_DIR",
    "/home/emiliatan/RemSpace/MLCHALLANGE/6ab10eb3b23ba_student_resource/student_resource/dataset"
)

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

OUT_MATCH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUT_CAND = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")


# ----------------------------
# Model / Training
# ----------------------------
RANDOM_STATE = 42
VAL_FRACTION = 0.20

LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    n_estimators=600,
    verbose=-1,
    random_state=RANDOM_STATE,
)

THRESHOLD_GRID = [
    round(x, 3)
    for x in np.arange(0.05, 0.96, 0.025)
]

MAX_CANDIDATES_PER_S1 = 500