import os

# ----------------------------
# Paths
# ----------------------------

# Set DATASET_DIR externally when running on Colab, e.g.:
# /content/drive/MyDrive/ML_CHALLENGE_DATA/dataset
#
# Otherwise, use the original local project-relative dataset path.

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DATASET_DIR = os.getenv(
    "DATASET_DIR",
    os.path.join(BASE_DIR, "dataset")
)

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR  = os.path.join(DATASET_DIR, "test")

# Output/cache can also be redirected externally on Colab.
OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    os.path.join(BASE_DIR, "output")
)

CACHE_DIR = os.getenv(
    "CACHE_DIR",
    os.path.join(BASE_DIR, "cache")
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1  = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2  = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3  = os.path.join(TEST_DIR, "test_source3.tsv")

OUT_MATCH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUT_CAND  = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")


RANDOM_STATE = 42
VAL_FRACTION = 0.10
S1_CHUNK     = 50_000
FEATURE_BATCH = 100_000
MAX_COMMON_KEY_FREQ = 5_000
MAX_CANDIDATES_PER_S1 = 200

LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    n_estimators=800,
    verbose=-1,
    random_state=RANDOM_STATE,
)