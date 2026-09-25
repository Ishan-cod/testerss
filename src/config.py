import os

# BASE_DIR = os.getenv(
#     "BASE_DIR",
#     "/home/emiliatan/RemSpace/MLCHALLANGE/6ab10eb3b23ba_student_resource/student_resource"
# )

# NON_BASE_DIR=os.getenv(
#     "BASE_DIR",
#     "/home/emiliatan/RemSpace/codeareana/ML_Challenge"
# )

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR  = os.path.join(BASE_DIR, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_DIR  = os.path.join(BASE_DIR, "cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
TEST_S1  = os.path.join(TEST_DIR,  "test_source1.tsv")
TEST_S2  = os.path.join(TEST_DIR,  "test_source2.tsv")
TEST_S3  = os.path.join(TEST_DIR,  "test_source3.tsv")

OUT_MATCH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUT_CAND  = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# =====================================================================
# SAMPLING MODE
#   True  → fits 16 GB laptop, uses ~7 GB peak. For development.
#   False → full pipeline. Requires ≥ 30 GB RAM (Kaggle / Colab Pro).
# =====================================================================
SAMPLE_MODE = False

TRAIN_S1_SAMPLE = 100_000     # S1 entities used for training
TEST_S1_SAMPLE  = 100_000      # S1 entities used for dev-time inference
S23_SAMPLE = 1_500_000    # max S2+S3 rows kept (per source) in SAMPLE_MODE

# Chunking — keep small on laptop
S1_CHUNK = 10_000
FUZZY_BATCH = 50_000
MAX_COMMON_KEY_FREQ = 1_000
MAX_CANDIDATES_PER_S1 = 30

RANDOM_STATE = 42
VAL_FRACTION = 0.10

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
    n_jobs=-1,
)