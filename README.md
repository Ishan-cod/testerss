# Business Entity Resolution

## Environment
- Python 3.14.7 (required — see compatibility matrix below)
- pandas 3.0.6, numpy 2.5.2, rapidfuzz 3.14.5
- scikit-learn 1.9.1, lightgbm 4.7.0
- indic-transliteration 2.3.82 (optional)
- tqdm 4.70.1

### Compatibility matrix
| Package | Min Python | Note |
|---|---|---|
| pandas 3.0 | 3.11+ | Major release |
| numpy 2.5 | 3.12–3.14 | Drops 3.11 |
| lightgbm 4.7 | 3.10+ | Bumped minimum |
| scikit-learn 1.9 | 3.10+ | — |

**Intersection: Python 3.12–3.14.** Use 3.14.7.

## Reproduce
```bash
pip install -r requirements.txt
python -m src.main