### Top-k retrieval under simulated OCR noise

| err/char | n | Top-1 | Top-3 | Top-5 |
|---:|---:|---:|---:|---:|
| 0.2 | 60 | 98.3% | 100.0% | 100.0% |

### LASA pair flagging vs ISMP/FDA ground truth

| rule | err/char | TP | FN | FP | TN | Precision | Recall | F1 | FP-rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RED | 0.2 | 60 | 268 | 2 | 198 | 96.8% | 18.3% | 30.8% | 1.0% |
| RED+YELLOW | 0.2 | 321 | 7 | 31 | 169 | 91.2% | 97.9% | 94.4% | 15.5% |

### Signal ablation (error rate 0.2)

| variant | Top-1 | Top-3 | Partner surfaced |
|---|---:|---:|---:|
| edit only | 100.0% | 100.0% | 99.7% |
| phonetic only | 90.0% | 100.0% | 87.5% |
| ngram only | 96.7% | 100.0% | 92.4% |
| fused (0.45/0.35/0.20) | 98.3% | 100.0% | 97.9% |