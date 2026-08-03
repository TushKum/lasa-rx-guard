| Model | Split | CER ↓ | WER ↓ | Exact ↑ | n |
|---|---|---:|---:|---:|---:|
| CRNN (greedy) | test_seen | 2.91% | 18.25% | 81.75% | 400 |
| CRNN (greedy) | test_unseen_font | 19.45% | 70.25% | 29.75% | 400 |
| CRNN (greedy) | test_unseen_word | 2.83% | 19.75% | 80.25% | 400 |
| CRNN (greedy) | test_hard | 22.71% | 74.75% | 25.25% | 400 |
| CRNN (beam) | test_seen | 2.80% | 17.75% | 82.25% | 400 |
| CRNN (beam) | test_unseen_font | 19.09% | 70.25% | 29.75% | 400 |
| CRNN (beam) | test_unseen_word | 2.85% | 19.75% | 80.25% | 400 |
| CRNN (beam) | test_hard | 22.46% | 74.50% | 25.50% | 400 |
| CRNN (lexicon) | test_seen | 0.48% | 1.25% | 98.75% | 400 |
| CRNN (lexicon) | test_unseen_font | 5.23% | 9.00% | 91.00% | 400 |
| CRNN (lexicon) | test_unseen_word | 0.08% | 0.25% | 99.75% | 400 |
| CRNN (lexicon) | test_hard | 8.16% | 13.00% | 87.00% | 400 |