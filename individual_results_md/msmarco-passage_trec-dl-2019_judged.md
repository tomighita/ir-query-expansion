---
Date: 22 May
Measurements: nDCG@10, RR(rel=2), AP(rel=2)
Dataset: msmarco-passage/trec-dl-2019/judged
---
|                 | RR\(rel=2\)@10 | nDCG@10  | AP\(rel=2\)@100 |
| :-------------- | :------------- | :------- | :-------------- |
| BM25            | 0.639655       | 0.479540 | 0.232165        |
| RM3             | 0.606681       | 0.515595 | 0.251896        |
| TCT-ColBERT     | 0.807752       | 0.692802 | 0.388712        |
| RM3+TCT-ColBERT | 0.857364       | 0.718088 | 0.403819        |
