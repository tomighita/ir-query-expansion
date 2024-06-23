---
Date: 27 May
Measurements: nDCG@10, RR(rel=2), AP(rel=2)
Dataset: msmarco-passage/trec-dl-hard
---

|                 | RR\(rel=2\)@10 | nDCG@10  | AP\(rel=2\)@100 |
| :-------------- | :------------- | :------- | :-------------- |
| BM25            | 0.415056       | 0.274333 | 0.135798        |
| RM3             | 0.389222       | 0.270870 | 0.143734        |
| TCT-ColBERT     | 0.531222       | 0.394371 | 0.221447        |
| RM3+TCT-ColBERT | 0.572056       | 0.404091 | 0.230071        |
