---
Date: 17 Jun
Measurements: nDCG@10, RR, AP
Dataset: beir/scifact/test
Avg_Q_Length: "13.05"
---
alpha: 0.1

|                 | RR@10    | nDCG@10  | AP@100   |
| :-------------- | :------- | :------- | :------- |
| BM25            | 0.632427 | 0.672167 | 0.626749 |
| RM3             | 0.562431 | 0.622227 | 0.559660 |
| TCT-ColBERT     | 0.652175 | 0.686199 | 0.644616 |
| RM3+TCT-ColBERT | 0.641512 | 0.680644 | 0.632925 |

