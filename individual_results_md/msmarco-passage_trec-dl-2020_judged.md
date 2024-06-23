---
Date: 22 May
Measurements: nDCG@10, RR(rel=2), AP(rel=2)
Dataset: msmarco-passage/trec-dl-2020/judged
---

|                 | RR\(rel=2\)@10 | nDCG@10  | AP\(rel=2\)@100 |
| :-------------- | :------------- | :------- | :-------------- |
| BM25            | 0.614675       | 0.493627 | 0.275282        |
| RM3             | 0.586464       | 0.504314 | 0.299899        |
| TCT-ColBERT     | 0.789506       | 0.686044 | 0.446461        |
| RM3+TCT-ColBERT | 0.796649       | 0.691352 | 0.457449        |
![[Screenshot 2024-06-18 at 12.53.47.png]]