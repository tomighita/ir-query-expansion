from pyterrier.measures import RR, nDCG, MAP
from pathlib import Path

import ir_datasets
import pyterrier as pt

if not pt.started():
    pt.init(
        tqdm="notebook",
        boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"]
    )

DATASET_NAME = "msmarco-passage/trec-dl-2019"
TESTSET_NAME = "msmarco-passage/trec-dl-2019/judged"
BM25_INDEX_PATH = 'indices/msmarco'
INDEX_PATH = 'indexes/ff_msmarco-v1-passage.tct_colbert.h5'
FIELDS = ["text"]

SHOULD_RUN_GRID = True
DEVSET_NAME = "irds:msmarco-passage/train"

dataset = pt.get_dataset('irds:' + DATASET_NAME)
ir_ds = ir_datasets.load(DATASET_NAME)

idx_path = Path(BM25_INDEX_PATH).absolute()

try:
    index_ref = pt.index.IterDictIndexer(
        str(idx_path),
        blocks=True,
        meta={'docno': ir_ds.docs_metadata()['fields']['doc_id']['max_len']},
    ).index(dataset.get_corpus_iter(), fields=FIELDS)
except:
    print("Index already exists")

index = pt.IndexFactory.of(str(idx_path))

bm25 = pt.BatchRetrieve(index, wmodel="BM25")
rm3 = pt.rewrite.RM3(index)
# rm3 = pt.rewrite.RM3(index, fb_terms = 5, fb_lambda = 0.9, fb_docs = 2)
testset = pt.get_dataset('irds:' + TESTSET_NAME)

print("Starting grid search...")
devset = pt.get_dataset(DEVSET_NAME)
tune_pipe = pt.GridSearch(
    bm25 % 5 >> rm3 >> bm25 % 100,
    {rm3: {"fb_docs": [3, 5, 7, 10], "fb_terms": [3, 5, 10, 15]}},
    devset.get_topics("text"),
    devset.get_qrels(),
    verbose=True,
)
res = pt.Experiment([tune_pipe], testset.get_topics('text'), testset.get_qrels(), eval_metrics=[RR @ 10])

print(rm3.fb_docs)
print(rm3.fb_terms)
