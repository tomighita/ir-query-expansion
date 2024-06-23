from pathlib import Path

import ir_datasets
import pyterrier as pt
import torch
from fast_forward import OnDiskIndex, Mode
from fast_forward.encoder import TCTColBERTQueryEncoder, TCTColBERTDocumentEncoder
from fast_forward.util.pyterrier import FFInterpolate
from fast_forward.util.pyterrier import FFScore
from pyterrier.measures import RR, nDCG, MAP

if not pt.started():
    pt.init(
        tqdm="notebook",
        boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"]
    )

DATASET_NAME = "beir/arguana"
TESTSET_NAME = "beir/arguana"
BM25_INDEX_PATH = 'indices/arguana'
INDEX_PATH = 'indexes/irds:beir_arguana.h5'
FIELDS = ["text"]

SHOULD_RUN_GRID = False
DEVSET_NAME = "irds:beir/dbpedia-entity/dev"

dataset = pt.get_dataset('irds:' + DATASET_NAME)
ir_ds = ir_datasets.load(DATASET_NAME)

idx_path = Path(BM25_INDEX_PATH).absolute()

if not idx_path.exists():
    index_ref = pt.index.IterDictIndexer(
        str(idx_path),
        blocks=True,
        meta={'docno': ir_ds.docs_metadata()['fields']['doc_id']['max_len'] },
        # stopwords=None,
        # stemmer=None,
    ).index(dataset.get_corpus_iter(), fields=FIELDS)

index = pt.IndexFactory.of(str(idx_path))

bm25 = pt.BatchRetrieve(index, wmodel="BM25")
rm3 = pt.rewrite.RM3(index)
# rm3 = pt.rewrite.RM3(index, fb_terms = 5, fb_lambda = 0.9, fb_docs = 2)
testset = pt.get_dataset('irds:' + TESTSET_NAME)

q_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")
d_encoder = TCTColBERTDocumentEncoder(
    "castorini/tct_colbert-msmarco",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)
ff_index = OnDiskIndex.load(
    Path(INDEX_PATH), query_encoder=q_encoder, mode=Mode.MAXP
)

ff_index = ff_index.to_memory()
ff_score = FFScore(ff_index)
candidates = (bm25 % 5)(testset.get_topics('text')) # Get the candidates
re_ranked = ff_score(candidates)
ff_int = FFInterpolate(alpha=0.05)
ff_int(re_ranked)

if SHOULD_RUN_GRID:
    devset = pt.get_dataset(DEVSET_NAME)
    pt.GridSearch(
        ~bm25 % 100 >> ff_score >> ff_int,
        {ff_int: {"alpha": [0.05, 0.1, 0.5, 0.9]}},
        devset.get_topics(),
        devset.get_qrels(),
        "map",
        verbose=True,
    )

result = pt.Experiment(
    [
        bm25,
        bm25 % 5 >> rm3 >> bm25,
        bm25 % 1000 >> ff_score >> ff_int,
        bm25 % 5 >> rm3 >> bm25 % 1000 >> pt.rewrite.reset() >> ff_score >> ff_int,
    ],
    testset.get_topics('text'),
    testset.get_qrels(),
    eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100],
    names=[
        "BM25",
        "RM3",
        "TCT-ColBERT",
        "RM3+TCT-ColBERT"
    ]
)

print(result)