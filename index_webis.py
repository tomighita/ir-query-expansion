import pyterrier as pt
from fast_forward.encoder import TCTColBERTQueryEncoder, TCTColBERTDocumentEncoder
import torch
from fast_forward import OnDiskIndex, Mode, Indexer
from pathlib import Path
import sys

if len(sys.argv) <= 1:
    exit(-1)

# beir/webis-touche2020/v2
dataset_name = sys.argv[1]
index_name = dataset_name.replace('/', '_')

if not pt.started():
    pt.init(
        tqdm="notebook",
        boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"]
    )

q_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")
d_encoder = TCTColBERTDocumentEncoder(
    "castorini/tct_colbert-msmarco",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)
dataset = pt.get_dataset(dataset_name)

should_reindex = False

ff_index = OnDiskIndex(
    Path(f'{index_name}.h5'), dim=768, query_encoder=q_encoder, mode=Mode.MAXP, max_id_length=40
)


def docs_iter():
    for d in dataset.get_corpus_iter():
        yield {"doc_id": d["docno"], "text": d["text"]}


ff_indexer = Indexer(ff_index, d_encoder, batch_size=8)
ff_indexer.index_dicts(docs_iter())
