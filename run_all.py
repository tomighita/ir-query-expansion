import prebuilt_index_info
from urllib.request import urlretrieve
import tarfile
import faiss
import os
from fast_forward import OnDiskIndex
import shutil
import pyterrier as pt
from pathlib import Path
from pyterrier.measures import RR, nDCG, MAP
import re
from fast_forward.encoder import TCTColBERTQueryEncoder, TCTColBERTDocumentEncoder
import torch
from fast_forward import OnDiskIndex, Mode
from fast_forward.util.pyterrier import FFScore
from fast_forward.util.pyterrier import FFInterpolate

from database_converter import CONVERTER

DIR_NAME = 'indexes'
REMOVE_AFTER_DOWNLOAD = True
INTERESTING_DATASETS = CONVERTER.keys()

def _remove_pollution(q) -> str:
    q_old = q["query"].replace('applypipeline:off', '')
    return q["query_1"] + " " + re.sub(r'\^(\d)+\.(\d)+', '', q_old)

if not pt.started():
    pt.init(
        tqdm="auto",
        boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"]
    )

for dataset_name in prebuilt_index_info.FAISS_INDEX_INFO_BEIR.keys():
    # placeholder
    if dataset_name not in INTERESTING_DATASETS:
        continue

    dataset_url = prebuilt_index_info.FAISS_INDEX_INFO_BEIR[dataset_name]['urls'][0]
    faiss_idx_filename = prebuilt_index_info.FAISS_INDEX_INFO_BEIR[dataset_name]['filename']
    faiss_dir_name = faiss_idx_filename[:-7]
    faiss_dir_full_path = os.path.join(DIR_NAME, faiss_dir_name)

    if not os.path.exists(Path(DIR_NAME, faiss_dir_name + ".h5")) and not os.path.exists(faiss_dir_full_path):
        print(f'Downloading archive: {faiss_idx_filename}')
        # Download archive
        urlretrieve(dataset_url, faiss_idx_filename)
        # Un-archive the .tar.gz
        archive_faiss = tarfile.open(faiss_idx_filename)
        archive_faiss.extractall(DIR_NAME)
        archive_faiss.close()
        # Delete archive
        os.remove(faiss_idx_filename)

        print("Indexing...")
        index = faiss.read_index(os.path.join(DIR_NAME, faiss_dir_name, "index"))
        with open(os.path.join(DIR_NAME, faiss_dir_name, "docid")) as fp:
            docids = list(fp.read().splitlines())

        vectors = index.reconstruct_n(0, len(docids))
        OnDiskIndex(Path(DIR_NAME, faiss_dir_name + ".h5"), 768, max_id_length=60).add(vectors, doc_ids=docids)

        print(f'Finished indexing {dataset_name}.')

    if os.path.exists(faiss_dir_full_path) and REMOVE_AFTER_DOWNLOAD:
        shutil.rmtree(os.path.join(DIR_NAME, faiss_dir_name))

    # Get database
    dataset = pt.get_dataset(CONVERTER[dataset_name]['DATASET_NAME'])

    idx_path = Path(CONVERTER[dataset_name]['BM25_INDEX_PATH']).absolute()

    if not os.path.exists(idx_path):
        index_ref = pt.index.IterDictIndexer(
            str(idx_path),
            blocks=True,
            meta={'docno': 60},
            # stopwords=None,
            # stemmer=None,
        ).index(dataset.get_corpus_iter(), fields=["text"])

    index = pt.IndexFactory.of(str(idx_path))

    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    rm3 = pt.rewrite.RM3(index)
    testset = pt.get_dataset(CONVERTER[dataset_name]['TESTSET_NAME'])

    q_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")
    d_encoder = TCTColBERTDocumentEncoder(
        "castorini/tct_colbert-msmarco",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    ff_index = OnDiskIndex.load(
        Path(DIR_NAME, faiss_dir_name + ".h5"), query_encoder=q_encoder, mode=Mode.MAXP
    )
    ff_index = ff_index.to_memory()

    ff_score = FFScore(ff_index)
    candidates = (bm25 % 5)(testset.get_topics('text'))  # Get the candidates
    re_ranked = ff_score(candidates)
    ff_int = FFInterpolate(alpha=0.1)
    ff_int(re_ranked)

    pipeline = bm25 % 5 >> rm3 >> pt.apply.query(_remove_pollution) >> bm25 % 1000

    pipeline(testset.get_topics('text'))

    print(f'Running experiment on dataset: {dataset_name}')

    rez = pt.Experiment(
        [
            bm25,
            bm25 >> rm3 >> bm25,
            bm25 % 1 >> rm3 >> bm25,
            bm25 % 1000 >> ff_score >> ff_int,
            pipeline >> ff_score >> ff_int
        ],
        testset.get_topics('text'),
        testset.get_qrels(),
        eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100],
        names=[
            "BM25",
            "RM3",
            "RM3 % 1",
            "BM25 >> FF",
            "BM25 >> RM3 >> FF"
        ],
    )
