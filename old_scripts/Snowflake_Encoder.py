from pathlib import Path
from typing import Union, Sequence

import torch
from fast_forward.encoder import Encoder
from transformers import AutoTokenizer, AutoModel


class SnowflakeDocumentEncoder(Encoder):

    def __init__(self, model: Union[str, Path] = 'Snowflake/snowflake-arctic-embed-m', device: str = 'cpu',
                 **tokenizer_args) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, add_pooling_layer=False)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.tokenizer_args = tokenizer_args

    def encode(self, documents: Sequence[str]) -> torch.Tensor:
        document_tokens = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=512)

        with torch.no_grad():
            document_embeddings = self.model(**document_tokens)[0][:, 0]

        document_embeddings = torch.nn.functional.normalize(document_embeddings, p=2, dim=1)

        return document_embeddings

    def __call__(self, documents: Sequence[str]) -> torch.Tensor:
        return self.encode(documents)


class SnowflakeQueryEncoder(SnowflakeDocumentEncoder):
    def __call__(self, queries: Sequence[str]) -> torch.Tensor:
        query_prefix = 'Represent this sentence for searching relevant passages: '
        queries_with_prefix = ["{}{}".format(query_prefix, i) for i in queries]
        return self.encode(queries_with_prefix)
