import os
from typing import Annotated, Any, List, Optional, TypeVar, Union
import chromadb
from dotenv import load_dotenv
from distilabel.steps import Step, StepInput, GeneratorStep
from pydantic import BaseModel, Field, PrivateAttr

_T = TypeVar("_T")
_RUNTIME_PARAMETER_ANNOTATION = "distilabel_step_runtime_parameter"
RuntimeParameter = Annotated[
    Union[_T, None], Field(default=None), _RUNTIME_PARAMETER_ANNOTATION
]

class GetTopkDocs(Step):
    k: RuntimeParameter[int] = 10
    collectionName: str
    _collection: Any = None

    def load(self):
        load_dotenv()
        client = chromadb.PersistentClient(path=os.getenv('CHROMA_PATH'))
        self._collection = client.get_collection(name=self.collectionName)
        super().load()

    @property
    def inputs(self) -> List[str]:
        return ["query_embedding"]
    
    @property
    def outputs(self) -> List[str]:
        return ["embeddings", "documents", "metadatas"]
    
    def process(self, *inputs: StepInput):
        for batch in inputs:
            result = []
            query_res = self._collection.query(
                query_embeddings=[row["query_embedding"] for row in batch],
                n_results=self.k,
                include=["embeddings", "metadatas", "documents"] 
            )
            for i, row in enumerate(batch):
                result.append(row | {
                    "embeddings": query_res["embeddings"][i],
                    "documents": query_res["documents"][i],
                    "metadatas": query_res["metadatas"][i],
                }) 
            yield result