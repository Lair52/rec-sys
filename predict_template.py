import os
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from typing import List, Optional

from pydantic import BaseModel


# This is the schema for the prediction request.
# Note you do not need to use all the fields listed.
# For example if your predictor only needs data related to the content
# then you can ignore the topic fields.
class TopicPredictionRequest(BaseModel):
    content_title: Optional[str] = None
    content_description: Optional[str] = None
    content_kind: Optional[str] = None
    content_text: Optional[str] = None
    topic_title: Optional[str] = None
    topic_description: Optional[str] = None
    topic_category: Optional[str] = None


class TopicPredictor:
    """
    A predictor that matches content to the most relevant topics using
    sentence embeddings and a FAISS similarity index.

    This class loads the embedding model, the precomputed FAISS index of
    topic vectors, and topic metadata. It receives a TopicPredictionRequest
    and returns a ranked list of topic IDs based on semantic similarity.
    """

    def __init__(self, vector_db_path: str = None, df_topics_path: str = None):
        """
        Initialize the TopicPredictor by loading the embedding model,
        the FAISS vector index, and the topics metadata.

        Args:
            vector_db_path (str, optional): Path to the FAISS index file.
            df_topics_path (str, optional): Path to the topics CSV file.

        """

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        base_path = os.path.dirname(__file__)

        self.vector_db_path = vector_db_path or os.path.join(
            base_path, "vector_db_v1.bin"
        )
        self.df_topics_path = df_topics_path or os.path.join(
            base_path,
            "data",
            "learning-equality-curriculum-recommendations",
            "topics.csv",
        )

        self.vector_db = faiss.read_index(self.vector_db_path)
        self.df_topics = pd.read_csv(self.df_topics_path)

    def predict(self, request: TopicPredictionRequest) -> List[str]:
        """Takes in the request and can use all or some subset of input parameters to
        predict the topics associated with a given piece of content.

        Args:
            request (TopicPredictionRequest): See class TopicPredictionRequest

        Returns:
            List[str]: Should be list of topic ids.
        """
        # ADD IN CODE TO MAKE PREDICTIONS
        predictions = []
        try:
            full_info = self._build_full_info_content(request)
            cont_vec = self._encode(full_info)

            # If you want to see the cosine similarity of each recommended item,
            # use predictions_with_score
            predictions, predictions_with_score = self._get_similars(cont_vec)

        except Exception as e:
            print(f"ERROR:{e}")

        return predictions

    def _build_full_info_content(self, request: TopicPredictionRequest) -> str:
        """
        Build a single text string for the content using title, description
        and text fields from the request. Fields that are None are ignored.
        Args:
            request (TopicPredictionRequest):
                An object containing textual fields related to the content
                (title, description, text, etc.). Only the content-related
                fields are used; topic fields are ignored.

        Returns:
            str: Consolidated content text for embedding.
        """
        parts = []
        if request.content_title:
            parts.append(f"title: {request.content_title}")
        if request.content_description:
            parts.append(f"description: {request.content_description}")
        if request.content_text:
            parts.append(f"text: {request.content_text}")

        full_info = " ".join(parts)
        if not full_info.strip():
            raise ValueError(
                "The content must contain a least one of the following data: title, description or text."
            )
        return full_info

    def _encode(self, full_info: str) -> np.ndarray:
        """
        Generate a normalized embedding vector for the given text using
        the SentenceTransformer model.
        Args:
            full_info (str):
                Consolidated text representation of the content, typically produced
                by `_build_full_info_content`.

        Returns:
            np.ndarray: 2D L2-normalized embedding suitable for FAISS search.

        Raises:
            ValueError:
                If `full_info` is empty or contains only whitespace, since the model
                cannot encode an empty string reliably.
        """
        vec = self.model.encode(full_info)
        vec = np.asarray(vec, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        return vec

    def _get_similars(self, cont_vec: np.ndarray, k: int = 10) -> tuple:
        """
        Retrieve the top-k most similar topics from the FAISS index based
        on the provided content embedding.
        Args:
            cont_vec (np.ndarray):
                A 2D L2-normalized vector of shape (1, embedding_dim) representing
                the encoded content.
                This should be produced by `_encode`.

            k (int, optional):
                Number of nearest neighbors to retrieve. Defaults to 10.

        Returns:
            (List[str], List[dict]): Topic IDs and similarity scores.
        """
        D, I = self.vector_db.search(cont_vec, k)
        topic_ids = []
        topics_with_score = []
        for i, line in enumerate(I[0]):
            topic_ids.append(self.df_topics.iloc[line]["id"])
            temp = {
                "topic_id": self.df_topics.iloc[line]["id"],
                "cos_similarity": round(float(D[0][i]), 4),
            }
            topics_with_score.append(temp)
        return topic_ids, topics_with_score
