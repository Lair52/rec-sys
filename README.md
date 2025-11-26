# Overview

This solution follows a recommender-based strategy, where the goal is to retrieve the most semantically relevant topics for each content item rather than train a traditional multi-label classifier. Both contents and topics are encoded using the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) Sentence Transformer, producing 384-dimensional embeddings that capture their semantic meaning. The topic embeddings are stored in [FAISS](https://github.com/facebookresearch/faiss), which acts as a high-performance vector database capable of fast cosine-similarity search at scale. For prediction, the system receive the content embedding and return the most similar topics thorugh a vector search based on cosine similarity.

This approach was chosen because it generalizes naturally to unseen data, avoids the need for supervised training, and relies solely on the textual fields provided in the dataset. It is also computationally efficient, easy to maintain, and aligns with modern retrieval-based architectures widely used in industry for recommendation, search, and matching task.


### How the Model Works

- Text Preprocessing: 
Only three fields are used: title, description, and text, as they contain the most meaningful information for each content item.

- Embedding Generation: 
The model encodes content and topic texts into 384-dimensional vectors using all-MiniLM-L6-v2.

- Vector Normalization: 
All embeddings are L2-normalized so FAISS can use inner product = cosine similarity.

- FAISS Indexing: 
Topic embeddings are stored in a IndexFlatIP index for fast nearest-neighbor search.

- Prediction:
A new content item is encoded and queried against the FAISS index to retrieve the top-k most similar topics.

# Environment Setup

Follow the steps below to create a clean and reproducible environment for running the project.

## 1. Install Git LFS (required for large files)

Some project files, such as vector databases and datasets, are managed using Git LFS (Large File Storage). 

Before cloning the repository, install Git LFS:

- **Linux**:
  ```bash
  sudo apt-get install git-lfs
  ```
- **macOS** :
  ```bash
  brew install git-lfs
  ```
- **Windows (PowerShell):** :
  ```bash
  choco install git-lfs
  ```
Then, initialize Git LFS by running:
   ```bash
    git lfs install
   ```

## 2. Create and activate a Python virtual environment

Create a Python virtual environment to isolate the project's dependencies:

- **Linux / macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

- **Windows (PowerShell)**:
```bash
venv\Scripts\Activate.ps1
```
## 3. Install dependencies

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## 4. :warning: Unzip the dataset files 

This repository includes a data.zip file containing both the original dataset and the processed files used in the project. Although storing raw data in a repo is not common, it is necessary here to ensure that all notebooks and the TopicPredictor run without path errors.
After installing the dependencies, extract the dataset by running the provided script:
```bash
python3 unzip_data.py
```

This will unpack data.zip into the appropriate project directory structure.


# Code Access Point

A clear sequence to understand the entire solution:

1. **notebooks/analysis.ipynb** –  
  This notebook contains the initial exploration of the dataset, inspection of text fields, and analysis of the distribution of titles, descriptions, and extracted text. It provides the necessary context for why a semantic retrieval strategy was chosen.

2. **notebooks/rec-sys.ipynb** –  
  This notebook shows the end-to-end process used to prepare embeddings, build the FAISS vector database, compute evaluation metrics, and validate the recommender-based approach. It bridges the analysis phase with the final model implementation.

3. **predict_template.py** –  
  This file contains all the final production-ready code used for inference, including the `TopicPredictor` class, embedding generation, FAISS lookup logic, and the request schema. The predictor encapsulates the entire retrieval system and serves as the core entry point for loading the model and making predictions.


# Testable Model
A test script (test_predictor.py) was included to validate that the final implementation meets the Testable Model requirement. The script initializes the TopicPredictor class, loads the FAISS vector index and the topics metadata, and performs predictions using both a valid request (constructed directly from the dataset) and a failure-case request with no textual fields. In both scenarios, the predictor executes without errors and returns a list of topic IDs, demonstrating that the model can be loaded, queried, and executed independently of the notebook. 

# Metrics

- **Definition**

<table border="1" cellpadding="6" cellspacing="0">
    <tr>
        <th>Metric</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><b>rec@k</b></td>
        <td>Recall@k = hits / true_count. Measures how many relevant items were recovered.</td>
    </tr>
    <tr>
        <td><b>prec@k</b></td>
        <td>Precision@k = hits / k. Measures how many retrieved items are correct.</td>
    </tr>
    <tr>
        <td><b>micro_rec@k</b></td>
        <td>Aggregated recall across the dataset: total_hits / total_true.</td>
    </tr>
    <tr>
        <td><b>micro_prec@k</b></td>
        <td>Aggregated precision: total_hits / (N * k), where N is the number of evaluated items.</td>
    </tr>
    <tr>
        <td><b>cosine similarity</b></td>
        <td>Measures how similar two embeddings are based on their direction; with L2-normalized vectors, it is equivalent to the inner product used by FAISS.</td>
    </tr>
</table>


- **Metrics (Dataset Size: 100)**

| k      | micro_rec@k | micro_prec@k |
| :------ | :-----------: | ------------: |
| **3**  | 0.1709      | 0.1133       |
| **5**  | 0.2111      | 0.0840       |
| **10** | 0.2764      | 0.0550       |

- **Metrics (Dataset Size: 10,000)**

| k      | micro_rec@k | micro_prec@k |
| ------ | ----------- | ------------ |
| **3**  | 0.1816      | 0.1104       |
| **5**  | 0.2258      | 0.0824       |
| **10** | 0.2876      | 0.0524       |


- **Cosine Similarity**
The cosine similarity score is not included in the metrics table because it varies for every individual prediction. Unlike aggregate metrics such as micro-precision@k and micro-recall@k, which summarize performance across the entire dataset, cosine similarity reflects the specific semantic distance between a single content item and each retrieved topic. Since its value depends on the particular input and the set of candidates returned in each search, it cannot be represented as a single global metric.

## Discussion 
The metrics for both the 100-sample and 10,000-sample evaluations show a very consistent pattern: micro-recall@k and micro-precision@k remain stable as the dataset size scales up. This stability indicates that the recommender-based approach generalizes well and does not rely on particular samples or overfit to small subsets of the data. Another positive aspect is the behavior of the metrics: recall@k increases as k grows, showing that the system retrieves more true topics when given more space, while precision@k decreases as expected, reflecting the broader set of predictions. Together, these results confirm that the embedding + FAISS similarity strategy behaves reliably, retrieves meaningful matches, and maintains performance consistency across different evaluation sizes.

Although the absolute values of recall and precision are relatively low, this behavior is expected for a zero-shot, embedding-based retrieval approach applied to this particular dataset. The topics often contain very limited textual information, many with no description at all, and the content texts vary greatly in length and richness. These factors make exact matching inherently difficult, especially without supervised fine-tuning. Given these constraints, the obtained metrics are reasonable and consistent with what is typically observed in semantic-retrieval systems operating purely on text similarity.

# Next Steps

Looking ahead, several enhancements could significantly improve the system’s accuracy and robustness. A natural next step would be to fine-tune a bi-encoder model directly on the content–topic correlations, allowing the embeddings to better capture the specific semantic relationships of this dataset. Additionally, incorporating a reranker on the top-k retrieved topics could refine the final ranking and boost precision. Other improvements include enriching topic representations using the full hierarchy (e.g., ancestors, siblings).
