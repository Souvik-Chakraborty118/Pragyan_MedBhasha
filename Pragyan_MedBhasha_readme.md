# MedBhasha: Bengali Clinical NLP Pipeline

**Team Pragyan** | Comsys Hackathon 8: MedBhasha Bengali Clinical NLP (Round 2) - Jadavpur University

!\[Python](https://img.shields.io/badge/Python-3.10%2B-blue)
!\[PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
!\[HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F7931E)
!\[LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
!\[Status](https://img.shields.io/badge/Status-Hackathon\_Ready-success)

## Overview

This repository contains Team Pragyan's complete, end-to-end NLP architecture for the MedBhasha clinical challenge. The system processes raw Bengali clinical notes to extract critical medical entities, groups patients by phenotype using high-dimensional clustering and provides a Retrieval-Augmented Generation (RAG) interface for grounded and medical query answering.

**Key Constraints Conquered:**

1. **Pure Bengali Output:** Strict Llama-3 prompt engineering ensures zero English hallucination and strict adherence to context.
2. **Hardware Limits:** Designed entirely within the 16GB VRAM constraint of a Google Colab T4 GPU via aggressive garbage collection, CPU-bound embedding generation, and 4-bit LLM quantization.



## Architecture 

Our pipeline is divided into three distinct phases to maximize both clinical accuracy and computational efficiency.

### Phase 1: Named Entity Recognition (NER)

To accurately extract medical entities from unstructured Bengali text, so we fine-tuned a base transformer model using a custom weighted loss function to handle class imbalances.

* **Base Model:** `sagorsarker/bangla-bert-base`
* **Entity Schema:** 10 official BIO categories (`SYMPTOM`, `BODY\_SITE`, `TEMPORAL`, `NEGATION\_UNCERTAINTY`, `DIAGNOSIS`, `TEST`, `DOSAGE`, `PROCEDURE`, `MEDICATION`, `TEST\_RESULT`).
* **Training Strategy:** Custom `WeightedNERTrainer` which handle cross-entropy loss weights. Trained over 25 epochs with a learning rate of `3e-5`.
* **Peak Performance (Epoch 13):**

  * **Precision:** 0.835
  * **Recall:** 0.897 *(Extremely high recall ensures minimal clinical miss rate)*
  * **F1-Score:** 0.865

### Phase 2: Phenotype Clustering 

To group clinical notes into distinct patient phenotypes, we utilized a hybrid feature extraction approach. To defeat the **Curse of Dimensionality** when combining dense semantic embeddings that is 768d with sparse entity frequencies that is 10d, we engineered a rigorous PCA-backed compression pipeline.

* **Semantic Embeddings:** `l3cube-pune/bengali-sentence-bert-nli` which is executed on CPU to preserve GPU VRAM.
* **Feature Fusion:** Entity co-occurrence frequencies were extracted and then standardized using `StandardScaler` and concatenated with the semantic embeddings.
* **Dimensionality Reduction:** The 778-dimensional hybrid feature matrix was compressed down to **5 Principal Components** via PCA prior to clustering.
* **Algorithm:** K-Means.
* **Evaluation Metrics:**

  * **Silhouette Score:** 0.2225 
  * **Davies-Bouldin Index:** 1.3319

PCA Projection of Phenotypes diagram is phenotype\_clusters\_final.png in repository
**Phase 3: RAG Medical Assistant** 
---

We implemented a localized RAG system to answer clinical queries directly from the clustered notes without connecting to external APIs.

* **Vector Database:** In-memory LangChain FAISS index using HuggingFace CPU embeddings.
* **LLM:** `unsloth/llama-3-8b-Instruct-bnb-4bit`
* **Quantization:** 4-bit precision loaded via `bitsandbytes` (`bnb\_4bit\_compute\_dtype=torch.float16`).
* **Execution:** Modern LangChain Expression Language (LCEL) chain bypassing deprecated legacy wrappers.
* **Guardrails:** We enforce a strict system prompt using Llama-3 specific `<|start\_header\_id|>` tokens. This forces the model to answer exclusively in Bengali and reject out-of-context queries with *"তথ্যটি নোটে পাওয়া যায়নি"*.



## VRAM Optimization Strategy

To run a NER trainer, dense embedders and a 8-billion parameter LLM on a single T4 GPU leads to highly fragmented memory and Out Of Memory (OOM) crashes. Our pipeline resolves this via:

1. **CPU Offloading:** `SentenceTransformer` and `FAISS` build processes are forced to device="cpu".
2. **Explicit Assassination:** `trainer` and `model` variables from Phase 1 are actively deleted using Python's `del`.
3. **Garbage Collection:** `gc.collect()` and `torch.cuda.empty\_cache()` are triggered explicitly before loading the Llama-3 pipeline to provide 5.7GB of contiguous VRAM.





## Setup \& Execution 

1. **Clone the repository**
2. **Load Environment:** Upload `Pragyan\_MedBasha.ipynb` to Google Colab (T4 GPU Runtime).
3. **Mount Model Weights:** Mount your Google Drive to load the pre-trained 25-epoch BanglaBERT NER model from the `/Pragyan\_NER\_Final` folder (Do not retrain the model on the test set)*.*
4. **Data Swap:** Replace `annoted\_corpus.csv` with the newly released hidden test set provided by the organizers.
5. **Run Pipeline:**

   * Execute the NER inference cell.
   * Execute Phase 2 CPU Clustering.
   * Execute Phase 3 FAISS Build \& LCEL Llama-3 RAG.
6. **Query Evaluation:** Pass the organizers' clinical questions into the RAG query block to generate pure Bengali grounded answers.

