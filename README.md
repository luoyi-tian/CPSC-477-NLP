# CPSC-477-NLP Semantic Understanding–Based Recommendation System for Biomedical Literature

This repository contains the code and data for:

> **“Semantic Understanding–Based Recommendation System for Biomedical Literature”**  
> Xin Lan, Luoyi Tian, Jialiao Wang  
> School of Public Health & Graduate School of Arts and Sciences, Yale University

We explore pre-trained Sentence-Transformer models (SBERT) for semantic retrieval on the RELISH dataset, comparing out-of-the-box embeddings, full fine-tuning, and LoRA adapter tuning. Our implementation demonstrates that LoRA achieves strong gains with minimal trainable parameters.

## Instructions for running

No complex setup is required—everything can be run directly in Google Colab. Simply open the notebook you want and execute all cells. Make sure the `RELISH_v1.json` file is in the same folder as the notebook.

Here’s what’s in this repo:

- **RELISH_v1.json**  
  Raw RELISH v1 dataset (≈180 k articles).  

- **RELISH.ipynb**  
  Data preprocessing: parses `RELISH_v1.json`, fetches metadata via NCBI E-Utilities, cleans text, and splits into train/val/test.  

- **SBERT+FAISS_baseline_final code.ipynb**  
  Baseline retrieval pipeline: generates SBERT embeddings, builds a FAISS `IndexFlatIP` index over ℓ₂-normalized vectors, and evaluates MAP/MRR/NDCG.  

- **SBERT Finetune.ipynb**  
  Standard full fine-tuning of SBERT (`all-mpnet-base-v2`) using triplet loss, with batch size 16, LR=1e-5, 3 epochs, and 10 % warmup.  

- **LoRA_Finetuning.ipynb**  
  LoRA adapter tuning on SBERT: injects rank-8 adapters in each layer’s query/value, trains only ~0.5 M parameters (α=32, dropout=0.1) under the same hyperparams.

### Running in Colab

No local setup is required—everything runs in Google Colab. The one extra step is to mount your Google Drive so that the `RELISH_v1.json` file can be accessed.

1. **Save `RELISH_v1.json` into your Google Drive.**  
   For example, put it in `MyDrive/CPSC-477-NLP/RELISH_v1.json`.

2. **Open the notebook from GitHub** (replace `RELISH.ipynb` with the one you want)

3. **Mount Google Drive** run the code at the top of the Colab notebook:
```python
from google.colab import drive
drive.mount('/content/drive')
```
4. **Copy the JSON file path into the Colab session**

That’s it—no further installation or setup needed!  


## All dependencies and external libraries used

- Python 3.8  
- torch==1.11.0  
- transformers==4.21.0  
- sentence-transformers==2.2.0  
- faiss-cpu==1.7.1  
- numpy>=1.21.0  
- pandas>=1.3.0  
- scikit-learn>=0.24.0  
- tqdm>=4.64.0  
- requests>=2.26.0  
- biopython>=1.79

## The code can also be found through these link:
SBERT+FAISS_baseline_final code: https://colab.research.google.com/drive/1Brtigj3f3abG8cPh3pMNe9IEdsM17ouq
