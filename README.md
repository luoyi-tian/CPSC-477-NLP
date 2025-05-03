# CPSC-477-NLP Semantic Understanding–Based Recommendation System for Biomedical Literature

This repository contains the code and data for:

> **“Semantic Understanding–Based Recommendation System for Biomedical Literature”**  
> Xin Lan, Luoyi Tian, Jialiao Wang  
> School of Public Health & Graduate School of Arts and Sciences, Yale University

We explore pre-trained Sentence-Transformer models (SBERT) for semantic retrieval on the RELISH dataset, comparing out-of-the-box embeddings, full fine-tuning, and LoRA adapter tuning. Our implementation demonstrates that LoRA achieves strong gains with minimal trainable parameters.

## Instructions for setting up

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
