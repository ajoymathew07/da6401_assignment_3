# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Wandb Report and Github Link

Wandb report: https://api.wandb.ai/links/cs23b101-indian-institute-of-technology-madras/scxwmz6u
Github link : https://github.com/ajoymathew07/da6401_assignment_3.git
## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
