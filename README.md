# Machine Translation

This repository contains an implementation of a Neural Machine Translation (NMT) system using sequence-to-sequence architecture with attention mechanism.

## Project Overview

This project implements a neural machine translation system that can translate text between different languages, particularly focusing on German to English translation.

### Architecture Visualization

![Neural Machine Translation Architecture](./images/nmt_architecture.png)

*The image shows our neural machine translation model architecture. The left side represents the encoder processing German input ("guten morgen"), while the right side shows the decoder generating English output ("good morning"). The middle component (z) represents the attention mechanism that connects the encoder and decoder, allowing the model to focus on relevant parts of the input when generating each output word.*

## Features

- Sequence-to-sequence model with attention mechanism
- Support for multiple language pairs
- Attention visualization capabilities
- Pre-trained models for German-English translation

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- NLTK (for tokenization and evaluation)
- Matplotlib (for visualizations)

