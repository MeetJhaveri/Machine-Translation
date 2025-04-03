# Machine Translation

This repository contains an implementation of a Neural Machine Translation (NMT) system using sequence-to-sequence architecture with attention mechanism.

## Project Overview

This project implements a neural machine translation system that can translate text between different languages, particularly focusing on German to English translation.

### Architecture Visualization

![Neural Machine Translation Architecture](https://github.com/MeetJhaveri/Machine-Translation/blob/master/architecture.png)

*The image shows our neural machine translation model architecture. The left side represents the encoder processing German input ("guten morgen"), while the right side shows the decoder generating English output ("good morning"). The middle component (z) represents the attention mechanism that connects the encoder and decoder, allowing the model to focus on relevant parts of the input when generating each output word.*

## Architecture Details

### Encoder
The encoder processes the input German sentence word by word. Each input token (including special tokens like `<sos>` and `<eos>`) is first converted to an embedding and then processed sequentially through recurrent neural network layers. The encoder produces hidden states (h₁, h₂, h₃, h₄) that capture the contextual representation of each input word.

### Attention Mechanism
The attention mechanism (represented by 'z' in the diagram) is the key innovation of this architecture. It allows the decoder to "focus" on different parts of the source sentence during translation. For each output word, the attention mechanism calculates a weighted sum of all the encoder hidden states, where the weights represent the relevance of each input word to the current output word.

The dotted lines in the diagram show these attention connections, illustrating how each decoder state can access and prioritize relevant parts of the input sequence.

### Decoder
The decoder generates the English translation one word at a time. At each step, it:
1. Takes the previous generated word (or `<sos>` for the first word)
2. Uses the attention mechanism to gather relevant information from the encoder
3. Updates its internal state (s₁, s₂, s₃)
4. Predicts the next word in the target language

This process continues until the decoder generates the `<eos>` token, signaling the end of the translation.


## Translation Example

**Input (German):** 

ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen

**Output (English):**

a boat carrying several men is pulled to shore by a large team of horses.

## Features

- Sequence-to-sequence model with attention mechanism
- German to English translation
- Attention visualization capabilities

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- NLTK (for tokenization and evaluation)
- Matplotlib (for visualizations)

