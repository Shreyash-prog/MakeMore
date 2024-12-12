# MakeMore: A Character-Level Language Model

## Overview
**MakeMore** is a character-level language model designed to generate text that mimics the stylistic, contextual, and grammatical features of the data it is trained on. Inspired by Andrej Karpathy's work on character-level recurrent neural networks and generative models, this project builds upon foundational ideas to experiment with modern architectures for text generation. The dataset utilized for training and evaluation contains a rich corpus of Shakespeare's works.

This project was developed as part of the course **Statistical Software 16:954:577:01** at Rutgers University.

## Aim
To build a robust character-level language model capable of producing coherent, stylistic, and contextually rich text inspired by Shakespeare's works.

## Dataset
The project used a comprehensive dataset inspired by Shakespeare's works. Key statistics include:

- **Total Lines**: 214,939
- **Total Tokens**: 1,135,671
- **Unique Characters**: 83
- **Total Words**: 1,135,671
- **Vocabulary Size**: 72,444

## Architectures Explored
The following non-transformer and transformer architectures were implemented and compared:

### Non-Transformer Models
1. **Bag of Words (BoW)**:
   - Focus: Word frequency distributions
   - Result: Lacked sequential order, leading to incoherent text generation
   - Validation Loss: 2.45

2. **Multilayer Perceptron (MLP)**:
   - Focus: Dense feedforward processing
   - Result: Ignored temporal structures, causing incoherence
   - Validation Loss: 1.85

3. **Bigram Model**:
   - Focus: Short-range dependencies between character pairs
   - Result: Limited contextual understanding
   - Validation Loss: 2.95

4. **Recurrent Neural Network (RNN)**:
   - Focus: Sequential dependencies
   - Result: Struggled with long-range dependencies due to compute restrictions
   - Validation Loss: 1.80

### Transformer Model
- **With GeLU Activation and Multi-Head Attention**:
   - Significant improvements in contextual understanding and gradient flow
   - Validation Loss: Improved from earlier attempts
   - Training Time: 5 hours on NVIDIA A100 GPU
   - Additional Features: Larger vocabulary, better tokenization

## Results
The Transformer-based model achieved notable results:
- Successfully mimicked Shakespearean syntax and vocabulary.
- Generated dramatic and stylistically consistent phrases.
- Struggled with maintaining narrative coherence and contextual depth in long sequences.

## Features of Generated Text
- **Shakespearean Style**: Use of Elizabethan English terms like "thou" and "methinks."
- **Character Naming**: Creation of names and dialogues resembling Shakespeare's works.
- **Syntax and Grammar**: Inverted sentence structures, though with occasional grammatical inaccuracies.
- **Vocabulary**: Archaic words and dramatic tone, with some nonsensical combinations.

## Inspiration
This project was inspired by the ideas presented in Andrej Karpathy's [blog post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) on recurrent neural networks and text generation, as well as his work on character-level language models like **NanoGPT** and **MakeMore**. His approach to accessible explanations of deep learning concepts served as a guiding principle for this work.

## Future Improvements
1. Addressing overfitting through advanced regularization techniques.
2. Enhancing dataset diversity for better generalization.
3. Experimenting with deeper architectures for improved coherence and narrative flow.
