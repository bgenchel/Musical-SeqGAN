# Curious-Musical-SeqGAN
Adapt and evaluate SeqGAN (https://arxiv.org/abs/1609.05473) for music generation. 

The original paper briefly mentions that the model was trained for music generation on the Nottingham folk music dataset, and evaluated using BLEU score. Unfortunately, there was not much information given on how the model was adapted for this use case, and how they acheived the scores they reported. Additionally, code from the authors and others are implemented exclusively towards the central task used in the paper, learning to model a randomly initialized LSTM.

Here, we attempt to adapt the model specifically for this purpose and give a clearer and more detailed evaluation on the task of music generation.

To run the model on the Nottingham dataset, navigate to src/models/nottingham and run
`python main.py`t
## A 
