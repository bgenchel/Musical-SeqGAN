# Curious-Musical-SeqGAN
Adapt and evaluate SeqGAN (https://arxiv.org/abs/1609.05473) for music generation. 

The original paper briefly mentions that the model was trained for music generation on the Nottingham folk music dataset, and evaluated using BLEU score. Unfortunately, there was not much information given on how the model was adapted for this use case, and how they acheived the scores they reported. Additionally, code from the authors and others are implemented exclusively towards the central task used in the paper, learning to model a randomly initialized LSTM.

Here, we attempt to adapt the model specifically for this purpose and give a clearer and more detailed evaluation on the task of music generation.

To run the model on the Nottingham dataset, navigate to src/models/nottingham and run
`python main.py`t
## A 

## Running the MGEval Toolkit
The MGEval toolkit requires the use of python 2, which is not the version of python needed to run SeqGAN.

The following steps outline how to create an environment suitable to run the toolkit.

**N.B. This environment does not work on Mac OSX due to matplotlib needing python as a framework.**

**Please do this on a Linux machine if possible.**

Replace ENV_NAME with a name you see fit.

- `conda create --name ENV_NAME python=2`
- `conda install scikit-learn`
- `pip install matplotlib`
- `pip install seaborn`
- `pip install pretty_midi`
- `pip install python-midi`
- `pip install metrics`

To generate MIDI files first run the following **using the python 3 rl4gm env**:

- From project root, `cd src/models/nottingham`
- `(rl4gm) python eval.py -r`
    - This defaults to generating 1000 sequences each for targets, pretrained generator, and adversarially trained generator.
    - Use `--num_midi_samples NUM` to change the number of MIDI files generated.
    
Then, **using the python 2 env**, run the following:

- From project root, `cd src/evaluation`
- `(python2) python toolkit.py`
    - The MGEval toolkit is not consistent on what it requires to generate/display each metric.
    - It is currently set up to show the `note_length_transition_matrix` for the targets and adversarially trained generator samples.
    - To explore further, change the metric name, and update the expected shape and args/kwargs as needed, per the source 
    [here](https://github.com/RichardYang40148/mgeval/blob/master/mgeval/core.py).