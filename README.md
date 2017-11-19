# 10-707-HW3
HW3 of the deep learning course: Language Models
### PREREQUISITES
* The code assumes a data/ directory, which contains train.txt and val.txt
* Constants.py defines global constants:
    - VOCAB_SIZE: 8000
    - VOCAB_FILE_NAME: data/vocab.pkl (generated by data.py)
    - START TOKEN : <START>
    - END TOKEN : <END>
    - NGRAM: 3 for this assignment

### DATA PREPROCESSING (data.py)
* Processed data
* Generates:
    -  data/vocab.pkl (the vocab : a dictionary mapping words to indices)
    - data/common_fourgrams.txt : The top 50 fourgrams
    - Images_NGRAM/fourgram_frequency.png: The frequency plot of four grams (Creates Images_NGRAM if not present)

### NGRAM model run (lm.py)
* Runs a NGRAM model
* python lm.py -ne 16 -nd 128 -a linear -epochs 100 -b 512 -lr 0.1 -p 2
    - ne: embedding size (default=16)
    - nd: Hidden layer size (default 128)
    - a: Activation (linear, sigmoid, tanh) (default linear)
    - epochs: Number of epochs (default 100)
    - b: Batch size (default 512)
    - lr: Learning Rate (default 0.1)
    - p: Patience (how many epochs to wait before halving lr if performance doesnt improve) (default 2)
* Runs the Language Model, computing the training loss, val loss, training ppx, and validation ppx
* Generates 
    - models_<Activation>/model_embed_EMBED_hidden_HIDDEN_epoch_EPOCH_ppx_PPX.model
    - Summary_NGRAM/summary_activation_ACTIVATION_Hidden_HIDDEN_valppx_VALPPX.pkl (See auxillary files)

### RNN Model run (rnn_lm.py)
* Runs a RNN model
* python rnn_lm.py -ne 16 -nd 128 -epochs 100 -b 512 -lr 0.1 -p 2 -t 2 -o False
    - ne: embedding size (default=16)
    - nd: Hidden layer size (default 128)
    - epochs: Number of epochs (default 100)
    - b: Batch size (default 512)
    - lr: Learning Rate (default 0.1)
    - p: Patience (how many epochs to wait before halving lr if performance doesnt improve) (default 2)
    - t: Truncate BPTT after (default None)
    - o: Initialize weights of RNN with orthonormal weights, known to generally help (default False)
* Runs the RNN Model, computing the training loss, val loss, training ppx, and validation ppx
* Truncated BPTT happens for 10% cases, when specified
* Generates 
    - models_<Activation>/model_ortho_ORTHO_truncated_TRUNCATED_embed_EMBED_hidden_HIDDEN_epoch_EPOCH_ppx_PPX.model
    - Summary_NGRAM/summary_ortho_ORTHO_truncated_TRUNCATED_Hidden_HIDDEN_Embed_EMBED_valppx_VALPPX.pkl (See auxillary files)

### Auxillary Files
#### LanguageModel.py
The actual language model
#### Layers
The layers used. Defines the following classes
* Variable: A convenient way of packing the data and gradients together
* DenseLayer
* EmbeddingLayer
* BatchNormLayer

#### Activations.py
The available activation functions

#### RNNLanguageModel.py
The RNN Language model. Coded up in pytorch (with cuda support)

#### Optimizer.py
The optimizer for SGD. Currently only contains SGD with momentum and l2

#### Loss.py
The loss functions (MSE, binary_crossentroy and categorical_crossentropy)

#### Module.py
An abstract class, used to keep track of parameters in a model

#### Summary.py
Defines a Summary and History object
* History Keeps track of the following
    - epoch: The epoch which the history corresponds to
    - metrics: The metrics that we want to keep track of
    - weights_file: The model file for a model we want to save
    - params: The actual params which we want to store in weights_file

* A Summary consists of a list of History objects, along with meta data (generally hyper-parameters used in a run)
    - It keeps track of the best parameters, and updates the model file, removing stale model files, and saving the best encountererd so far.

It also defines some auxillary functions
* plot_summaries: to plot metrics of different summaries
* generate_csv: Iterates over multiple summaries, generating a csv of accuracies etc

Finally, there are some other functions for the language model
* generate_language : Generates sentences given a model, a vocab, and a seed
* find_nearest_k: Finds the k nearest neighbors based on euclidien distance
* visualize_embeddings: Scatterplot of the embedding matrix
