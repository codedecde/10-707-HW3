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
* Generates 
    - models_<Activation>/model_embed_EMBED_hidden_HIDDEN_epoch_EPOCH_ppx_PPX.model
    - Summary_NGRAM/summary_activation_ACTIVATION_Hidden_HIDDEN_valppx_VALPPX.pkl




