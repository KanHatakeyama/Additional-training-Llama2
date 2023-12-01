# Additional training of Llama 2 with scientific articles

## About
- Source codes for [preprint]().

## Setup
- Clone this repo.
    ~~~
    git clone https://github.com/KanHatakeyama/Additional-training-Llama2.git
    ~~~
- Download dataset from HuggingFace
    ~~~
    git lfs clone https://huggingface.co/datasets/kanhatakeyama/nature-family-CC-papers
    mv nature-family-CC-papers/database/ .
    mv nature-family-CC-papers/smallDB/ .
    ~~~
- Create env (conda)
    ~~~
    conda env create -f environment.yml
    ~~~

## Codes
- Fictional datasets
    - Result analysis
        - [Full-parameter](bayes/1026anal_zero.ipynb)
        - [LoRA](bayes/1023anal.ipynb)
    - Training
        - Full-parameter
            - [Change number of target texts](bayes/1027ds_change_n_lit.py)
            - [Change number of irrelevant texts](bayes/1026ds_change_SN.py)
        - LoRA
            - 7b model
                - [Random search](bayes/1025random16.py)
                - [Optimization](bayes/1025optuna16.py)
            - 7,13,70b models
                - Selected adapter layers
                    - [Change number of target texts](bayes/1027_2comp_models_n_lit.py)
                    - [Change number of irrelevant texts](bayes/1027_3comp_models_n_irr.py)
                - Full adapter layers
                    - [Change number of target texts](bayes/1027_2comp_models_n_lit_full.py)
                    - [Change number of irrelevant texts](bayes/1027_3comp_models_n_irr_full.py)
- Scientific papers
    - [Result analysis](bayes/1111anal_optuna.ipynb)
    - [Training](bayes/1113optuna.py)
## Author
- Kan Hatakeyama
- Tokyo Tech., Japan