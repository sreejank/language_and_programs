## Using natural language and program abstractions to instill human inductive biases in machines

Data

* data/500_gsp_samples_text_human_encoded.npy: Human language embeddings 

* data/500_gsp_samples_text_synth_encoded.npy: Synthetic language embeddings 

* data/500_gsp_samples_text_synth.npy: Synthetic language descriptions 

* data/500_gsp_samples_text_human.npy: Human language descriptions 

* data/gsp_samples-recognition_activations-structurePenalty2.npz: Program embeddings 

* data/500_gsp_samples.npy: Training boards 

* data/gsp_4x4_full.npy: Boards from the GSP chain 

* data/gsp_4x4_full_probs.npy: Frequencies of each board in the GSP chain. 

* data/gsp_4x4_sample.npy: Test set GSP boards 

* data/gsp_4x4_sample_starts.npy: Start tiles for test set GSP boards 

* data/gsp_4x4_null_sample.npy: Test set control boards 

* data/gsp_4x4_null_sample-starts.npy: Start tiles for test set control boards 

* data/hyperparams_grounding.pkl: Hyperparams for grounding agents.

* data/hyperparams_nogrounding.pkl: Hyperparams for non-grounding agents. 

Meta-RL Agent Code
* auxillary_model.py: Training code for agents w/ auxillary loss.
* auxillary_polcy.py: Setup for agent training code w/ auxillary loss.
* small_env_lang_4x4.py: Modified training enviornment for grounded meta-rl agent for the GSP task distribution. 
* small_env_4x4.py: Training enviornment for baseline meta-rl agent for the GSP task distribution. 
* task_performance_zscore.py: Code to calculate task performance metric of paper. 


Program Induction with DreamCoder 
* ec-master/: Folder with program induction code. This is a fork from: https://github.com/ellisk42/ec The vast majority of the code here is from the public repository of DreamCoder (Ellis et al. 2021) 
here: https://github.com/ellisk42/ec. Our additions are the implementation of DreamCoder in our enviornment (mostly in: dreamcoder/domains/grid, which has its own readme). 

