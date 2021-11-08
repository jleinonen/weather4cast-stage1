This code was used for the entry by the team "antfugue" for the Weather4cast 2021 Challenge. Below, you can find the instructions for generating predictions, evaluating pre-trained models and training new models.

## Installation

To use the code, you need to:
1. Clone the repository.
1. Setup a conda environment. You can find an environment verified to work in the `environment.yml` file. However, you might have to adapt it to your own CUDA installation.
1. Fetch the data you want from the competition website. Follow the instructions [here](https://github.com/iarai/weather4cast#get-the-data). The data should should be in the `data` directory following the structure specified [here](https://github.com/iarai/weather4cast#start-here).
1. (Optional) If you want to use the pre-trained models, load them from https://doi.org/10.5281/zenodo.5101213. Place the `.h5` files in the `models/best` directory.


## Running the code

Go to the `weather4cast` directory. There you can either launch the `main.py` script with instructions provided below, or launch an interactive prompt (e.g. `ipython`) and then import modules and call functions from them.

### Reproducing predictions
Run:
```bash
python main.py submit --comp_dir=w4c-core-stage-1 --submission_dir="../submissions/test"
```
where you can change `--comp_dir` to indicate which competition you want to create predictions for (these correspond to the directory names in the `data` directory) and `--submission_dir` to indicate where you want to save the predictions.

This script automatically loads the best model weights corresponding to the "V4pc" submission that produced the best scores on the leaderboards. To experiment with other weights, see the function `combined_model_with_weights` in `models.py` and the call to that in `main.py`. You can change the combination of models and weights with the argument `var_weights` in `combined_model_with_weights`.

Generating the predictions should be possible in a reasonable time also on a CPU.

### Evaluate pre-trained model
```bash
python main.py evaluate --comp_dir=w4c-core-stage-1 --model=resgru --weights="../models/best/resrnn-temperature.h5" --dataset=CTTH --variable=temperature
```
This example trains the ResGRU model for the _temperature_ variable, loading the pre-trained weights from the `--weights` file. You can change the model and the variable using the `--model`, `--weights`, `--dataset` and `--variable` arguments.

A GPU is recommended for this although in principle it can be done on a CPU.

### Train a model
```bash
python main.py train --comp_dir="w4c-core-stage-1" --model="resgru" --weights=model.h5 --dataset=CTTH --variable=temperature
```
The arguments are the same as for `evaluate` except the `--weights` parameter indicates instead the weights file that the training process keeps saving in the `models` directory.

A GPU is basically mandatory. The default batch size is set to 32 used in the study but you may have to reduce it if you don't have a lot of GPU memory.

**Hint**: It is not recommended to train like this except for demonstration purposes. Instead I recommend you look at how the `train` function in `main.py` works and follow that in an interactive prompt. The batch generators `batch_gen_train` and `batch_gen_valid` are very slow at first but get faster as they cache data. Once the cache is fully populated they will be much faster. You can avoid this overhead by pickling a fully loaded generator. For example:
```python
import pickle

for i in range(len(batch_gen_train)):
    batch_gen_train[i] # fetch all batches

with open("batch_gen_train.pkl", 'wb') as f:
    pickle.dump(batch_gen_train, f)
```
