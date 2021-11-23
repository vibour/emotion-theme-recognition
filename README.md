Emotion and Theme Recognition in Music
==============================

The repository contains code for the submission of the lileonardo team to the [2021 Emotion and Theme Recognition in Music](https://multimediaeval.github.io/2021-Emotion-and-Theme-Recognition-in-Music-Task/) task of MediaEval 2021 ([results](https://multimediaeval.github.io/2021-Emotion-and-Theme-Recognition-in-Music-Task/results)).

Requirements
------------

*   `python >= 3.7`
*   `pip install -r requirements.txt` in a [virtual environment](https://docs.python.org/3/tutorial/venv.html)
*   Download data from the [MTG-Jamendo Dataset](https://github.com/MTG/mtg-jamendo-dataset) in `data/jamendo`.
    Audio files go to `data/jamendo/mp3` and melspecs to `data/jamendo/melspecs`.
*   Process 128 bands mel spectrograms and store them in `data/jamendo/melspecs2` by running:
    ```bash
    python preprocess.py experiments/preprocessing/melspecs2.json
    ```

Usage
-----

Run `python main.py experiments/DIR` where `DIR` contains the parameters.

Parameters are overridable by command line arguments:
```bash
python main.py --help
```
``` 
usage: main.py [-h] [--data_dir DATA] [--num_workers NUM] [--restart_training] [--restore_name NAME]
               [--num_epochs EPOCHS] [--learning_rate LR] [--weight_decay WD] [--dropout DROPOUT]
               [--batch_size BS] [--manual_seed SEED] [--model MODEL] [--loss LOSS]
               [--calculate_stats]
               DIRECTORY

Train according to parameters in DIRECTORY

positional arguments:
  DIRECTORY            path of the directory containing parameters

optional arguments:
  -h, --help           show this help message and exit
  --data_dir DATA      path of the directory containing data (default: data)
  --num_workers NUM    number of workers for dataloader (default: 4)
  --restart_training   overwrite previous training (default is to resume previous training)
  --restore_name NAME  name of checkpoint to restore (default: last)
  --num_epochs EPOCHS  override number of epochs in parameters
  --learning_rate LR   override learning rate
  --weight_decay WD    override weight decay
  --dropout DROPOUT    override dropout
  --batch_size BS      override batch size
  --manual_seed SEED   override manual seed
  --model MODEL        override model
  --loss LOSS          override loss
  --calculate_stats    recalculate mean and std of data (default is to calculate only when they
                       don't exist in parameters)
```

Ensemble predictions
--------------------

The predictions are averaged by running:

```bash
python average.py --outputs experiments/convs-m96*/predictions/test-last-swa-outputs.npy --targets experiments/convs-m96*/predictions/test-last-swa-targets.npy --preds_path predictions/convs.npy
```

```bash
python average.py --outputs experiments/filters-m128*/predictions/test-last-swa-outputs.npy --targets experiments/filters-m128*/predictions/test-last-swa-targets.npy --preds_path predictions/filters.npy
```

```bash
python average.py --outputs predictions/convs.npy predictions/filters.npy --targets predictions/targets.npy
```
