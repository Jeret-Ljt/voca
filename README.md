# Sync-Lip model: lip generation from speech

lip-sync model is a machine learning model which take people audio as input and output 52 blendshapes as the facial expression prediction

## Set-up

The code uses Python 3.6.8. Please make sure your Python version is correct. This code can be ran on Windows, Mac or Linux OS

Clone the git project:
```
git clone git@github.com:Jeret-Ljt/voca.git
```

Set up virtual environment:
```
mkdir <your_home_dir>/.virtualenvs
python3 -m venv <your_home_dir>/.virtualenvs/voca
```

Activate virtual environment:
```
cd voca
source <your_home_dir>/.virtualenvs/voca/bin/activate
```

Make sure your pip version is up-to-date:
```
pip install -U pip
```

The requirements (including tensorflow) can be installed using:
```
pip install -r requirements.txt
```
## Data

#### the pretrained model to run the demo 

Download the pretrained DeepSpeech model (v0.5.0) from [Mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech/releases/download/v0.5.0/deepspeech-0.5.0-models.tar.gz) (i.e. deepspeech-0.5.0-models.tar.gz).


Download the pretrained model checkpoints from [pretrained model checkpoints oneDrive sharing link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/ljt2021_connect_hku_hk/Eci3joLLQjhHghxLkG45lXQBVphilHSH9_0v9wJZNzBjXg?e=RKa4ip) (i.e. checkpoints.zip)

#### the Data used to train the model

Download the training data from [training data oneDrive sharing link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/ljt2021_connect_hku_hk/Ecxv2d31RiNNjbKLX6XUwBoBicmQlWoozky_UspqLiCXsg?e=dXH1LG)

Training subjects:
```
1.mp4 2.mp4 3.mp4 4.mp4 5.mp4 6.mp4
```

Validation subjects:
```
7.mp4
```


#### demo

This demo runs sync-lip modle , which outputs the 52 blendshapes sequence for the given audio sequences
```
python run_voca.py --tf_model_fname './model/gstep_134310.model' --ds_fname '/ds_graph/deepspeech-0.5.0-models/output_graph.tflite' --audio_fname './audio/test_sentence.wav'
```
## Training

We provide code to train a sync-lip model. Prior to training, run the above demo, as the training shares the requirements.
Additionally, download the sync-lip training data from [training data onedrive sharing link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/ljt2021_connect_hku_hk/Ecxv2d31RiNNjbKLX6XUwBoBicmQlWoozky_UspqLiCXsg?e=dXH1LG)

The training code requires a config file containing all model training parameters. To create a config file, run
```
python config_parser.py
```

To start training, run
```
python run_training.py
```

To visualize the training progress, run
```
tensorboard --logdir='./training/summaries/' --port 6006
```
This generates a [link](http://localhost:6006/) on the command line.  Open the link with a web browser to show the visualization.





