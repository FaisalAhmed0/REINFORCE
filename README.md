# REINFORCE
This repo. contains a reimplementation of the REINFORCE algorithm (vanilla policy gradient)<br/>


## Clone the repository

```bash
git clone https://github.com/FaisalAhmed0/REINFORCE
```

## You can setup a new environment and install requirements.txt

```bash
conda create -n vpg_env 
pip3 install -r requirements.txt 
```

## You can set up the configuration of the experiment in `congif.py`
</br>

## Activate the new environment and run `train.py` and pass the environment name as a cmd argument

```bash
conda activate vpg_env
python train.py --env "CartPole-v0"
```

### to track the training dynamics run tensorbaord on the same directory
```bash
tensorboard --logir ./runs
```
### after the training finshs you can run a video on the videos folder

