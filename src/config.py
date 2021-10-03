from argparse import Namespace


args = Namespace(
# policy learning rate
lr = 1e-3,

# environment name
env_name = "CartPole-v0",

# batch size
batch_size = 32,

# iterations
iterations = 100,

# gamma
gamma = 0.99,

# size of the mlp hidden layers
n_hiddens = [64],

# temperature of the entropy term
temp = 0.1,

# use a baseline 
baseline = False,

# use entropy regulrization
entropy = True, 

# radnom seed
seed = 1234)