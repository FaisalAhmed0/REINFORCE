from argparse import Namespace


args = Namespace(
# policy learning rate
lr = 1e-3,

# environment name
env_name = "CartPole-v0",

# batch size
batch_size = 32,

# iterations
iterations = 205,

# gamma
gamma = 0.99,

# size of the mlp hidden layers
n_hiddens = [64],

# temperature of the entropy term
temp = 0.1,

# baseline type
# baseline type 1: avergae reward, 2: value function
baseline = 2,

# use entropy regulrization
entropy = False, 

# radnom seed
seed = 42)