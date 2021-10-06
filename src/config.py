from argparse import Namespace


args = Namespace(
# policy learning rate
lr = 1e-3,

# environment name
env_name = "CartPole-v0",

# batch size
batch_size = 32,

# iterations
iterations = 300,

# gamma
gamma = 0.99,

# size of the mlp hidden layers
n_hiddens = [64, 32],

# temperature of the entropy term
temp = 1,

# baseline type
# baseline type 1: avergae reward, 2: value function
baseline = 0,

# use entropy regulrization
entropy = True, 

# radnom seed
seed = 42)