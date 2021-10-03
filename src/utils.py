import gym
from gym.wrappers import Monitor

import torch
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter

from src.model import MLP
from src.config import args


from tqdm import tqdm


def reward_to_go(rewards, gamma=0.99):
  '''
  This function calculate the discounted cumultive reward a.k.a the reward to go
  rewards: list of reward at each time step from t=0 to t=T
  gamma: discount factor
  '''
  cum_rewards = []
  sum_rewards = 0
  for r in reversed(rewards):
      sum_rewards = gamma * sum_rewards + r
      cum_rewards.append( sum_rewards )
  cum_rewards.reverse()
  return torch.tensor(cum_rewards)


def train(env_name, action_type="discrete"):
    # create the environment
    env = gym.make(env_name)

    # create the policy model
    if action_type == "discrete":
        model = MLP(env.observation_space.shape[0], args.n_hiddens, env.action_space.n, "discrete")
    else:
        model = MLP(env.observation_space.shape[0], args.n_hiddens, env.action_space.shape[0], "continuous")

    # create the optimizer
    optimizer = opt.Adam(model.parameters(), lr=args.lr)

    # create a summaryWriter
    sw = SummaryWriter(comment=f"env_name:{env_name}, lr={args.lr}, batch_size={args.batch_size}, baseline: {args.baseline}, entropy: {args.entropy}")
    temp = args.temp

    avg_reward_baseline = None

    for iter in range(args.iterations):
        rewards_avg = 0
        batch_loss = torch.zeros(1)
        for traj in tqdm(range(args.batch_size)):
            rewards = []
            log_probs = []
            state = env.reset()
            done = False
            while not done:
                state_t = torch.FloatTensor(state)
                action, log_prob = model.sample_action(state_t)
                if action.dim() >= 1 and len(action) >=2 :
                    next_state, reward, done, info = env.step(action.cpu().numpy())
                else:
                    next_state, reward, done, info = env.step(action.cpu().item())
                rewards.append(reward)
                log_probs.append(log_prob)
                state = next_state
            cum_reward = reward_to_go(rewards, gamma=args.gamma)
            loss = torch.zeros(1)
            if args.baseline:
                avg_reward_baseline = sum(cum_reward) / len(cum_reward)
            # print(rewards)
            # print(avg_reward_baseline)
            for log, r in zip(log_probs, cum_reward):
                if avg_reward_baseline and (not args.entropy):
                    loss -= (log * (r - avg_reward_baseline))
                elif avg_reward_baseline and (args.entropy):
                    loss -= (log * (r - avg_reward_baseline + temp * model.entropy()))
                elif args.entropy:
                    # print("here")
                    loss -= (log * (r + temp * model.entropy()))
                else:
                    loss -= (log * r )
            batch_loss += loss
            rewards_avg += sum(rewards)
        batch_loss /= args.batch_size
        rewards_avg /= args.batch_size
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        print(f"Iteration:{iter+1}, reward: {rewards_avg}, batch loss: {batch_loss}")
        sw.add_scalar("reward", rewards_avg, iter)
        sw.add_scalar("loss", batch_loss, iter)
    return model


# record the agent performance environment 
def record_video(env_name, model):
    filename = f"f{env_name}.mp4"
    env = gym.make(env_name)
    env = Monitor(env, f'./videos/{filename}', force=True)
    state = env.reset()
    total_reward = 0
    with torch.no_grad():
        done = False
        while not done:
            action,_ = model.sample_action( torch.FloatTensor(state) )
            if action.dim() >= 1 and len(action) >= 2:
                state, reward, done, info = env.step(action.cpu().numpy())
            else:
                state, reward, done, info = env.step(action.cpu().item())
            total_reward += reward
    env.close()
    return reward