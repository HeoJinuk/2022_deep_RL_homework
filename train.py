import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from envs import create_atari_env
from model import A3C


def ensure_shared_grads(model, global_network):
    for param, shared_param in zip(model.parameters(), global_network.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, global_network, optimizer=None):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = A3C(env.observation_space.shape[0], env.action_space.n)

    if optimizer is args.optimizer:
        optimizer = optim.Adam(global_network.parameters(), lr=args.lr)
    else:
        optimizer = optim.RMSprop(global_network.parameters(), lr=args.lr)

    model.train()
    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0
    start_time = time.time()
    while True:        
        episode_length += 1
        # Sync with the shared model
        model.load_state_dict(global_network.state_dict())
        if done: hx = torch.zeros(1, 256)
        else: hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        #Take Action
        for step in range(args.num_steps):            
            value, logit, hx = model((state.unsqueeze(0), hx))
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            m = model.distribution(log_prob.data)
            action = m.sample().numpy()[0]

            #Compute States and Rewards
            state, reward, done, _ = env.step(action)
            done = done or episode_length >= args.max_episode_length

            if done:
                reward = -1
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            env.render(mode='rgb_array')

        if episode_length >= args.max_episode_length:
            break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), hx))
            R = value.detach()

        #Calculating Gradients
        values.append(R)
        policy_loss = 0
        value_loss = 0
  
        GAE = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            #Discounted Sum of Future Rewards + reward for the given state
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]

            value_loss = value_loss + advantage.pow(2)

            # Generalized Advantage Estimataion(GAE)
            delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
            GAE = GAE * args.gamma * args.tau + delta_t
            policy_loss = policy_loss - log_probs[i] * GAE - 0.01 * entropies[i]
       
        optimizer.zero_grad()

        total_loss = (policy_loss + value_loss).mean()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
        ensure_shared_grads(model, global_network)
        optimizer.step()
    env.close()