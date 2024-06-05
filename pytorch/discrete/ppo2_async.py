import gym
from gym.envs.registration import register
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy
import time
import datetime

import ray
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class ActorCritic(nn.Module):
    def __init__(self, state_dim,action_dim):
        super().__init__()
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
              )
        self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),

              )

    def forward(self,x):
        logits=self.actor(x)
        pi=F.softmax(logits,dim=1)
        v=self.critic(x)
        return pi,v




class Memory(Dataset):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.probs=[]
        self.values=[]

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype=np.float32), \
            np.array(self.actions[idx], dtype=np.float32),\
            np.array([self.rewards[idx]], dtype=np.float32), \
            np.array([self.dones[idx]], dtype=np.float32), \
            np.array(self.next_states[idx], dtype=np.float32),\
            np.array(self.probs[idx],dtype=np.float32),\
            np.array(self.values[idx],dtype=np.float32)

    def get_all(self):
        return self.states, self.actions, self.rewards, self.dones, self.next_states,self.probs,self.values

    def save_all(self, states, actions, rewards, dones, next_states,probs,values):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.next_states = next_states
        self.probs=probs
        self.values=values

    def save_eps(self, state, action, reward, done, next_state,prob,value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.probs.append(prob)
        self.values.append(value)

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]
        del self.probs[:]
        del self.values[:]


class Distributions():
    def __init__(self, myDevice=None):
        self.device = myDevice if myDevice != None else device

    def sample(self, datas):
        distribution = Categorical(datas)
        return distribution.sample().float().to(self.device)

    def entropy(self, datas):
        distribution = Categorical(datas)
        return distribution.entropy().float().to(self.device)

    def logprob(self, datas, value_data):
        distribution = Categorical(datas)
        return distribution.log_prob(value_data).unsqueeze(1).float().to(self.device)

    def kl_divergence(self, datas1, datas2):
        distribution1 = Categorical(datas1)
        distribution2 = Categorical(datas2)

        return kl_divergence(distribution1, distribution2).unsqueeze(1).float().to(self.device)


class PolicyFunction():
    def __init__(self, gamma=0.99, lam=0.95, policy_kl_range=0.03, policy_params=2):
        self.gamma = gamma
        self.lam = lam
        self.policy_kl_range = policy_kl_range
        self.policy_params = policy_params

    def monte_carlo_discounted(self, rewards, dones):
        running_add = 0
        returns = []

        for step in reversed(range(len(rewards))):
            running_add = rewards[step] + (1.0 - dones[step]) * self.gamma * running_add
            returns.insert(0, running_add)

        return torch.stack(returns)

    def temporal_difference(self, reward, next_value, done):
        q_values = reward + (1 - done) * self.gamma * next_value
        return q_values

    def generalized_advantage_estimation(self, values, rewards, next_values, dones):
        gae = 0
        adv = []

        delta = rewards + (1.0 - dones) * self.gamma * next_values - values
        for step in reversed(range(len(rewards))):
            gae = delta[step] + (1.0 - dones[step]) * self.gamma * self.lam * gae
            adv.insert(0, gae)

        return torch.stack(adv)


class Learner():
    def __init__(self, state_dim, action_dim, is_training_mode, policy_kl_range, policy_params, value_clip,
                 entropy_coef, vf_loss_coef,clip_coef,max_grad_norm,
                 minibatch, PPO_epochs, gamma, lam, learning_rate):
        self.clip_coef=clip_coef
        self.max_grad_norm=max_grad_norm
        self.policy_kl_range = policy_kl_range
        self.policy_params = policy_params
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.vf_loss_coef = vf_loss_coef
        self.minibatch = minibatch
        self.PPO_epochs = PPO_epochs
        self.is_training_mode = is_training_mode
        self.action_dim = action_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.learner_model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = Adam(self.learner_model.parameters(), lr=learning_rate)

        self.memory = Memory()
        self.policy_function = PolicyFunction(gamma, lam)
        self.distributions = Distributions()

        if is_training_mode:
            self.learner_model.train()
        else:
            self.learner_model.eval()

    def save_all(self, states, actions, rewards, dones, next_states,probs,values):
        self.memory.save_all(states, actions, rewards, dones, next_states,probs,values)

    # Loss for PPO
    def get_loss(self, probs, values, old_probs, old_values, next_values, actions, rewards, dones):
        # Don't use old value in backpropagation
        Old_values = old_values.detach()

        # Finding the ratio (pi_theta / pi_theta__old):
        logprobs = self.distributions.logprob(probs, actions)
        Old_logprobs = self.distributions.logprob(old_probs, actions).detach()

        # Getting general advantages estimator
        Advantages = self.policy_function.generalized_advantage_estimation(values, rewards, next_values, dones)
        Returns = (Advantages + values).detach()
        Advantages = ((Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)).detach()

        ratios = (logprobs - Old_logprobs).exp()
        Kl = self.distributions.kl_divergence(old_probs, probs)

        # Combining TR-PPO with Rollback (Truly PPO)
        # pg_loss = torch.where(
        #     (Kl >= self.policy_kl_range) & (ratios > 1),
        #     ratios * Advantages - self.policy_params * Kl,
        #     ratios * Advantages
        # )
        # pg_loss = -pg_loss.mean()

        pg_loss1 = -Advantages * ratios
        pg_loss2 = -Advantages * torch.clamp(ratios, 1-self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Getting entropy from the action probability
        dist_entropy = self.distributions.entropy(probs).mean()

        # Getting critic loss by using Clipped critic value
        vpredclipped = Old_values + torch.clamp(values - Old_values, -self.value_clip,
                                                self.value_clip)  # Minimize the difference between old value and new value
        vf_losses1 = (Returns - values).pow(2) * 0.5  # Mean Squared Error
        vf_losses2 = (Returns - vpredclipped).pow(2) * 0.5  # Mean Squared Error
        critic_loss = torch.max(vf_losses1, vf_losses2).mean()

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss
        loss = pg_loss+(critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef)
        return loss

    # Get loss and Do backpropagation
    def training_ppo(self, states, actions, rewards, dones, next_states,old_probs,old_values):
        probs, values = self.learner_model(states)
        # old_probs, old_values = self.actor_old(states), self.critic_old(states)
        _,next_values = self.learner_model(next_states)

        # print(probs.shape,old_probs.shape,next_values.shape,values.shape)

        loss = self.get_loss(probs, values, old_probs, old_values, next_values, actions, rewards, dones)


        # === Do backpropagation ===

        self.optimizer.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm_(self.learner_model.parameters(), max_norm=self.max_grad_norm, norm_type=2)

        self.optimizer.step()

        # === backpropagation has been finished ===

    # Update the model
    def update_ppo(self):
        batch_size = int(len(self.memory) / self.minibatch)
        dataloader = DataLoader(self.memory, batch_size, shuffle=False)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):
            for states, actions, rewards, dones, next_states,probs,values in dataloader:
                self.training_ppo(states.float().to(device), actions.float().to(device), rewards.float().to(device), \
                                  dones.float().to(device), next_states.float().to(device),probs.float().to(device),\
                                  values.float().to(device))

        # Clear the memory
        self.memory.clear_memory()

        # Copy new weights into old policy:
        # self.actor_old.load_state_dict(self.actor.state_dict())
        # self.critic_old.load_state_dict(self.critic.state_dict())

    def get_weights(self):
        return self.learner_model.state_dict()

    def save_weights(self):
        torch.save(self.learner_model.state_dict(), 'agent.pth')


class Agent:
    def __init__(self, state_dim, action_dim, is_training_mode):
        self.is_training_mode = is_training_mode
        self.device = torch.device('cpu')

        self.memory = Memory()
        self.distributions = Distributions(self.device)
        self.actor_model = ActorCritic(state_dim, action_dim).to(self.device)

        if is_training_mode:
            self.actor_model.train()
        else:
            self.actor_model.eval()

    def save_eps(self, state, action, reward, done, next_state,prob,value):
        self.memory.save_eps(state, action, reward, done, next_state,prob,value)

    def get_all(self):
        return self.memory.get_all()

    def clear_all(self):
        self.memory.clear_memory()

    @torch.no_grad
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device).detach()
        prob,value = self.actor_model(state)

        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action = self.distributions.sample(prob)
        else:
            action = torch.argmax(prob, 1)

        # print(prob.shape)

        return int(action.cpu().item()),prob.flatten(),value.flatten()

    def set_weights(self, weights):
        self.actor_model.load_state_dict(weights)

    def load_weights(self):
        self.actor_model.load_state_dict(torch.load('agent.pth', map_location=self.device))


@ray.remote
class Runner():
    def __init__(self, env_name, training_mode, render, n_update, tag):
        self.env = gym.make(env_name)
        self.states = self.env.reset()

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.agent = Agent(self.state_dim, self.action_dim, training_mode)

        self.render = render
        self.tag = tag
        self.training_mode = training_mode
        self.n_update = n_update


        self.last_model_load_time = time.time()

        self.load_model_start_time= self.last_model_load_time

        self.load_model_cnt = 0


    def run_episode(self, i_episode, total_reward, eps_time):
        self.agent.load_weights()
        self.load_model_cnt += 1
        print(
            f"Process {self.tag} loaded new model, load cnt is {self.load_model_cnt},load time is {time.time() - self.load_model_start_time} !")
        self.last_model_load_time = time.time()  # 更新载入时间
        self.agent.clear_all()
        for _ in range(self.n_update):
            action,prob,value = self.agent.act(self.states)
            next_state, reward, done, _ = self.env.step(action)



            eps_time += 1
            total_reward += reward

            if self.training_mode:
                self.agent.save_eps(self.states.tolist(), action, reward, float(done), next_state.tolist(),prob.tolist(),value)

            self.states = next_state

            if self.render:
                self.env.render()

            if done:
                self.states = self.env.reset()
                i_episode += 1
                print('Episode {} \t t_reward: {} \t time: {} \t process no: {} \t'.format(i_episode, total_reward,
                                                                                           eps_time, self.tag))

                total_reward = 0
                eps_time = 0

        return self.agent.get_all(), i_episode, total_reward, eps_time, self.tag


def plot(datas):
    print('----------')

    plt.plot(datas)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Datas')
    plt.show()

    print('Max :', np.max(datas))
    print('Min :', np.min(datas))
    print('Avg :', np.mean(datas))


def main():
    ############## Hyperparameters ##############
    training_mode = True  # If you want to train the agent, set this to True. But set this otherwise if you only want to test it

    render = False  # If you want to display the image, set this to True. Turn this off if you run this in Google Collab
    n_update = 128  # How many episode before you update the Policy. Recommended set to 1024 for Continous
    n_episode = 100000  # How many episode you want to run
    n_agent = 6  # How many agent you want to run asynchronously

    clip_coef=0.2
    max_grad_norm=0.5
    policy_kl_range = 0.0008  # Recommended set to 0.03 for Continous
    policy_params = 20  # Recommended set to 5 for Continous
    value_clip = 1.0  # How many value will be clipped. Recommended set to the highest or lowest possible reward
    entropy_coef = 0.01  # How much randomness of action you will get. Because we use Standard Deviation for Continous, no need to use Entropy for randomness
    vf_loss_coef = 0.5  # Just set to 1
    minibatch = 4  # How many batch per update. size of batch = n_update / minibatch. Recommended set to 32 for Continous
    PPO_epochs = 4  # How many epoch per update. Recommended set to 10 for Continous

    gamma = 0.99  # Just set to 0.99
    lam = 0.95  # Just set to 0.95
    learning_rate = 2.5e-4  # Just set to 0.95
    #############################################
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    learner = Learner(state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef,
                      vf_loss_coef,clip_coef,max_grad_norm,
                      minibatch, PPO_epochs, gamma, lam, learning_rate)
    #############################################
    start = time.time()
    ray.init()
    try:
        runners = [Runner.remote(env_name, training_mode, render, n_update, i) for i in range(n_agent)]
        learner.save_weights()

        episode_ids = []
        for i, runner in enumerate(runners):
            episode_ids.append(runner.run_episode.remote(i, 0, 0))
            time.sleep(1)

        for _ in range(1, n_episode + 1):
            ready, not_ready = ray.wait(episode_ids)
            trajectory, i_episode, total_reward, eps_time, tag = ray.get(ready)[0]

            states, actions, rewards, dones, next_states,probs,values = trajectory
            learner.save_all(states, actions, rewards, dones, next_states,probs,values)

            learner.update_ppo()
            learner.save_weights()

            episode_ids = not_ready
            episode_ids.append(runners[tag].run_episode.remote(i_episode, total_reward, eps_time))

            gc.collect()
    except KeyboardInterrupt:
        print('\nTraining has been Shutdown \n')
    finally:
        ray.shutdown()

        finish = time.time()
        timedelta = finish - start
        print('Timelength: {}'.format(str(datetime.timedelta(seconds=timedelta))))


if __name__ == '__main__':
    main()