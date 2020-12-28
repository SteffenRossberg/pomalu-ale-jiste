import torch
import numpy as np
from torch import nn as nn
from ptan.agent import TargetNet, DQNAgent
from ptan.actions import EpsilonGreedyActionSelector
from ptan.experience import ExperienceSourceFirstLast, ExperienceReplayBuffer


class Gym:

    def __init__(self, manager):
        self.manager = manager
        self.device = self.manager.device
        self.RL_BATCH_SIZE = 32
        self.RL_REPLAY_SIZE = 100_000
        self.RL_REPLAY_INITIAL = 10_000
        self.RL_STATES_TO_EVALUATE = 1_000
        self.RL_EVAL_EVERY_STEP = 100
        self.RL_TARGET_NET_SYNC = 1_000
        self.RL_GAMMA = 0.99
        self.RL_EPSILON_START = 1.0
        self.RL_EPSILON_STOP = 0.1
        self.RL_EPSILON_STEPS = 1_000_000
        self.RL_REWARD_STEPS = 2
        self.RL_MAX_STEPS = 500_000

    def train_auto_encoder(self, name, agent, optimizer, features, min_loss, max_steps=100, batch_size=5000):

        def save(manager, loss):
            manager.save_net(f'{name}.autoencoder', agent, optimizer, loss=loss)
            manager.save_net(f'{name}.encoder', agent.encoder, loss=loss)
            manager.save_net(f'{name}.decoder', agent.decoder, loss=loss)

        def output(epoch, step, loss, is_saved):
            Gym.print_step(epoch, step, f'{name}.autoencoder', loss, is_saved)

        objective = nn.MSELoss()
        self.train(agent, optimizer, objective, features, features, min_loss, max_steps, batch_size, save, output)

    def train_classifier(self, name, agent, optimizer,
                         signal_features, signal_labels,
                         none_signal_features, none_signal_labels,
                         min_loss, max_steps=100, batch_size=50000):

        def save(manager, loss):
            manager.save_net(f'{name}.classifier', agent, optimizer, loss=loss)

        def output(epoch, step, loss, is_saved):
            Gym.print_step(epoch, step, f'{name}.classifier', loss, is_saved)

        objective = nn.CrossEntropyLoss()
        self.train(agent, optimizer, objective, signal_features, signal_labels, min_loss, max_steps, batch_size,
                   save, output, none_signal_features, none_signal_labels)

    def train(self, agent, optimizer, objective, features, labels, min_loss, max_steps, batch_size, save, output,
              none_features=None, none_labels=None):
        step = 0
        epoch = 0
        while True:
            epoch += 1
            agent_loss = self.train_run(features, labels, none_features, none_labels, agent, optimizer, objective,
                                        batch_size)
            if agent_loss < min_loss:
                min_loss = float(agent_loss)
                step = 0
                save(self.manager, min_loss)
                output(epoch, step, min_loss, True)
            elif max_steps > step:
                output(epoch, step, agent_loss, False)
                step += 1
            else:
                return

    def train_run(self, features, labels, none_features, none_labels, agent, agent_optimizer, objective, batch_size):
        if none_features is not None and none_labels is not None:
            random = torch.randperm(len(none_features))
            none_features = torch.from_numpy(none_features)[random]
            none_labels = torch.from_numpy(none_labels)[random]
            none_features = none_features.numpy()[:len(features)]
            none_labels = none_labels.numpy()[:len(features)]
            features = np.concatenate((features, none_features))
            labels = np.concatenate((labels, none_labels))
        random = torch.randperm(len(features))
        features = torch.from_numpy(features)[random].to(self.device)
        labels = torch.from_numpy(labels)[random].to(self.device)
        losses = []
        for start in range(0, len(features), batch_size):
            stop = start + (batch_size if len(features) - start > batch_size else len(features) - start)
            batch_features = features[start:stop].to(self.device)
            batch_labels = labels[start:stop].to(self.device)
            # train net
            agent_optimizer.zero_grad()
            prediction = agent(batch_features)
            agent_loss = objective(prediction, batch_labels)
            agent_loss.backward()
            agent_optimizer.step()
            losses.append(agent_loss.item())
        return np.mean(np.array(losses))

    @staticmethod
    def print_step(epoch, step, name, loss, is_saved):
        message = f"{epoch:4} - {step:4} - {name}: {loss:.7f}{' ... saved' if is_saved else ''}"
        print(message)

    def train_decision_maker(self,
                             name,
                             model,
                             optimizer,
                             best_mean_val,
                             stock_exchange):
        def save(manager, loss_value):
            manager.save_net(f'{name}.decision_maker', model, optimizer, loss=loss_value)

        criterion = nn.MSELoss()
        target_net = TargetNet(model)
        selector = EpsilonGreedyActionSelector(self.RL_EPSILON_START)
        agent = DQNAgent(model, selector, device=self.device)
        experience_source = ExperienceSourceFirstLast(stock_exchange,
                                                      agent,
                                                      self.RL_GAMMA,
                                                      steps_count=self.RL_REWARD_STEPS)
        experience_buffer = ExperienceReplayBuffer(experience_source, self.RL_REPLAY_SIZE)

        step_index = 0
        evaluation_states = None

        while self.RL_MAX_STEPS > step_index:
            step_index += 1
            experience_buffer.populate(1)
            selector.epsilon = max(self.RL_EPSILON_STOP, self.RL_EPSILON_START - step_index / self.RL_EPSILON_STEPS)

            if len(experience_buffer) < self.RL_REPLAY_INITIAL:
                continue

            if evaluation_states is None:
                print("Initial buffer populated, start training")
                evaluation_states = experience_buffer.sample(self.RL_STATES_TO_EVALUATE)
                evaluation_states = [np.array(transition.state, copy=False) for transition in evaluation_states]
                evaluation_states = np.array(evaluation_states, copy=False)

            if step_index % self.RL_EVAL_EVERY_STEP == 0:
                mean_val = self.calculate_values_of_states(evaluation_states, model)
                if best_mean_val is None or best_mean_val < mean_val:
                    if best_mean_val is not None:
                        print(f"{step_index}: Best mean value updated {best_mean_val:.7f} -> {mean_val:.7f}")
                    best_mean_val = mean_val
                    save(self.manager, best_mean_val)
                else:
                    print(f"{step_index}: Mean value {mean_val:.7f}")

            optimizer.zero_grad()
            batch = experience_buffer.sample(self.RL_BATCH_SIZE)
            loss_v = self.calculate_loss(batch,
                                         model,
                                         target_net.target_model,
                                         criterion,
                                         self.RL_GAMMA ** self.RL_REWARD_STEPS)
            loss_v.backward()
            optimizer.step()

            if step_index % self.RL_TARGET_NET_SYNC == 0:
                target_net.sync()

    def calculate_loss(self, batch, model, target_model, criterion, gamma):
        states, actions, rewards, dones, next_states = self.unpack_batch(batch)

        states_v = torch.tensor(states).to(self.device)
        next_states_v = torch.tensor(next_states).to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.BoolTensor(dones).to(self.device)

        state_action_values = model(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_actions = model(next_states_v).max(1)[1]
        next_state_values = target_model(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
        next_state_values[done_mask] = 0.0

        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
        return criterion(state_action_values, expected_state_action_values)

    def calculate_values_of_states(self, states, model):
        mean_values = []
        for batch in np.array_split(states, 64):
            mean_value = self.calculate_mean_value_of_states(batch, model)
            mean_values.append(mean_value)
        return np.mean(mean_values)

    def calculate_mean_value_of_states(self, batch, model):
        states_v = torch.tensor(batch).to(self.device)
        action_values_v = model(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_value = best_action_values_v.mean().item()
        return mean_value

    @staticmethod
    def unpack_batch(batch):
        actions = [exp.action for exp in batch]
        rewards = [exp.reward for exp in batch]
        dones = [exp.last_state is None for exp in batch]
        states = [np.array(exp.state, copy=False) for exp in batch]
        last_states = [(np.array(exp.state, copy=False)
                        if exp.last_state is None
                        else np.array(exp.last_state, copy=False)) for exp in batch]
        return \
            np.array(states, copy=False), \
            np.array(actions), \
            np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.bool), \
            np.array(last_states, copy=False)
