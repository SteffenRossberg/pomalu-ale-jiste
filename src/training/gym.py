import torch
import numpy as np
from torch import nn as nn
from ptan.agent import TargetNet, DQNAgent
from ptan.actions import EpsilonGreedyActionSelector
from ptan.experience import ExperienceSourceFirstLast, ExperienceReplayBuffer
from prometheus_client import Gauge
from src.environment.stock_exchange import StockExchange
from src.environment.enums import Actions
from src.networks.manager import NetManager


class Gym:

    def __init__(self, manager: NetManager):
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
        self.RL_MAX_LEARN_STEPS_WITHOUT_CHANGE = 50
        self.gauges = {}

    def get_gauge(self, name, description):
        if name not in self.gauges:
            self.gauges[name] = Gauge(
                name,
                description)
        return self.gauges[name]

    def train_auto_encoder(
            self,
            name,
            trader,
            agent,
            optimizers,
            features,
            results,
            max_steps=100,
            batch_size=5000):

        def save(manager: NetManager):
            manager.save_trader('trader', trader)
            manager.save_optimizers('trader', optimizers, results)

        def output(epoch, step, loss, is_saved):
            Gym.print_step(epoch, step, f'{name}.auto.encoder', loss, is_saved)

        best_value_gauge = self.get_gauge(
            f'train_{name}_auto_encoder_best_value',
            f'Best value of {name}_auto_encoder training')
        current_value_gauge = self.get_gauge(
            f'train_{name}_auto_encoder_current_value',
            f'Current value of {name}_auto_encoder training')

        objective = nn.MSELoss()
        self.train(
            name,
            agent,
            optimizers,
            objective,
            features,
            features,
            results,
            max_steps,
            batch_size,
            save,
            output,
            best_value_gauge,
            current_value_gauge)

    def train(
            self,
            name,
            agent,
            optimizers,
            objective,
            features,
            labels,
            results,
            max_steps,
            batch_size,
            save,
            output,
            best_value_gauge,
            current_value_gauge):
        step = 0
        epoch = 0
        while True:
            epoch += 1
            agent_loss = \
                self.train_run(
                    features,
                    labels,
                    agent,
                    optimizers[name],
                    objective,
                    batch_size)
            agent_loss = float(agent_loss)
            if agent_loss < results[name]:
                results[name] = agent_loss
                step = 0
                save(self.manager)
                output(epoch, step, results[name], True)
                best_value_gauge.set(results[name])
            elif max_steps > step:
                output(epoch, step, agent_loss, False)
                step += 1
            else:
                return
            current_value_gauge.set(agent_loss)

    def train_run(
            self,
            features,
            labels,
            agent,
            agent_optimizer,
            objective,
            batch_size):
        random = torch.randperm(len(features))
        features = torch.tensor(features)[random]
        labels = torch.tensor(labels)[random]
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

    def train_trader(
            self,
            name,
            trader,
            optimizers,
            results,
            train_stock_exchange,
            validation_stock_exchange):

        def save(manager: NetManager, suffix=None):
            suffix = f'.{suffix}' if suffix is not None else ''
            manager.save_trader(f'trader{suffix}', trader)
            manager.save_optimizers(f'trader{suffix}', optimizers, results)

        best_value_gauge = self.get_gauge(
            f'train_{name}_best_value',
            f'Best value of {name} training')
        current_value_gauge = self.get_gauge(
            f'train_{name}_current_value',
            f'Current value of {name} training')

        best_trader_value_gauge = self.get_gauge(
            'train_trader_best_value',
            'Best value of trader training')
        current_trader_value_gauge = self.get_gauge(
            'train_trader_current_value',
            'Current value of trader training')

        optimizer = optimizers[name]
        criterion = nn.MSELoss()
        target_net = TargetNet(trader)
        selector = EpsilonGreedyActionSelector(self.RL_EPSILON_START)
        agent = DQNAgent(trader, selector, device=self.device)
        experience_source = ExperienceSourceFirstLast(
            train_stock_exchange,
            agent,
            self.RL_GAMMA,
            steps_count=self.RL_REWARD_STEPS)
        experience_buffer = ExperienceReplayBuffer(experience_source, self.RL_REPLAY_SIZE)

        step_index = 0
        evaluation_states = None
        learn_step = 0
        best_mean_val = 0
        profit_rate_counter = 0
        while learn_step < self.RL_MAX_LEARN_STEPS_WITHOUT_CHANGE:
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
                mean_val = self.calculate_values_of_states(evaluation_states, trader)
                if best_mean_val is None or best_mean_val < mean_val:
                    if best_mean_val is not None:
                        print(f"{step_index:6}:{learn_step:4}:{train_stock_exchange.train_level} " +
                              f"Mean value updated {best_mean_val:.3f} -> {mean_val:.3f}")
                    best_mean_val = mean_val
                    learn_step = 0
                    best_value_gauge.set(best_mean_val)
                else:
                    print(f"{step_index:6}:{learn_step:4}:{train_stock_exchange.train_level} " +
                          f"Mean value {mean_val:.3f}")
                    learn_step += 1
                current_value_gauge.set(mean_val)

                means = self.validation_run(validation_stock_exchange, trader, 50)
                mean_profit_rate = means['order_profit_rates']
                if results[name] < mean_profit_rate:
                    print(f"{step_index:6}:{str(train_stock_exchange.train_level)} " +
                          f"Mean profit rate updated {results[name]:.2f} -> {mean_profit_rate:.2f}")
                    save(self.manager)
                    results[name] = mean_profit_rate
                    best_trader_value_gauge.set(results[name])
                else:
                    print(f"{step_index:6}:{str(train_stock_exchange.train_level)} " +
                          f"Mean profit rate: {mean_profit_rate:.2f}")

                current_trader_value_gauge.set(mean_profit_rate)

                profit_rate_counter = (profit_rate_counter + 1) if mean_profit_rate > 0.0 else 0
                if profit_rate_counter >= 10:
                    break

            optimizer.zero_grad()
            batch = experience_buffer.sample(self.RL_BATCH_SIZE)
            loss_v = self.calculate_loss(batch,
                                         trader,
                                         target_net.target_model,
                                         criterion,
                                         self.RL_GAMMA ** self.RL_REWARD_STEPS)
            loss_v.backward()
            optimizer.step()

            if step_index % self.RL_TARGET_NET_SYNC == 0:
                target_net.sync()

        save(self.manager, 'last')

    def validation_run(self, env: StockExchange, trader, episodes=100):
        stats = {
            'episode_reward': [],
            'episode_steps': [],
            'order_profits': [],
            'order_profit_rates': [],
            'order_steps': [],
        }

        for episode in range(episodes):
            obs = env.reset()

            total_reward = 0.0
            position_steps = None
            episode_steps = 0

            while True:
                features = torch.tensor([obs], dtype=torch.float32).to(self.device)
                prediction = trader(features)
                action_idx = prediction.max(dim=1)[1].item()
                obs, reward, done, _ = env.step(action_idx)
                action = Actions(action_idx)
                if action == Actions.Buy and position_steps is None:
                    position_steps = 0
                elif action == Actions.Sell and position_steps is not None:
                    stats['order_profits'].append(env.state.profit)
                    stats['order_profit_rates'].append(env.state.profit_rate)
                    stats['order_steps'].append(position_steps)
                    position_steps = None
                total_reward += reward
                episode_steps += 1
                if position_steps is not None:
                    position_steps += 1
                if done:
                    break

            stats['episode_reward'].append(total_reward)
            stats['episode_steps'].append(episode_steps)

        return {key: np.mean(vals) for key, vals in stats.items()}

    def calculate_loss(
            self,
            batch,
            model,
            target_model,
            criterion,
            gamma):
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
