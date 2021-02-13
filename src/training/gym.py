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
from src.networks.models import TrainClassifier, AutoEncoder


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
        self.RL_EPSILON_STEPS = 100_000
        self.RL_REWARD_STEPS = 2
        self.RL_MAX_LEARN_STEPS_WITHOUT_CHANGE = 20
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
            auto_encoder: AutoEncoder,
            optimizer,
            features,
            result,
            max_epochs=None,
            max_steps=100,
            batch_size=5000):

        def save(manager: NetManager, loss):
            manager.save_net(name, auto_encoder)
            manager.save_optimizer(name, optimizer, loss)

        def output(epoch, step, loss, accuracy, is_saved):
            Gym.print_step(epoch, step, f'{name}.auto.encoder', loss, accuracy, is_saved)

        def calculate_accuracies(x, y):
            x = x + 0.0000000001
            ratio = y / x
            ratio = ratio - 1.0
            ratio = np.abs(ratio)
            mean_ratio = np.mean(ratio, axis=(2, 3))
            mean_ratio = 1.0 - mean_ratio
            accuracies = np.where((mean_ratio < 0.0) | (mean_ratio > 1.0), 0.0, mean_ratio)
            accuracies = accuracies.flatten()
            return accuracies

        objective = nn.MSELoss()
        return self.train(
            name,
            auto_encoder,
            optimizer,
            objective,
            features,
            features,
            result,
            max_epochs,
            max_steps,
            batch_size,
            save,
            output,
            calculate_accuracies=calculate_accuracies)

    def train_classifier(
            self,
            name,
            trader,
            optimizer,
            features,
            labels,
            result,
            max_epochs=None,
            max_steps=20,
            batch_size=5000):

        def save(manager: NetManager, loss):
            manager.save_net(name, trader.classifier)
            manager.save_optimizer(name, optimizer, loss)

        def output(epoch, step, loss, accuracy, is_saved):
            Gym.print_step(epoch, step, name, loss, accuracy, is_saved)

        def calculate_accuracies(actual, expected):
            actual = np.argmax(actual, axis=-1)
            return np.array([1 if actual[i] == expected[i] else 0 for i in range(len(actual))])

        objective = nn.CrossEntropyLoss()
        agent = TrainClassifier(trader)
        return self.train(
            name,
            agent,
            optimizer,
            objective,
            features,
            labels,
            result,
            max_epochs,
            max_steps,
            batch_size,
            save,
            output,
            calculate_accuracies=calculate_accuracies)

    def train(
            self,
            name,
            agent,
            optimizer,
            objective,
            features,
            labels,
            result,
            max_epochs,
            max_steps,
            batch_size,
            save,
            output,
            calculate_accuracies=None):

        best_loss_gauge = self.get_gauge(
            f'train_{name}_best_loss',
            f'Best value of {name} training')
        current_loss_gauge = self.get_gauge(
            f'train_{name}_current_loss',
            f'Current value of {name} training')
        best_accuracy_gauge = self.get_gauge(
            f'train_{name}_best_accuracy',
            f'Best accuracy of {name} training')
        current_accuracy_gauge = self.get_gauge(
            f'train_{name}_current_accuracy',
            f'Current accuracy of {name} training')

        step = 0
        epoch = 0
        train_features, train_labels, val_features, val_labels = self.split_train_and_val(features, labels, 0.75)
        best_accuracy = 0.0
        while True:
            epoch += 1
            loss, accuracy = \
                self.train_run(
                    train_features,
                    train_labels,
                    val_features,
                    val_labels,
                    agent,
                    optimizer,
                    objective,
                    batch_size,
                    calculate_accuracies)
            loss = float(loss)
            is_saved = False
            if best_accuracy < accuracy:
                save(self.manager, result)
                is_saved = True
                best_accuracy = accuracy
                best_accuracy_gauge.set(accuracy)
                step = 0
            if loss < result:
                result = loss
                best_loss_gauge.set(loss)
                step = 0
            elif max_steps > step:
                step += 1
            else:
                return result
            output(epoch, step, loss, accuracy, is_saved)
            current_loss_gauge.set(loss)
            current_accuracy_gauge.set(accuracy)
            if max_epochs is not None and epoch >= max_epochs:
                return result

    @staticmethod
    def split_train_and_val(features, labels, rate=0.75):
        features = torch.tensor(features)
        labels = torch.tensor(labels)
        train_count = int(features.shape[0] * rate)
        train_features = features[:train_count]
        train_labels = labels[:train_count]
        val_features = features[train_count:]
        val_labels = labels[train_count:]
        return train_features, train_labels, val_features, val_labels

    def train_run(
            self,
            train_features,
            train_labels,
            val_features,
            val_labels,
            agent,
            agent_optimizer,
            objective,
            batch_size,
            calculate_accuracies):

        # noinspection PyUnusedLocal
        def dummy_accuracy(current, exp):
            return 0.0

        # train net
        random = torch.randperm(len(train_features))
        features = torch.tensor(train_features)[random]
        labels = torch.tensor(train_labels)[random]
        agent.train()
        for start in range(0, len(features), batch_size):
            stop = start + (batch_size if len(features) - start > batch_size else len(features) - start)
            batch_features = features[start:stop].to(self.device)
            batch_labels = labels[start:stop].to(self.device)
            agent_optimizer.zero_grad()
            prediction = agent(batch_features)
            agent_loss = objective(prediction, batch_labels)
            agent_loss.backward()
            agent_optimizer.step()
        # validate net
        random = torch.randperm(len(val_features))
        features = torch.tensor(val_features)[random]
        labels = torch.tensor(val_labels)[random]
        losses = []
        accuracies = []
        if calculate_accuracies is None:
            calculate_accuracies = dummy_accuracy
        agent.eval()
        for start in range(0, len(features), batch_size):
            stop = start + (batch_size if len(features) - start > batch_size else len(features) - start)
            batch_features = features[start:stop].to(self.device)
            batch_labels = labels[start:stop].to(self.device)
            prediction = agent(batch_features)
            agent_loss = objective(prediction, batch_labels)
            losses.append(agent_loss.item())
            actual = prediction.cpu().detach().numpy()
            expected = batch_labels.cpu().detach().numpy()
            accuracies = np.concatenate((accuracies, calculate_accuracies(actual, expected)), axis=0)
        return np.mean(losses), np.mean(accuracies) * 100.0

    @staticmethod
    def print_step(epoch, step, name, loss, accuracy, is_saved):
        message = f"{epoch:4} - {step:4} - {name}: {loss:.7f}: {accuracy:.3f}% {' ... saved' if is_saved else ''}"
        print(message)

    def train_trader(
            self,
            name,
            trader,
            optimizer,
            result,
            train_stock_exchange,
            validation_stock_exchange):

        def save(manager: NetManager, loss):
            manager.save_net(name, trader.decision_maker)
            manager.save_optimizer(name, optimizer, loss)

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
        profit_rates = []
        trader.train(mode=True)
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

                if mean_val > 0:
                    means = self.validation_run(validation_stock_exchange, trader, 20)
                    mean_profit_rate = means['order_profit_rates']
                    profit_rates.append(mean_profit_rate)
                    if result < mean_profit_rate:
                        print(f"{step_index:6}:{str(train_stock_exchange.train_level)} " +
                              f"Mean profit rate updated {result:.2f} -> {mean_profit_rate:.2f}")
                        result = mean_profit_rate
                        save(self.manager, result)
                        best_trader_value_gauge.set(result)
                    else:
                        print(f"{step_index:6}:{str(train_stock_exchange.train_level)} " +
                              f"Mean profit rate: {mean_profit_rate:.2f}")
                    if self.is_upper_outlier(mean_profit_rate, profit_rates, 10):
                        target_net.sync()
                    current_trader_value_gauge.set(mean_profit_rate)
                elif step_index % self.RL_TARGET_NET_SYNC == 0:
                    target_net.sync()

            optimizer.zero_grad()
            batch = experience_buffer.sample(self.RL_BATCH_SIZE)
            loss_v = self.calculate_loss(batch,
                                         trader,
                                         target_net.target_model,
                                         criterion,
                                         self.RL_GAMMA ** self.RL_REWARD_STEPS)
            loss_v.backward()
            optimizer.step()

        return result

    def is_upper_outlier(self, mean_profit_rate, profit_rates, count):
        if len(profit_rates) < count:
            return False
        profit_rates = profit_rates[-count:]
        _, _, upper_threshold = self.calculate_outliers(profit_rates)
        return upper_threshold <= mean_profit_rate

    def calculate_outliers(self, data):
        sorted_data = np.sort(data)
        median = self.calc_median(sorted_data)
        q1 = self.calc_quartil(sorted_data, 1)
        q3 = self.calc_quartil(sorted_data, 3)
        q_range = (q3 - q1) * 1.25
        lower_threshold = q1 - q_range
        upper_threshold = q3 + q_range
        return lower_threshold, median, upper_threshold

    @staticmethod
    def calc_median(sorted_data):
        count = len(sorted_data)
        if count % 2 == 0:
            mid2 = int(count / 2)
            mid1 = mid2 - 1
            return (sorted_data[mid1] + sorted_data[mid2]) / 2
        mid = count // 2
        return sorted_data[mid]

    @staticmethod
    def calc_quartil(sorted_data, position):
        count = len(sorted_data)
        index = (count // 4) * position - 1
        return sorted_data[index]

    def validation_run(self, env: StockExchange, trader, episodes=100):
        trader.eval()
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
        trader.train()
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
