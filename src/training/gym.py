import torch
import numpy as np
from torch import nn as nn
from ptan.agent import TargetNet


class Gym:

    def __init__(self, manager):
        self.manager = manager
        self.device = self.manager.device

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

    def train_decision_maker(self,
                             name,
                             model,
                             optimizer,
                             rl_frames,
                             max_yield,
                             gamma=0.99,
                             max_epochs=5,
                             target_net_sync=1_000,
                             start_investment=50_000):
        def save(manager, loss_value):
            manager.save_net(f'{name}.decision_maker', model, optimizer, loss=loss_value)

        criterion = nn.MSELoss()
        target_net = TargetNet(model)
        epoch = 0
        tax_rate = 0.25 * 1.055
        fee = 1.0
        step = 0
        target_net.sync()
        while epoch < max_epochs:
            for frame in rl_frames:
                dates = frame['dates']
                prices = frame['prices']
                windows = frame['windows']
                last_days = frame['last_days']
                buy_price = 0.0
                sell_price = prices[0]
                rewards = 0.0
                investment = start_investment
                count = 0
                for index in range(len(prices)):
                    optimizer.zero_grad()
                    date = dates[index]
                    price = prices[index]
                    last_day = last_days[index][-1]
                    window = torch.tensor([windows[index]], dtype=torch.float32, device=self.device)
                    state = [((investment + count * price) / start_investment) - 1.0, last_day]
                    expected_prediction = target_net.target_model(window, state=state).cpu()
                    test_prediction = model(window, state=state).cpu()
                    test_decision = np.argmax(test_prediction.detach().numpy())
                    current_rewards = self.calculate_rewards(test_decision,
                                                             count,
                                                             buy_price,
                                                             sell_price,
                                                             price)
                    if test_decision == 1 and count == 0:
                        buy_price = price
                        investment -= fee
                        count = int(investment / buy_price)
                        if count == 0:
                            break
                        investment -= count * buy_price
                        message = f'{date} - {frame["ticker"]:5} - {frame["company"]:40} - '
                        message += f'buy:  {count:5} x ${buy_price:7.2f} = ${count * buy_price:.2f}'
                        print(message)
                        sell_price = 0.0
                    elif test_decision == 2 and count > 0:
                        sell_price = price
                        new_investment = Gym.calculate_investment(investment, count, buy_price, price, tax_rate)
                        message = f'{date} - {frame["ticker"]:5} - {frame["company"]:40} - '
                        message += f'sell: {count:5} x ${price:7.2f} = ${count * price:.2f} '
                        message += f'(${new_investment - (count * buy_price + investment):.2f} / '
                        message += f'{(new_investment / (count * buy_price + investment) - 1.0) * 100.0:.2f}%)'
                        print(message)
                        investment = new_investment
                        count = 0
                        buy_price = 0.0
                    expected_prediction = expected_prediction.detach() * (gamma ** 2) + current_rewards
                    loss = criterion(test_prediction, expected_prediction)
                    loss.backward()
                    optimizer.step()
                    rewards += current_rewards
                    step += 1
                    if step % target_net_sync == 0:
                        target_net.sync()
                years = len(prices) / 250.0
                investment += count * buy_price
                current_yield = ((investment / start_investment) ** (1.0 / years) - 1.0) * 100.0
                if current_yield > max_yield:
                    max_yield = current_yield
                    save(self.manager, max_yield)
                message = f'{frame["ticker"]} - {frame["company"]} - '
                message += f'rate of return: {(investment / start_investment - 1.0) * 100.0:6.2f}% - '
                message += f'yield: {current_yield:6.2f}% p.a.'
                print(message)
            epoch += 1

    @staticmethod
    def calculate_rewards(decision,
                          count,
                          buy_price,
                          sell_price,
                          current_price):
        reward = -10.0
        if decision == 1:
            if count == 0:
                reward = (((current_price / sell_price) - 1.0) * -1.0) * 100.0
        elif decision == 2:
            if count > 0:
                reward = ((current_price / buy_price) - 1.0) * 100.0
        elif decision == 0:
            if count == 0:
                reward = (((current_price / sell_price) - 1.0) * -1.0) * 49.0
            else:
                reward = ((current_price / buy_price) - 1.0) * 49.0
        return reward

    @staticmethod
    def calculate_investment(investment, count, buy_price, current_price, tax_rate):
        investment -= 1.0
        profit = ((count * current_price) - (count * buy_price))
        if profit > 0.0:
            profit *= (1.0 - tax_rate)
        investment += count * buy_price + profit
        return investment

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
