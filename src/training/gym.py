import torch
import numpy as np
from torch import nn as nn


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
                         min_loss, max_steps=100, batch_size=5000):

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
