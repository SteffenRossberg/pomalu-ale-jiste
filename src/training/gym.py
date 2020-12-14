import torch
import numpy as np
from torch import nn as nn


class Gym:

    def __init__(self, manager):
        self.manager = manager
        self.device = self.manager.device

    def train_auto_encoder(self, agent_name, agent, agent_optimizer, train_data, min_agent_loss,
                           steps_to_wait=100, batch_size=5000):
        objective = nn.MSELoss()
        steps = 0
        epoch = 0
        while True:
            epoch += 1
            agent_loss = self.__train_auto_encoder_run(train_data, agent, agent_optimizer, objective, batch_size)
            if agent_loss < min_agent_loss:
                min_agent_loss = float(agent_loss)
                steps = 0
                self.manager.save_net(f'{agent_name}.autoencoder', agent, agent_optimizer, min_agent_loss)
                self.manager.save_net(f'{agent_name}.encoder', agent.encoder, loss=min_agent_loss)
                self.manager.save_net(f'{agent_name}.decoder', agent.decoder, loss=min_agent_loss)
                print(f'{steps:4} - {agent_name}: {min_agent_loss:.7f} ... saved')
            elif steps_to_wait > steps:
                print(f'{steps:4} - {agent_name}: {agent_loss:.7f}')
                steps += 1
            else:
                return

    def __train_auto_encoder_run(self, train_data, agent, agent_optimizer, objective, batch_size):
        train_data = np.array(train_data, dtype=np.float32)
        np.random.shuffle(train_data)
        losses = []
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + (batch_size if len(train_data) - i > batch_size else len(train_data) - i)]
            batch = torch.from_numpy(batch).to(self.device)
            # train auto encoder
            agent_optimizer.zero_grad()
            results = agent(batch)
            agent_loss = objective(results, batch)
            agent_loss.backward()
            agent_optimizer.step()
            losses.append(agent_loss.item())
        return np.mean(np.array(losses))
