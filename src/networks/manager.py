import torch
import os
import torch.optim as optim
from src.networks.models import Trader
import numpy as np
import random


class NetManager:

    def __init__(self, device, seed, deterministic, data_directory='data/networks'):
        self.net_device = device
        self.seed = seed
        self.deterministic = deterministic
        self.data_directory = data_directory
        os.makedirs(f'{self.data_directory}', exist_ok=True)

    @property
    def device(self):
        return self.net_device

    # noinspection PyMethodMayBeStatic,PyUnresolvedReferences
    def init_seed(self, seed, deterministic=True):
        print("Init random seed ...")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def create_trader(self, days, state_size=2):
        trader = Trader(days, state_size).to(self.device)
        return trader

    def load_net(self, file_name, net):
        file_path = f'{self.data_directory}/{file_name}.pt'
        if os.path.exists(file_path):
            state = torch.load(file_path)
            net.load_state_dict(state)
            net.eval()
            return True
        return False

    def save_net(self, file_name, net):
        file_path = f'{self.data_directory}/{file_name}.pt'
        torch.save(net.state_dict(), file_path)

    def load(self, name, net, optimizer, reset, create_optimizer, default):
        if self.load_net(name, net):
            return self.load_optimizer(name, optimizer, default)
        self.init_seed(self.seed, self.deterministic)
        reset(self.net_device)
        optimizer, result = create_optimizer()
        return optimizer, result

    @staticmethod
    def create_buyer_optimizer(trader):
        return optim.Adam(trader.buyer.parameters()), 100.0

    @staticmethod
    def create_seller_optimizer(trader):
        return optim.Adam(trader.seller.parameters()), 100.0

    @staticmethod
    def create_classifier_optimizer(trader):
        return optim.Adam(trader.classifier.parameters()), 100.0

    def load_optimizer(self, file_name, optimizer, default_result):
        file_path = f'{self.data_directory}/{file_name}.optimizer.pt'
        result = default_result
        if os.path.exists(file_path):
            data = torch.load(file_path)
            optimizer.load_state_dict(data['state'])
            result = data['result']
        return optimizer, result

    def save_optimizer(self, file_name, optimizer, result):
        file_path = f'{self.data_directory}/{file_name}.optimizer.pt'
        if os.path.exists(file_path):
            data = torch.load(file_path)
            data['state'] = optimizer.state_dict()
            data['result'] = result
        else:
            data = {'state': optimizer.state_dict(), 'result': result}
        torch.save(data, file_path)
