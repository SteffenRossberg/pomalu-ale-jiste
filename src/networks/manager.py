import torch
import os
import torch.optim as optim
from src.networks.models import Trader
import numpy as np
import random

class NetManager:

    def __init__(self, device, data_directory='data/networks'):
        self.net_device = device
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

    def create_trader(self, days, state_size=7):
        trader = Trader(days, state_size).to(self.device)
        return trader

    def load_trader(self, file_name, trader):
        file_path = f'{self.data_directory}/{file_name}.pt'
        if os.path.exists(file_path):
            state = torch.load(file_path)
            trader.load_state_dict(state)
            trader.eval()
            return True
        return False

    def save_trader(self, file_name, trader):
        file_path = f'{self.data_directory}/{file_name}.pt'
        torch.save(trader.state_dict(), file_path)

    @staticmethod
    def create_buyer_optimizer(trader):
        return optim.Adam(trader.buyer.parameters()), 100.0

    @staticmethod
    def create_seller_optimizer(trader):
        return optim.Adam(trader.seller.parameters()), 100.0

    @staticmethod
    def create_decision_maker_optimizer(trader):
        return optim.Adam(trader.decision_maker.parameters(), lr=0.0001), -100.0

    def create_optimizers(self, trader):
        buyer_optimizer, buyer_loss = self.create_buyer_optimizer(trader)
        seller_optimizer, seller_loss = self.create_seller_optimizer(trader)
        decision_maker_optimizer, decision_maker_loss = self.create_decision_maker_optimizer(trader)
        optimizers = {
            'buyer': buyer_optimizer,
            'seller': seller_optimizer,
            'decision_maker': decision_maker_optimizer
        }
        default_results = {
            'buyer': buyer_loss,
            'seller': seller_loss,
            'decision_maker': decision_maker_loss
        }
        return optimizers, default_results

    def load_optimizers(self, file_name, optimizers, default_results):
        file_path = f'{self.data_directory}/{file_name}.optimizers.pt'
        results = {optimizer_id: default_results[optimizer_id] for optimizer_id in optimizers.keys()}
        if os.path.exists(file_path):
            data = torch.load(file_path)
            for optimizer_id, optimizer in optimizers.items():
                if optimizer_id in data:
                    optimizer.load_state_dict(data[optimizer_id]['state'])
                    results[optimizer_id] = data[optimizer_id]['result']
        return results

    def save_optimizers(self, file_name, optimizers, results):
        file_path = f'{self.data_directory}/{file_name}.optimizers.pt'
        if os.path.exists(file_path):
            data = torch.load(file_path)
            for optimizer_id, optimizer in optimizers.items():
                data[optimizer_id]['state'] = optimizer.state_dict()
                data[optimizer_id]['result'] = results[optimizer_id]
        else:
            data = {
                optimizer_id: {'state': optimizer.state_dict(), 'result': results[optimizer_id]}
                for optimizer_id, optimizer in optimizers.items()
            }
        torch.save(data, file_path)
