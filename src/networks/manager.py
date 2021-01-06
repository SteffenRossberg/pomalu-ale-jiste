import torch
import os
import torch.optim as optim
from src.networks.encoder import Encoder, Decoder, AutoEncoder
from src.networks.classifier import Classifier
from src.networks.decision_maker import DecisionMaker


class NetManager:

    def __init__(self, device, data_directory='data/networks'):
        self.net_device = device
        self.data_directory = data_directory
        os.makedirs(f'{self.data_directory}', exist_ok=True)

    @property
    def device(self):
        return self.net_device

    def save_net(self, file_name, net, optimizer=None, loss=100.0):
        file_path = f'{self.data_directory}/{file_name}.pt'
        data = {
            'net': net.state_dict(),
            'loss': loss
        }
        if optimizer is not None:
            data['optimizer'] = optimizer.state_dict()
        torch.save(data, file_path)

    def load_net(self, file_name, net, optimizer=None, default_loss=100.0):
        file_path = f'{self.data_directory}/{file_name}.pt'
        loss = default_loss
        if os.path.exists(file_path):
            data = torch.load(file_path)
            net.load_state_dict(data['net'])
            net.eval()
            if optimizer is not None:
                optimizer.load_state_dict(data['optimizer'])
            loss = float(data['loss'])
        return loss

    def create_auto_encoder(self, days, net_name):
        encoder = Encoder(input_shape=(days, 4), output_size=days).to(self.device)
        decoder = Decoder(input_size=days, output_shape=(days, 4)).to(self.device)
        auto_encoder = AutoEncoder(encoder, decoder).to(self.device)
        optimizer = optim.Adam(auto_encoder.parameters())
        return auto_encoder, optimizer

    def create_classifier(self, buyer, seller, net_name):
        classifier = Classifier(buyer, seller).to(self.device)
        optimizer = optim.Adam(classifier.parameters())
        return classifier, optimizer

    def create_decision_maker(self, classifier, net_name, state_size=7):
        agent = DecisionMaker(classifier, state_size=state_size).to(self.device)
        optimizer = optim.Adam(agent.parameters(), lr=0.0001)
        return agent, optimizer
