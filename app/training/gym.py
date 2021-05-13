import torch
import numpy as np
from torch import nn as nn
from prometheus_client import Gauge
from app.networks.manager import NetManager
from app.networks.models import TrainClassifier, AutoEncoder, AutoEncoder2, Classifier2


class Gym:

    def __init__(self, manager: NetManager):
        self.manager = manager
        self.device = self.manager.device
        self.gauges = {}

    def get_gauge(self, name, description):
        if name not in self.gauges:
            self.gauges[name] = Gauge(
                name,
                description)
        return self.gauges[name]

    @classmethod
    def calculate_accuracies(cls, actual, expected):
        expected_range_x = expected.max(axis=(2, 3)) - expected.min(axis=(2, 3))
        ratio = ((actual - expected) / expected_range_x.reshape((expected_range_x.shape[0], 1, 1, 1)))
        ratio = np.abs(ratio)
        mean_ratio = np.mean(ratio, axis=(2, 3))
        mean_ratio = 1.0 - mean_ratio
        accuracies = np.where((mean_ratio < 0.0) | (mean_ratio > 1.0), 0.0, mean_ratio)
        accuracies = accuracies.flatten()
        return accuracies

    def train_auto_encoder2(
            self,
            name,
            auto_encoder: AutoEncoder2,
            features,
            max_epochs=200,
            batch_size=5000):

        def save(manager: NetManager, loss):
            manager.save_net(name, auto_encoder)
            manager.save_optimizer(name, auto_encoder.optimizer, loss)

        def output(e, accuracy, is_saved):
            Gym.print_step(e, 0, f'{name}.auto.encoder2', 0.0, accuracy, is_saved)

        train_features, train_labels, val_features, val_labels = self.split_train_and_val(features, features, 0.75)

        for epoch in range(max_epochs):
            # train
            train_features, train_labels = self.shuffle(train_features, train_labels)
            auto_encoder.train()
            for index in range(0, len(train_features), batch_size):
                count = batch_size if index + batch_size < len(train_features) else len(train_features) - index
                if not count > 0:
                    break
                features = train_features[index:index + count]
                labels = train_labels[index:index + count]
                auto_encoder.train_net(features, labels)
            # evaluate
            val_features, val_labels = self.shuffle(val_features, val_labels)
            auto_encoder.eval()
            accuracies = []
            for index in range(0, len(val_features), batch_size):
                count = batch_size if index + batch_size < len(val_features) else len(val_features) - index
                if not count > 0:
                    break
                features = val_features[index:index + count]
                labels = val_labels[index:index + count].detach().cpu().numpy()
                prediction = auto_encoder(features).detach().cpu().numpy()
                accuracies += self.calculate_accuracies(prediction, labels).flatten().tolist()
            mean_accuracy = np.mean(accuracies) * 100.0
            output(epoch, mean_accuracy, False)

        save(self.manager, 1.0)

    def train_classifier2(
            self,
            name,
            classifier: Classifier2,
            auto_encoder: AutoEncoder2,
            features,
            labels,
            max_epochs=200,
            batch_size=5000):

        def save(manager: NetManager, loss):
            manager.save_net(name, classifier)
            manager.save_optimizer(name, classifier.optimizer, loss)

        def output(e, accuracy, is_saved):
            Gym.print_step(e, 0, f'{name}.classifier2', 0.0, accuracy, is_saved)

        def calculate_accuracies(actual, expected):
            actual = np.argmax(actual, axis=-1)
            return np.sum([1 if actual[i] == expected[i] else 0 for i in range(len(actual))]) / len(actual)

        train_features, train_labels, val_features, val_labels = self.split_train_and_val(features, labels, 0.75)

        for epoch in range(max_epochs):
            # train
            train_features, train_labels = self.shuffle(train_features, train_labels)
            classifier.train()
            for index in range(0, len(train_features), batch_size):
                count = batch_size if index + batch_size < len(train_features) else len(train_features) - index
                if not count > 0:
                    break
                features = train_features[index:index + count]
                labels = train_labels[index:index + count]
                classifier.train_net(auto_encoder, features, labels)
            # evaluate
            val_features, val_labels = self.shuffle(val_features, val_labels)
            classifier.eval()
            accuracies = []
            for index in range(0, len(val_features), batch_size):
                count = batch_size if index + batch_size < len(val_features) else len(val_features) - index
                if not count > 0:
                    break
                features = val_features[index:index + count]
                decoded = auto_encoder(features)
                features = torch.cat([features, decoded], dim=1)
                prediction = classifier(features.detach()).detach().cpu().numpy()
                labels = val_labels[index:index + count].detach().cpu().numpy()
                accuracies += calculate_accuracies(prediction, labels).flatten().tolist()
            mean_accuracy = np.mean(accuracies) * 100.0
            output(epoch, mean_accuracy, False)

        save(self.manager, 1.0)

    @classmethod
    def shuffle(cls, features, labels):
        random = torch.randperm(len(features))
        features = features[random]
        labels = labels[random]
        return features, labels

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
            calculate_accuracies=self.calculate_accuracies)

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
        self.manager.init_seed(self.manager.seed, self.manager.deterministic)
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

    def split_train_and_val(self, features, labels, rate=0.75):
        features = torch.tensor(features).to(self.device)
        labels = torch.tensor(labels).to(self.device)
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
        features, labels = self.shuffle(train_features, train_labels)
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
        features = val_features
        labels = val_labels
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
