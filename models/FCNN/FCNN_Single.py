import ccobra
import torch
import torch.nn as nn
import sys
#sys.path.append('C:/Users/Manuel/Desktop/Interdisciplinary_Project_Decision_Making/models')
sys.path.append('..')

from DataLoader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import numpy as np


class MyModel(ccobra.CCobraModel):
    """
    CCOBRA baseline model for the decision making task.
    """
    def __init__(self, name='FCNN'):
        """ Model constructor.

        """
        # set to path if you would like to load a model instead of training. Else False.
        # self.load = 'runs/FCNN_ep1000_bs750_lr0.001/best_model.pth'
        self.load = False
        self.lr = 0.000129
        self.num_epochs = 100
        self.batch_size = 20
        name = name + '_ep{}_bs{}_lr{}'.format(self.num_epochs, self.batch_size, self.lr)
        supported_domains = ['decision-making']
        supported_response_types = ['single-choice']
        # Call the super constructor to fully initialize the model
        super(MyModel, self).__init__(
            name, supported_domains, supported_response_types)

        self.FCNN = nn.Sequential(nn.Linear(22, 310),
                                  nn.ReLU(),
                                  nn.Dropout(0.10),
                                  nn.Linear(310, 420),
                                  nn.ReLU(),
                                  nn.Dropout(0.10),
                                  nn.Linear(420, 800),
                                  nn.ReLU(),
                                  nn.Dropout(0.10),
                                  nn.Linear(800, 400),
                                  nn.ReLU(),
                                  nn.Dropout(0.10),
                                  nn.Linear(400, 170),
                                  nn.ReLU(),
                                  nn.Dropout(0.10),
                                  nn.Linear(170, 2)).cuda()

        self.prev_answer = [0.0, 0.0]
        self.cnt = 0

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.RMSprop(self.FCNN.parameters(), lr=self.lr)

    def start_participant(self, **kwargs):
        """ Model initialization method. Used to setup the initial state of
        its datastructures, memory, etc.

        """
        pass

    def pre_train(self, dataset):
        """ Pre-trains the model based on one or more datasets.

        """
        if not self.load:
            writer = SummaryWriter('runs/' + self.name)
            data_loader = DataLoader(dataset, self.batch_size, single=True)
            data_loader = data_loader.data_loader
            train_size = int(0.9 * len(data_loader))
            val_size = len(data_loader) - train_size
            train_data, val_data = torch.utils.data.random_split(data_loader, [train_size, val_size])

            print('Starting training on {} batches of train data with {} batches of validation data.'.format(train_size,
                                                                                                             val_size))
            best_loss = np.inf
            best_model = None
            for epoch in range(self.num_epochs):
                mean_train_loss = 0
                mean_train_acc = 0
                self.FCNN.train()
                for data, label in train_data:
                    self.optimizer.zero_grad()
                    predictions = self.FCNN(data)
                    loss = self.criterion(predictions, torch.argmax(label, dim=1).long())
                    eq = np.equal(torch.argmax(predictions, dim=1).cpu(), torch.argmax(label, dim=1).cpu())
                    acc = torch.mean(eq.float())
                    mean_train_acc += acc
                    mean_train_loss += loss
                    loss.backward()
                    self.optimizer.step()
                mean_train_loss = mean_train_loss / train_size
                mean_train_acc = mean_train_acc / train_size
                mean_val_loss = 0
                mean_val_acc = 0
                self.FCNN.eval()
                for data, label in val_data:
                    with torch.no_grad():
                        predictions = self.FCNN(data)
                        loss = self.criterion(predictions, torch.argmax(label, dim=1).long())
                        eq = np.equal(torch.argmax(predictions, dim=1).cpu(), torch.argmax(label, dim=1).cpu())
                        acc = torch.mean(eq.float())
                        mean_val_loss += loss
                        mean_val_acc += acc
                mean_val_loss = mean_val_loss / val_size
                mean_val_acc = mean_val_acc / val_size

                if mean_val_loss < best_loss:
                    best_model = deepcopy(self.FCNN)
                    torch.save(self.FCNN.state_dict(), 'runs/' + self.name + '/best_model.pth')
                    best_loss = mean_val_loss
                writer.add_scalar('Loss/val', mean_val_loss, epoch)
                writer.add_scalar('Loss/train', mean_train_loss, epoch)
                writer.add_scalar('Acc/train', mean_train_acc, epoch)
                writer.add_scalar('Acc/val', mean_val_acc, epoch)
                print('Epoch {}/{} train loss: {}, train acc: {}, val loss: {}, val acc: {}'.format(epoch,
                      self.num_epochs-1,mean_train_loss, mean_train_acc, mean_val_loss, mean_val_acc))
            self.FCNN = best_model
        else:
            print('Loading model settings from {}'.format(self.load))
            self.load_model()

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given problem.
        a task look like this:
        task: [[Ha, pHa, La, LotShapeA, NumLotA],[[Hb, pHb, Lb, LotShapeB, NumLotB]],[Amb, Corr]]
        """
        self.FCNN.eval()
        choice = ['A', 'B']
        data = DataLoader([item, kwargs, self.prev_answer], batch_size=1, eval=True, single=True)
        data = data.data_loader
        with torch.no_grad():
            predictions = self.FCNN(torch.Tensor(data).cuda())
            answer = torch.argmax(predictions)
            return choice[answer]

    def adapt(self, item, target, **kwargs):
        """ Trains the model based on a given problem-target combination.

        """
        if target[0][0] == 'A':
            self.prev_answer = [1.0, 0.0]
        else:
            self.prev_answer = [0.0, 1.0]
        if self.cnt % 25 == 0 and self.cnt > 0:
            self.prev_answer = [0.0, 0.0]
        self.cnt += 1

    def load_model(self):
        state_dict = torch.load(self.load)
        self.FCNN.load_state_dict(state_dict)
