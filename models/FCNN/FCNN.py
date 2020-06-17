import ccobra
import torch
import torch.nn as nn
import sys
sys.path.append('C:/Users/Manuel/Desktop/Interdisciplinary_Project_Decision_Making/models')
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
        self.load = 'runs/FCNN_ep1000_bs750_lr0.001/best_model.pth'
        self.lr = 0.001
        self.num_epochs = 100
        self.batch_size = 750
        name = name + '_ep{}_bs{}_lr{}'.format(self.num_epochs, self.batch_size, self.lr)
        supported_domains = ['decision-making']
        supported_response_types = ['single-choice']
        # Call the super constructor to fully initialize the model
        super(MyModel, self).__init__(
            name, supported_domains, supported_response_types)

        self.FCNN = nn.Sequential(nn.Linear(20, 200),
                                  nn.ReLU(),
                                  nn.Dropout(0.15),
                                  nn.Linear(200, 275),
                                  nn.ReLU(),
                                  nn.Dropout(0.15),
                                  nn.Linear(275, 100),
                                  nn.ReLU(),
                                  nn.Dropout(0.15),
                                  nn.Linear(100, 2),
                                  nn.Sigmoid()).cuda()

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
            data_loader = DataLoader(dataset, self.batch_size)
            data_loader = data_loader.data_loader
            train_size = int(0.9 * len(data_loader))
            val_size = len(data_loader) - train_size
            train_data, val_data = torch.utils.data.random_split(data_loader, [train_size, val_size])

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.RMSprop(self.FCNN.parameters(), lr=self.lr)
            print('Starting training on {} batches of train data with {} batches of validation data.'.format(train_size,
                                                                                                             val_size))
            best_loss = np.inf
            best_model = None
            for epoch in range(self.num_epochs):
                mean_train_loss = 0
                mean_train_acc = 0
                self.FCNN.train()
                for data, label in train_data:
                    optimizer.zero_grad()
                    predictions = self.FCNN(data)
                    loss = criterion(predictions, torch.argmax(label, dim=1).long())
                    eq = np.equal(torch.argmax(predictions, dim=1).cpu(), torch.argmax(label, dim=1).cpu())
                    acc = torch.mean(eq.float())
                    mean_train_acc += acc
                    mean_train_loss += loss
                    loss.backward()
                    optimizer.step()
                mean_train_loss = mean_train_loss / train_size
                mean_train_acc = mean_train_acc / train_size
                mean_val_loss = 0
                mean_val_acc = 0
                self.FCNN.eval()
                for data, label in val_data:
                    with torch.no_grad():
                        predictions = self.FCNN(data)
                        loss = criterion(predictions, torch.argmax(label, dim=1).long())
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
            self.load_model()
        self.FCNN.eval()

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given problem.
        a task look like this:
        task: [[Ha, pHa, La, LotShapeA, NumLotA],[[Hb, pHb, Lb, LotShapeB, NumLotB]],[Amb, Corr]]
        """
        choice = ['A', 'B']
        data = DataLoader([item, kwargs], batch_size=1, eval=True)
        data = data.data_loader
        with torch.no_grad():
            predictions = self.FCNN(torch.Tensor(data).cuda())
            answer = torch.argmax(predictions)
            return choice[answer]

    def adapt(self, item, target, **kwargs):
        """ Trains the model based on a given problem-target combination.

        """
        # retrain model on person
        #self.FCNN.train()
        #criterion = nn.CrossEntropyLoss()
        #optimizer = torch.optim.RMSprop(self.FCNN.parameters(), lr=self.lr)
        #optimizer.zero_grad()

        #data = DataLoader([item, kwargs], batch_size=1, eval=True)
        #data = data.data_loader
        #predictions = self.FCNN(torch.Tensor(data).cuda())
        #if target[0][0] == 'A':
        #    label = torch.Tensor([0]).float()
        #else:
        #    label = torch.Tensor([1]).float()
        #loss = criterion(predictions, label)
        #loss.backward()
        #optimizer.step()
        #self.FCNN.eval()

    def load_model(self):
        state_dict = torch.load(self.load)
        self.FCNN.load_state_dict(state_dict)


