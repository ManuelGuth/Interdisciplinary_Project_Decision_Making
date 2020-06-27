import torch
import ccobra
import sys
sys.path.append('..')
from DataLoader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import numpy as np

class LSTM(ccobra.CCobraModel):
    """
    CCOBRA baseline model for the decision making task.
    """
    def __init__(self, name='LSTM'):
        """ Model constructor.

        """
        # set to path if you would like to load a model instead of training. Else False.
        self.load = False
        self.lr = 0.01
        self.num_epochs = 100
        self.batch_size = 5
        self.seq_length = 750
        name = name + '_ep{}_bs{}_sq{}_lr{}'.format(self.num_epochs, self.batch_size, self.seq_length, self.lr)

        supported_domains = ['decision-making']
        supported_response_types = ['single-choice']
        # Call the super constructor to fully initialize the model
        super(LSTM, self).__init__(
            name, supported_domains, supported_response_types)

        self.LSTM = torch.nn.LSTM(22, 2).cuda()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.LSTM.parameters(), lr=self.lr)

        self.prev_answer = [0.0, 0.0]
        self.cnt = 0
        self.c0 = None
        self.h0 = None

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
            data = DataLoader(dataset, 1)
            data = data.data_loader
            new_data = []
            for i in range(0, len(data), self.seq_length):
                _data = []
                _label = []
                for each in data[i:i+self.seq_length]:
                    _data.append(each[0].tolist()[0])
                    _label.append(each[1].tolist()[0])
                new_data.append([torch.Tensor(_data).cuda(), torch.Tensor(_label).cuda()])
            data = new_data

            train_size = int(0.9 * len(data))
            val_size = len(data) - train_size
            train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])
            print('Starting training on {} batches of train data with {} batches of validation data.'.format(train_size,
                                                                                                            val_size))
            best_loss = np.inf
            best_model = None
            for epoch in range(self.num_epochs):
                mean_train_loss = 0
                mean_train_acc = 0
                self.LSTM.train()
                for data, label in train_data:
                    self.optimizer.zero_grad()
                    data = torch.unsqueeze(data, dim=1)
                    predictions = torch.squeeze(self.LSTM(data)[0])
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
                self.LSTM.eval()
                for data, label in val_data:
                    with torch.no_grad():
                        data = torch.unsqueeze(data, dim=1)
                        predictions = torch.squeeze(self.LSTM(data)[0])
                        loss = self.criterion(predictions, torch.argmax(label, dim=1).long())
                        eq = np.equal(torch.argmax(predictions, dim=1).cpu(), torch.argmax(label, dim=1).cpu())
                        acc = torch.mean(eq.float())
                        mean_val_loss += loss
                        mean_val_acc += acc
                mean_val_loss = mean_val_loss / val_size
                mean_val_acc = mean_val_acc / val_size

                if mean_val_loss < best_loss:
                    best_model = deepcopy(self.LSTM)
                    torch.save(self.LSTM.state_dict(), 'runs/' + self.name + '/best_model.pth')
                    best_loss = mean_val_loss
                    print('saving at: ', 'runs/' + self.name + '/best_model.pth')
                writer.add_scalar('Loss/val', mean_val_loss, epoch)
                writer.add_scalar('Loss/train', mean_train_loss, epoch)
                writer.add_scalar('Acc/train', mean_train_acc, epoch)
                writer.add_scalar('Acc/val', mean_val_acc, epoch)
                print('Epoch {}/{} train loss: {}, train acc: {}, val loss: {}, val acc: {}'.format(epoch,
                      self.num_epochs-1, mean_train_loss, mean_train_acc, mean_val_loss, mean_val_acc))
            self.LSTM = best_model
        else:
            print('Loading model settings from {}'.format(self.load))
            self.load_model()

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given problem.
        a task look like this:
        task: [[Ha, pHa, La, LotShapeA, NumLotA],[[Hb, pHb, Lb, LotShapeB, NumLotB]],[Amb, Corr]]
        """
        self.LSTM.eval()
        choice = ['A', 'B']
        data = DataLoader([item, kwargs, self.prev_answer], batch_size=1, eval=True)
        data = torch.unsqueeze(torch.Tensor(data.data_loader).cuda(), dim=0)
        data = torch.unsqueeze(data, dim=0)
        with torch.no_grad():
            self.LSTM.flatten_parameters()
            if self.h0 is not None:
                prediction, (self.c0, self.h0) = self.LSTM(data, (self.h0, self.c0))
            else:
                prediction, (self.c0, self.h0) = self.LSTM(data)
            answer = int(torch.argmax(prediction))
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
        self.LSTM.load_state_dict(state_dict)
