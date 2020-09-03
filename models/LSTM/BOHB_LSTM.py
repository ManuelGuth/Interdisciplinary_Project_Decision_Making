import ccobra
import torch
import torch.nn as nn
import sys
#sys.path.append('C:/Users/Manuel/Desktop/Interdisciplinary_Project_Decision_Making/models')
sys.path.append('..')
from DataLoader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import ConfigSpace as CS
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB as BOHB

validation_data = None
training_data = None



class BOHB_LSTM(ccobra.CCobraModel):
    """
    CCOBRA baseline model for the decision making task.
    """
    def __init__(self, name='BOHB_LSTM'):
        """ Model constructor.
        """
        # set to path if you would like to load a model instead of training. Else False.
        # self.load = 'runs/FCNN_ep1000_bs750_lr0.001/best_model.pth'
        supported_domains = ['decision-making']
        supported_response_types = ['single-choice']
        # Call the super constructor to fully initialize the model
        super(BOHB_LSTM, self).__init__(
            name, supported_domains, supported_response_types)

        self.prev_answer = [0.0, 0.0]
        self.cnt = 0
        self.seq_len = 750  # 750 for full 30 for single
        self.single = False    # True for single false in other cases

    def start_participant(self, **kwargs):
        """ Model initialization method. Used to setup the initial state of
        its datastructures, memory, etc.

        """
        pass

    def pre_train(self, dataset):
        """ Pre-trains the model based on one or more datasets.
        ccobra FCNN_BOHB.json
        """
        run_id = 'first_run'
        host = '127.0.0.1'
        n_workers = 4
        n_iterations = 100

        data_loader = DataLoader(dataset, batch_size=1, single=self.single)
        data = data_loader.data_loader
        new_data = []
        for i in range(0, len(data), self.seq_len):
            _data = []
            _label = []
            for each in data[i:i+self.seq_len]:
                _data.append(each[0].tolist()[0])
                _label.append(each[1].tolist()[0])
            new_data.append([torch.Tensor(_data).cuda(), torch.Tensor(_label).cuda()])
        data = new_data

        train_size = int(0.9 * len(data))
        val_size = len(data) - train_size
        train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])
        global training_data
        global validation_data
        training_data = train_data
        validation_data = val_data

        min_budget = 2
        max_budget = 15

        NS = hpns.NameServer(run_id=run_id, host=host, port=None)
        NS.start()

        workers = []
        print('starting workers:')
        for i in range(n_workers):
            print(i)
            w = FCNNWorker(nameserver=host, run_id=run_id, id=i)
            w.run(background=True)
            workers.append(w)

        bohb = BOHB(configspace=workers[0].get_configspace(), run_id=run_id,
                    min_budget=min_budget, max_budget=max_budget)
        print('running BOHB')
        res = bohb.run(n_iterations=n_iterations, min_n_workers=n_workers)
        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()

        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()

        all_runs = res.get_all_runs()

        print('Best found configuration:', id2config[incumbent]['config'])
        print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
        print('A total of %i runs where executed.' % len(res.get_all_runs()))
        print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/max_budget))
        print('The run took  %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))

        with open('config.txt', 'a') as f:
            print(id2config[incumbent]['config'], file=f)

        self.LSTM = nn.LSTM(22, 2, dropout=id2config[incumbent]['config']['dropout'], num_layers=id2config[incumbent]['config']['layers']).cuda()
        lr = id2config[incumbent]['config']['lr']
        train_model(self.LSTM, lr, max_budget)

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given problem.
        a task look like this:
        task: [[Ha, pHa, La, LotShapeA, NumLotA],[[Hb, pHb, Lb, LotShapeB, NumLotB]],[Amb, Corr]]
        """
        self.LSTM.eval()
        choice = ['A', 'B']
        data = DataLoader([item, kwargs, self.prev_answer], batch_size=1, eval=True, single=self.single)
        data = data.data_loader
        with torch.no_grad():
            predictions = self.LSTM(torch.Tensor(data).cuda())
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


def train_model(model, lr, num_epochs):
    global training_data
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    best_loss = np.inf
    model.train()

    for epoch in range(num_epochs):
        print("Epoch{}".format(epoch))
        mean_train_loss = 0
        mean_train_acc = 0
        for data, label in training_data:
            optimizer.zero_grad()
            data = torch.unsqueeze(data, dim=1)
            predictions = torch.squeeze(model(data)[0])
            loss = criterion(predictions, torch.argmax(label, dim=1).long())
            eq = np.equal(torch.argmax(predictions, dim=1).cpu(), torch.argmax(label, dim=1).cpu())
            acc = torch.mean(eq.float())
            mean_train_acc += acc
            mean_train_loss += loss
            loss.backward()
            optimizer.step()
        mean_train_loss = mean_train_loss / len(training_data)
        if mean_train_loss < best_loss:
            best_loss = mean_train_loss

    return float(best_loss)


def eval_model(model):
    global validation_data
    criterion = nn.CrossEntropyLoss()

    mean_val_loss = 0
    mean_val_acc = 0
    model.eval()
    with torch.no_grad():
        for data, label in validation_data:
            data = torch.unsqueeze(data, dim=1)
            predictions = torch.squeeze(model(data)[0])
            loss = criterion(predictions, torch.argmax(label, dim=1).long())
            eq = np.equal(torch.argmax(predictions, dim=1).cpu(), torch.argmax(label, dim=1).cpu())
            acc = torch.mean(eq.float())
            mean_val_loss += loss
            mean_val_acc += acc
        mean_val_loss = mean_val_loss / len(validation_data)
        mean_val_acc = mean_val_acc / len(validation_data)
    return float(mean_val_loss), float(mean_val_acc)


class FCNNWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, config, budget, **kwargs):
        """
        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        model = nn.LSTM(22, 2, dropout=config['dropout'], num_layers=config['layers']).cuda()
        lr = config['lr']
        train_loss = train_model(model, lr, int(budget))
        eval_loss, eval_acc = eval_model(model)
        print("Eval Loss: {}, Eval Acc: {}, Train Loss: {}".format(eval_loss, eval_acc, train_loss))
        return({
                    'loss': eval_loss,  # this is the a mandatory field to run hyperband
                    'info': eval_acc #{'Eval Loss:': eval_loss, 'Eval Acc:': eval_acc, 'Train Loss:': train_loss}
                })

    @staticmethod
    def get_configspace():

        max_layers = 10
        config_space = CS.ConfigurationSpace()
        lr = CS.UniformFloatHyperparameter("lr", lower=1e-6, upper=1e-1, log=True)
        num_layers = CS.UniformIntegerHyperparameter('layers', lower=1, upper=max_layers)
        dropout = CS.UniformFloatHyperparameter('dropout', lower=0, upper=1)

        config_space.add_hyperparameters([lr, num_layers, dropout])

        return config_space
