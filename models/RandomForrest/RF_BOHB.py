import ccobra
import logging
logging.basicConfig(level=logging.INFO)

import ConfigSpace as CS
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB as BOHB
import numpy as np
import sys
sys.path.append('..')
from DataLoader import DataLoader
from sklearn.ensemble import RandomForestClassifier

X = None
y = None


class BOHB_RandomForrest(ccobra.CCobraModel):
    """
    CCOBRA baseline model for the decision making task.
    """
    def __init__(self, name='RandomForrest_BOHB'):
        """ Model constructor.

        """
        self.max_depth = None
        self.n_trees = None
        self.rf_classifier = None

        name = name

        supported_domains = ['decision-making']
        supported_response_types = ['single-choice']
        # Call the super constructor to fully initialize the model
        super(BOHB_RandomForrest, self).__init__(
            name, supported_domains, supported_response_types)

        self.prev_answer = [0.0, 0.0]
        self.cnt = 0

    def start_participant(self, **kwargs):
        """ Model initialization method. Used to setup the initial state of
        its datastructures, memory, etc.

        """
        print('Participant: ', kwargs)

    def pre_train(self, dataset):
        """ Pre-trains the model based on one or more datasets.

        """
        run_id = 'first_run'
        host = '127.0.0.1'
        n_workers = 4
        n_iterations = 40

        data = DataLoader(dataset, 1, cuda=False)
        data = data.data_loader
        np.random.shuffle(data)
        X_ = []
        y_ = []
        for point in data:
            X_.append(point[0].tolist()[0])
            y_.append(np.argmax(point[1].tolist()))
        global X
        global y
        X = np.array(X_)
        y = np.array(y_)

        # shuffle the data
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

        min_budget = len(X)*0.01
        max_budget = len(X)*0.25

        NS = hpns.NameServer(run_id=run_id, host=host, port=None)
        NS.start()

        workers = []
        print('starting workers:')
        for i in range(n_workers):
            print(i)
            w = RFWorker(nameserver=host, run_id=run_id, id=i)
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

        self.max_depth = id2config[incumbent]['config']['max_depth']
        self.n_trees = id2config[incumbent]['config']['n_trees']
        self.rf_classifier = RandomForestClassifier(n_estimators=self.n_trees, max_depth=self.max_depth)
        self.rf_classifier.fit(X, y)
        with open('config.txt', 'a') as f:
            print('Best config: n_trees: {}, max_depth: {}'.format(self.n_trees, self.max_depth), file=f)

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given problem.
        a task look like this:
        task: [[Ha, pHa, La, LotShapeA, NumLotA],[[Hb, pHb, Lb, LotShapeB, NumLotB]],[Amb, Corr]]
        """
        choice = ['A', 'B']
        data = DataLoader([item, kwargs, self.prev_answer], batch_size=1, eval=True)
        data = np.array([data.data_loader])
        prediction = self.rf_classifier.predict(data)
        return choice[int(prediction)]

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


class RFWorker(Worker):

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
        global X
        global y

        X_ = X[0:int(budget)]
        y_ = y[0:int(budget)]
        n_trees = config['n_trees']
        max_depth = config['max_depth']
        RF = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, oob_score=True)
        RF.fit(X_, y_)
        res = RF.oob_score_
        print('Config of n_trees: {}, max_depth: {}, OOB score: {}'.format(n_trees, max_depth, res))
        return({
                    'loss': 1 - res,  # this is the a mandatory field to run hyperband
                    'info': res  # can be used for any user-defined information - also mandatory
                })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_depth', lower=1, upper=22))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('n_trees', lower=40, upper=1000))
        return config_space
