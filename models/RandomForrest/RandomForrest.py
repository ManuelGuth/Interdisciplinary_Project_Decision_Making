import ccobra
import numpy as np
import sys
sys.path.append('..')
from DataLoader import DataLoader
from sklearn.ensemble import RandomForestClassifier


class RandomForrest(ccobra.CCobraModel):
    """
    CCOBRA baseline model for the decision making task.
    """
    def __init__(self, name='RandomForrest'):
        """ Model constructor.

        """
        self.max_depth = 15
        self.n_trees = 142

        name = name + 'n_trees_{}_maxdepth_{}'.format(self.n_trees, self.max_depth)

        supported_domains = ['decision-making']
        supported_response_types = ['single-choice']
        # Call the super constructor to fully initialize the model
        super(RandomForrest, self).__init__(
            name, supported_domains, supported_response_types)

        self.rf_classifier = RandomForestClassifier(n_estimators=self.n_trees, max_depth=self.max_depth, oob_score=True)

        self.prev_answer = [0.0, 0.0]
        self.cnt = 0

    def start_participant(self, **kwargs):
        """ Model initialization method. Used to setup the initial state of
        its datastructures, memory, etc.

        """
        pass

    def pre_train(self, dataset):
        """ Pre-trains the model based on one or more datasets.

        """
        data = DataLoader(dataset, 1)
        data = data.data_loader
        X = []
        y = []
        for point in data:
            X.append(point[0].tolist()[0])
            y.append(np.argmax(point[1].tolist()))
        X = np.array(X)
        y = np.array(y)
        self.rf_classifier.fit(X, y)
        print(self.rf_classifier.oob_score_)

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
