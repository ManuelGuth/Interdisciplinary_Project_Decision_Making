import ccobra
import numpy as np


def getDifferentAnswer(last_answer):
    if last_answer == 'A':
        return 'B'
    return 'A'


class MyModel(ccobra.CCobraModel):
    """
    CCOBRA baseline model for the decision making task.
    The model predicts the answer the participant gave previously for the same task.
    """
    def __init__(self, name='SimulatedAnnealing'):
        """ Model constructor.

        """
        # Call the super constructor to fully initialize the model
        supported_domains = ['decision-making']
        supported_response_types = ['single-choice']
        super(MyModel, self).__init__(
            name, supported_domains, supported_response_types)

        self.common_answer = None
        self.last_answer = None
        self.p = 1
        self.turnCounter = 0
        self.change_probs = []

    def start_participant(self, **kwargs):
        """ Model initialization method. Used to setup the initial state of
        its datastructures, memory, etc.

        """
        self.last_answer = None
        self.p = 1

    def pre_train(self, dataset):
        """ Pre-trains the model based on one or more datasets.

        """
        cnt = {'A': 0, 'B': 0}
        when_changed = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # count what is the most common answer.
        cnter = 0
        person_cnt = 0
        changed = 0
        for person in dataset:
            person_changed = False
            for item in person:
                response = item['response'][0][0]
                cnt[response] += 1
                if self.last_answer != response:
                    if cnter % 25 != 0:
                        person_changed = True
                        changed += 1
                        when_changed[cnter % 25] += 1
                self.last_answer = response
                cnter += 1
            if person_changed:
                person_cnt += 1
        if cnt['A'] > cnt['B']:
            self.common_answer = 'A'
        else:
            self.common_answer = 'B'
        for i in range(len(when_changed)):
            when_changed[i] = when_changed[i]/cnter
        self.change_probs = when_changed
        print(changed, cnt, cnter, changed/cnter, when_changed, person_cnt, cnter/750)

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given problem.
        a task look like this:
        task: [[Ha, pHa, La, LotShapeA, NumLotA],[[Hb, pHb, Lb, LotShapeB, NumLotB]],[Amb, Corr]]
        """
        if self.last_answer is not None:
            p = np.random.rand()
            if p > self.change_probs[self.turnCounter]:
                answer = self.last_answer
            else:
                answer = getDifferentAnswer(self.last_answer)
        else:
            answer = self.common_answer

        self.turnCounter += 1
        if self.turnCounter >= 25:
            self.turnCounter = 0
        return answer

    def adapt(self, item, target, **kwargs):
        """ Trains the model based on a given problem-target combination.

        """
        self.last_answer = target
