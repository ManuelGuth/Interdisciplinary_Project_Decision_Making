import ccobra
import numpy as np
from CPC18BEAST.BEASTsd_pred import CPC18_BEASTsd_pred as BEAST_PRED


class BEAST(ccobra.CCobraModel):
    """
    CPC18 BEASTmd model implementation adapted to the needs of the CCOBRA task.
    """
    def __init__(self, name='BEAST'):
        """ Model constructor.

        """
        # Call the super constructor to fully initialize the model
        supported_domains = ['decision-making']
        supported_response_types = ['single-choice']
        super(BEAST, self).__init__(
            name, supported_domains, supported_response_types)
        self.trail_cnt = 0
        self.Prediction = None

    def start_participant(self, **kwargs):
        """ Model initialization method. Used to setup the initial state of
        its datastructures, memory, etc.

        """
        print('evaluating participant', kwargs)

    def pre_train(self, dataset):
        """ Pre-trains the model based on one or more datasets.

        """
        pass

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given problem.
        a task look like this:
        task: [[Ha, pHa, La, LotShapeA, NumLotA],[[Hb, pHb, Lb, LotShapeB, NumLotB]],[Amb, Corr]]
        """
        self.trail_cnt += 1
        if self.trail_cnt == 1:
            Ha, pHa, La, LotShapeA, LotNumA = item.task[0]
            Hb, pHb, Lb, LotShapeB, LotNumB = item.task[1]
            Amb, Corr = item.task[2]
            self.Prediction = BEAST_PRED(float(Ha), float(pHa), float(La), LotShapeA, int(LotNumA), float(Hb), float(pHb),
                                         float(Lb), LotShapeB, int(LotNumB), Amb, Corr)
        if self.trail_cnt <= 5:
            prediction = self.Prediction[0][0]
        elif 5 < self.trail_cnt <= 10:
            prediction = self.Prediction[0][1]
        elif 10 < self.trail_cnt <= 15:
            prediction = self.Prediction[0][2]
        elif 15 < self.trail_cnt <= 20:
            prediction = self.Prediction[0][3]
        elif 20 < self.trail_cnt <= 25:
            prediction = self.Prediction[0][4]

        if self.trail_cnt >= 25:
            self.trail_cnt = 0

        if prediction > 0.5:
            return 'B'
        else:
            return 'A'

    def adapt(self, item, target, **kwargs):
        """ Trains the model based on a given problem-target combination.

        """
        pass
