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

    def start_participant(self, **kwargs):
        """ Model initialization method. Used to setup the initial state of
        its datastructures, memory, etc.

        """
        print('Start evaluating participant:', kwargs)

    def pre_train(self, dataset):
        """ Pre-trains the model based on one or more datasets.

        """
        pass

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given problem.
        a task look like this:
        task: [[Ha, pHa, La, LotShapeA, NumLotA],[[Hb, pHb, Lb, LotShapeB, NumLotB]],[Amb, Corr]]
        """
        Ha, pHa, La, LotShapeA, LotNumA = item.task[0]
        Hb, pHb, Lb, LotShapeB, LotNumB = item.task[1]
        Amb, Corr = item.task[2]
        Prediction = BEAST_PRED(float(Ha), float(pHa), float(La), LotShapeA, int(LotNumA), float(Hb), float(pHb),
                                float(Lb), LotShapeB, int(LotNumB), Amb, Corr)
        mean_B_choice = np.mean(Prediction)
        if mean_B_choice > 0.5:
            return 'B'
        return 'A'

    def adapt(self, item, target, **kwargs):
        """ Trains the model based on a given problem-target combination.

        """
        pass
