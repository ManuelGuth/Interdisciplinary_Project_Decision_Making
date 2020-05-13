import ccobra


class MyModel(ccobra.CCobraModel):
    """
    CCOBRA baseline model for the decision making task.
    The model predicts the gamble with the highest expected value to be chosen by the participant.
    """
    def __init__(self, name='MaxExpValue'):
        """ Model constructor.

        """

        # Call the super constructor to fully initialize the model
        supported_domains = ['decision-making']
        supported_response_types = ['single-choice']
        super(MyModel, self).__init__(
            name, supported_domains, supported_response_types)

    def start_participant(self, **kwargs):
        """ Model initialization method. Used to setup the initial state of
        its datastructures, memory, etc.

        """
        pass

    def pre_train(self, dataset):
        """ Pre-trains the model based on one or more datasets.

        """
        pass

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given problem.
        a task look like this:
        task: [[Ha, pHa, La, LotShapeA, NumLotA],[[Hb, pHb, Lb, LotShapeB, NumLotB]],[Amb, Corr]]
        """
        task = item.task
        # calculate ExpValue for A
        ha, pHa, la, _, _ = task[0]
        expVA = float(ha) * float(pHa) + (1-float(pHa))*float(la)

        # calculate ExpValue for B
        hb, pHb, lb, _, _ = task[1]
        expVB = float(hb) * float(pHb) + (1-float(pHb))*float(lb)

        amb, _ = task[2]

        # case where the participant did not know the probabilities for the B option:
        if amb == '1':
            if expVA > int(hb)*0.5 + int(lb)*0.5:
                return 'A'
            else:
                return 'B'

        if expVA > expVB:
            return 'A'
        else:
            return 'B'

    def adapt(self, item, target, **kwargs):
        """ Trains the model based on a given problem-target combination.

        """
        pass
