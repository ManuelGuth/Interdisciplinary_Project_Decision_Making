import ccobra


class MyModel(ccobra.CCobraModel):
    """
    CCOBRA baseline model for the decision making task.
    The model predicts the answer the participant gave previously for the same task.
    """
    def __init__(self, name='lastGivenAnswer'):
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
        for data in dataset:
            print(data)
            return

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given problem.
        a task look like this:
        task: [[Ha, pHa, La, LotShapeA, NumLotA],[[Hb, pHb, Lb, LotShapeB, NumLotB]],[Amb, Corr]]
        """
        pass

    def adapt(self, item, target, **kwargs):
        """ Trains the model based on a given problem-target combination.

        """
        pass
