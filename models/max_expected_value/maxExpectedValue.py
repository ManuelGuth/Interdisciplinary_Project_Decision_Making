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

        """
        print(item)

    def adapt(self, item, target, **kwargs):
        """ Trains the model based on a given problem-target combination.

        """
        pass
