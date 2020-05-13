import ccobra

class MyModel(ccobra.CCobraModel):
    def __init__(self, name='MyModel'):
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
        print("Start Participant")
        print("kwargs: ", kwargs)

    def pre_train(self, dataset):
        """ Pre-trains the model based on one or more datasets.

        """
        print("Pre Training")
        print(dataset)

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given problem.

        """
        print("Predict")
        print(item)
        print("kwargs: ", kwargs)
        return "A"

    def adapt(self, item, target, **kwargs):
        """ Trains the model based on a given problem-target combination.

        """
        print("Adapt")
        print(target)
        print("kwargs: ", kwargs)
