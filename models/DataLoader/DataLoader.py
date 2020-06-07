import torch


class DataLoader:
    """
    encoding:
    one hot encoding for:
    LotShapeA/B: Symm, R-skew, L-skew
    [Ha, pHa, La, LotShapeA, NumLotA],[[Hb, pHb, Lb, LotShapeB, NumLotB]],[Amb, Corr]]
    """
    def __init__(self, data, batch_size=25):
        self.data = data
        self.batch_size = batch_size
        self.transform_data()

    def transform_data(self):
        if self.batch_size == 1:
            # TODO implement data loader for prediction case
            raise NotImplementedError


        data = []
        # pre training case
        for person in self.data:
            for task in person:
                item = task['item']
                response = task['response']
                feedback = task['aux']['prev_feedback']
                Ha, pHa, La, LotShapeA, LotNumA = item.task[0]
                Hb, pHb, Lb, LotShapeB, LotNumB = item.task[1]
                Amb, Corr = item.task[2]
                payoff, forgone = feedback
                array = [int]
                print(item)
                print(response)
                print(feedback)
                raise NotImplementedError
