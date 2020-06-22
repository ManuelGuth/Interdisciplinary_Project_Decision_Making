import torch


class DataLoader:
    """
    encoding:
    one hot encoding for:
    LotShapeA/B: -, Symm, R-skew, L-skew
    Previous answer: prev_answer_A, prev_answer_B
    [Ha, pHa, La, No_ShapeA, S_ShapeA, R_ShapeA, L_ShapeA, NumLotA,
    Hb, pHb, Lb, No_ShapeB, S_ShapeB, R_ShapeB, L_ShapeB, NumLotB,
    Amb, Corr, payoff, forgone, prev_answer_A, prev_answer_B]
    """

    def __init__(self, data, batch_size=25, eval=False, cuda=True):
        self.eval = eval
        self.cuda = cuda
        self.data = data
        self.data_loader = []
        self.batch_size = batch_size
        self.transform_data()

    def transform_data(self):
        if self.eval:
            item = self.data[0]
            feedback = self.data[1]
            prev_anwer = self.data[2]
            Ha, pHa, La, LotShapeA, LotNumA = item.task[0]
            Hb, pHb, Lb, LotShapeB, LotNumB = item.task[1]
            Amb, Corr = item.task[2]

            payoff = feedback['payoff']
            forgone = feedback['forgone']
            self.data_loader = self.get_array(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB,
                                                   Amb, Corr, payoff, forgone, prev_anwer)
        else:
            data = []
            label = []
            # pre training case
            for person in self.data:
                prev_answer = [0.0, 0.0]
                for i, task in enumerate(person):
                    item = task['item']
                    response = task['response'][0][0]
                    Ha, pHa, La, LotShapeA, LotNumA = item.task[0]
                    Hb, pHb, Lb, LotShapeB, LotNumB = item.task[1]
                    Amb, Corr = item.task[2]
                    payoff = task['aux']['payoff']
                    forgone = task['aux']['forgone']
                    array_form = self.get_array(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB,
                                                Amb, Corr, payoff, forgone, prev_answer)
                    data.append(array_form)
                    if response == 'B':
                        response = [0.0, 1.0]
                    else:
                        response = [1.0, 0.0]
                    if i % 25 == 0 and i > 0:
                        prev_answer = [0.0, 0.0]
                    else:
                        prev_answer = response
                    label.append(response)

            for i in range(0, len(label), self.batch_size):
                if self.cuda:
                    self.data_loader.append([torch.Tensor(data[i:i + self.batch_size]).cuda(),
                                             torch.Tensor(label[i:i + self.batch_size]).cuda()])
                else:
                    self.data_loader.append([torch.Tensor(data[i:i + self.batch_size]),
                                             torch.Tensor(label[i:i + self.batch_size])])

    def get_array(self, Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr, payoff, forgone,
                                                                                                        prev_answer):
        # check lot shapes
        Ha = float(Ha)
        pHa = float(pHa)
        La = float(La)
        No_ShapeA = S_ShapeA = R_ShapeA = L_ShapeA = 0.0
        if LotShapeA == '-':
            No_ShapeA = 1.0
        elif LotShapeA == 'Symm':
            S_ShapeA = 1.0
        elif LotShapeA == 'L-skew':
            R_ShapeA = 1.0
        elif LotShapeA == 'R-skew':
            L_ShapeA = 1.0
        LotNumA = float(LotNumA)
        Hb = float(Hb)
        pHb = float(pHb)
        Lb = float(Lb)
        No_ShapeB = S_ShapeB = R_ShapeB = L_ShapeB = 0.0
        if LotShapeB == '-':
            No_ShapeB = 1.0
        elif LotShapeB == 'Symm':
            S_ShapeB = 1.0
        elif LotShapeB == 'L-skew':
            R_ShapeB = 1.0
        elif LotShapeA == 'R-skew':
            L_ShapeB = 1.0
        LotNumB = float(LotNumB)
        Amb = float(Amb)
        Corr = float(Corr)
        if payoff == '-':
            payoff = 0.0
        else:
            payoff = float(payoff)
        if forgone == '-':
            forgone = 0.0
        else:
            forgone = float(forgone)

        return [Ha, pHa, La, No_ShapeA, S_ShapeA, R_ShapeA, L_ShapeA, LotNumA,
                Hb, pHb, Lb, No_ShapeB, S_ShapeB, R_ShapeB, L_ShapeB, LotNumB,
                Amb, Corr, payoff, forgone, prev_answer[0], prev_answer[1]]

