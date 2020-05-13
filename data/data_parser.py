import csv


class DataParser:
    """
    Class to parse the raw experiment data form the CPC-18 and transform it to a format readable by ccobra.
    """
    def __init__(self, data):
        self.previous_feedback = None
        self.file_name = data[:-4] + 'transformed.csv'

        data_loader = self.load_data(data)
        transformed_data = self.transform_data(data_loader)
        self.safe_data(transformed_data)

    def load_data(self, data):
        with open(data) as data_set:
            data_reader = csv.reader(data_set, delimiter=';')
            data = []
            for row in data_reader:
                data.append(row)
        return data

    def transform_data(self, data):
        # important/need features
        features = ['SubjID', 'Location', 'Gender', 'Age', 'Ha', 'pHa', 'La', 'LotShapeA', 'LotNumA',
                    'Hb', 'pHb', 'Lb', 'LotShapeB', 'LotNumB', 'Amb', 'Corr', 'B', 'Payoff', 'Forgone',
                    'Feedback', 'Order', 'Trial']

        # get feature index
        feature_dict = {}
        for feature in features:
            feature_dict[feature] = data[0].index(feature)

        # get feature value
        transformed_data = []
        for data_point in data[1:]:
            data_dict = {}
            for feature in feature_dict:
                data_dict[feature] = data_point[feature_dict[feature]]
            transformed_data.append(self.transform_to_ccobra(data_dict))
        return transformed_data

    def transform_to_ccobra(self, data):
        """
        :param data: dictionary with feature values for one data point
        :return: dictionary with features in the needed ccobra style
        """
        ccobra_dict = {}
        # get the easy data points first
        ccobra_dict['id'] = data['SubjID']
        ccobra_dict['choices'] = 'A;B'
        ccobra_dict['response_type'] = 'single-choice'
        ccobra_dict['domain'] = 'decision-making'
        ccobra_dict['age'] = data['Age']
        ccobra_dict['gender'] = data['Gender']
        ccobra_dict['location'] = data['Location']

        if data['B'] == '1':
            response = 'B'
        else:
            response = 'A'
        ccobra_dict['response'] = response

        ccobra_dict['sequence'] = str(int(data['Order'])*100 + int(data['Trial']))

        ccobra_dict['task'] = data['Ha'] + ';' + data['pHa'] + ';' + data['La'] + ';' + data['LotShapeA'] + ';' + \
                              data['LotNumA'] + '/' + data['Hb'] + ';' + data['pHb'] + ';' + data['Lb'] + ';' + \
                              data['LotShapeB'] + ';' + data['LotNumB'] + '/' + data['Amb'] + ';' + data['Corr']

        if self.previous_feedback is not None:
            ccobra_dict['prev_feedback'] = self.previous_feedback['payoff'] + ';' + self.previous_feedback['forgone']
        else:
            ccobra_dict['prev_feedback'] = '-;-'

        if data['Feedback'] == '0' or data['Order'] == '30' and data['Trial'] == '25':
            self.previous_feedback = None
        else:
            self.previous_feedback = {'payoff': data['Payoff'], 'forgone': data['Forgone']}

        return ccobra_dict

    def safe_data(self, data):
        with open(self.file_name, 'w') as file:
            header = ''
            for entry in data[0]:
                header += ',' + entry
            print(header[1:], file=file)

            for di in data:
                data_point = ''
                for entry in di:
                    data_point += ',' + di[entry]
                print(data_point[1:], file=file)


if __name__ == "__main__":
    data_Parser = DataParser('raw_data/raw-comp-set-data-Track-1.csv')
    data_Parser = DataParser('raw_data/All estimation raw data.csv')

