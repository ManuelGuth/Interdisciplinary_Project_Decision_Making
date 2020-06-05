from data.raw_data.data_parser import DataParser

# intended to do a Train/Val split on the original training set in order to train a ML/DL model with ccobra.


class DataParserTrainVal(DataParser):
    """
    Class to split the train data into train and validation in order to be able to use train val split in ccobra for
    ML models.
    """
    def __init__(self, data=None):
        super(DataParserTrainVal, self).__init__()
        if data is not None:
            if data.rfind('/') < 0:
                self.file_name_test = data[:-4] + '_train_transformed.csv'
                self.file_name_val = data[:-4] + '_val_transformed.csv'
            else:
                index = data.rfind('/') + 1
                self.file_name_test = data[index:-4] + '_train_transformed.csv'
                self.file_name_val = data[index:-4] + '_val_transformed.csv'
            data_loader = self.load_data(data)
            train, val = self.split_data(data_loader)
            train = self.transform_data(train)
            val = self.transform_data(val)
            self.safe_data(train, val)

    def split_data(self, data):
        """
        there are 681 participants, we want to use 20% of the data as validation which is 137 participants.
        :param data: the data to split as a list
        :return: train and val data as a list
        """
        header = data[0]
        data = data[1:]
        train = [header]
        val = [header]
        person = -1
        for i, row in enumerate(data):
            if i % 750 == 0:
                person += 1
            if person % 5 == 0:     # val person
                val.append(row)
            else:                   # train person
                train.append(row)

        return train, val

    def safe_data(self, train, val):
        with open(self.file_name_test, 'w') as file:
            header = ''
            for entry in train[0]:
                header += ',' + entry
            print(header[1:], file=file)

            for di in train:
                data_point = ''
                for entry in di:
                    data_point += ',' + di[entry]
                print(data_point[1:], file=file)

        with open(self.file_name_val, 'w') as file:
            header = ''
            for entry in val[0]:
                header += ',' + entry
            print(header[1:], file=file)

            for di in val:
                data_point = ''
                for entry in di:
                    data_point += ',' + di[entry]
                print(data_point[1:], file=file)


if __name__ == "__main__":
    data_Parser = DataParserTrainVal(data='../raw_data/All estimation raw data.csv')
