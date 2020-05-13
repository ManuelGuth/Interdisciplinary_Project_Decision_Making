import csv


class DataParser:
    def __init__(self, data):
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
        print(data[0], data[1])
        for row in data:
            pass
        return True

    def safe_data(self, data):
        pass


if __name__ == "__main__":
    data_Parser = DataParser("raw_data/raw-comp-set-data-Track-1.csv")

