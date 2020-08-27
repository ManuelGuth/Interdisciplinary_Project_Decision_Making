if __name__ == "__main__":
    with open('decision-making-train.csv') as testDataFile:
        with open('singel_prediction_data/decision-making-train.csv', 'a') as newFile:
            for i, line in enumerate(testDataFile):
                if i == 0 or (i-1) % 25 == 0:
                    print(line, file=newFile, end='')
