import csv
import pickle

datafolder = '../data/'

X = []
y = []
temperatures = []
with open(datafolder + 'temperature_history.csv', 'r') as temp:
    temperature_reader = csv.reader(temp, delimiter=',')  # read file
    next(temperature_reader)  # skip header
    for line in temperature_reader:
        for i in range(1, 25):  # from each line 24 new training examples are generated (one for each hour)
            if len(line[i+3]) == 0:  # skip empty fields
                break
            values = []
            values.extend(int(line[j]) for j in range(4))  # id, year, month, day
            values.append(i)  # hour
            values.append(int(line[i+3]))  # temperature
            temperatures.append(values)

with open(datafolder + 'Load_history.csv', 'r') as load:
    load_reader = csv.reader(load, delimiter=',')  # read file
    next(load_reader)  # skip header
    count = 0
    for line in load_reader:
        for i in range(1, 25):  # from each line 24 new training examples are generated (one for each hour)
            if len(line[i+3]) == 0:  # skip empty fields
                break
            features = []
            features.extend(int(line[j]) for j in range(4))  # id, year, month, day
            features.append(i)  # hour
            for row in temperatures:  # append temperature values from all stations
                if row[1] == features[1] and row[2] == features[2] and row[3] == features[3] and row[4] == features[4]:
                    features.append(row[5])
            X.append(features)  # add example to X matrix
            y.append(int(line[i+3].replace(',', '')))  # add label to y vactor
        if count % 1000 == 0:  # checking progress
            print count
        count += 1
pickle.dump(X, open(datafolder + 'features.p', 'wb'))
pickle.dump(y, open(datafolder + 'labels.p', 'wb'))

# writer = csv.writer(open(datafolder + 'load.csv', 'w', newline=''), delimiter=',')
# for i in range(len(X)):
#     writer.writerow(X[i])
