import os
import struct
import pandas as pd
import datetime

dirname = os.path.dirname(__file__)
print(dirname)
filename = dirname + '/features_ecu_full_rgb/im000'


def rows_generator(path):
    with open(path, "rb") as binary_file:
        n_features = struct.unpack("i", binary_file.read(4))[0]
        n_labels = struct.unpack("i", binary_file.read(4))[0]
        h = struct.unpack("i", binary_file.read(4))[0]
        w = struct.unpack("i", binary_file.read(4))[0]
        for _ in range(h * w):
            features = [struct.unpack_from("f", binary_file.read(4))[0] for _ in range(n_features)]
            labels = [struct.unpack_from("f", binary_file.read(4))[0] for _ in range(n_labels)]
            yield features, labels


x_list = []
y_list = []

for i in range(1, 51):
    if i < 10:
        path = filename + "0" + str(i)
    else:
        path = filename + str(i)
    print(path)
    for feature, label in rows_generator(path):
        x_list.append(feature)
        y_list.append(label)
X = pd.DataFrame(x_list)
Y = pd.DataFrame(y_list)

X.columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16",
             "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31",
             "f32"]

Y.columns = ["l1"]
print(X.head())
print(Y.head())
