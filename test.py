import copy
import itertools
import pickle

pickle_file = open("pickles/Oxford/Oxford_train_queries.pickle","rb")

data = pickle.load(pickle_file)

print(data[0].id)
