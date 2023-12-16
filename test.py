import copy
import itertools


train_tuples = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
tuples = copy.deepcopy(list(itertools.chain.from_iterable(train_tuples)))

print("1",itertools.chain.from_iterable(train_tuples))
print("2",list(itertools.chain.from_iterable(train_tuples)))
print("3",tuples)