from WrappedMedMNIST import WrappedMedMNIST

# Ver lista completa
print(WrappedMedMNIST.all_datasets())

# Ver somente os datasets compatíveis com aprendizado supervisionado multi-classe
print(WrappedMedMNIST.available_datasets())

# Carregando um dataset qualquer (ex: multilabel, binário etc)
dataset = WrappedMedMNIST.load_any('chestmnist')

X, y = dataset['data'], dataset['target']
print(X.shape, y.shape)