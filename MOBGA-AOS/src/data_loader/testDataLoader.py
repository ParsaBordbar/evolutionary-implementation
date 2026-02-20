from data import load_dataset, normalize
from configs.configs import DATA_PATH

X_train, X_test, y_train, y_test, n_features = load_dataset('{DATA_PATH}/1771446100658_DS02.csv')
X_train, X_test = normalize(X_train, X_test)
print(f'DS02 loaded successfully')
print(f'  Features : {n_features}')
print(f'  Train    : {X_train.shape}  labels: {y_train.shape}')
print(f'  Test     : {X_test.shape}   labels: {y_test.shape}')
print(f'  Classes  : {len(set(y_train.tolist() + y_test.tolist()))}')
print(f'  X_train range: [{X_train.min():.2f}, {X_train.max():.2f}]')