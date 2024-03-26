# take the data.pickle file, load the data, and then train a model on it!

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Transform the data (which is a list) into arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    stratify=labels)
print('Data Split')

model = RandomForestClassifier()
print('Model Initiated')

model.fit(X_train, y_train)
print('Model Fit')

y_predict = model.predict(X_test)

score = accuracy_score(y_predict, y_test)

print(f'{score*100}% of samples were classified correctly')

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
