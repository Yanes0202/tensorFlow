import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf

from sklearn.model_selection import train_test_split

dataset = pd.read_csv('../../resources/insurance.csv')

translated_dataset = pd.get_dummies(dataset, columns=['sex', 'smoker', 'region'], dtype=int)

x = translated_dataset.drop('expenses', axis=1)
y = translated_dataset['expenses']

train_dataset, test_dataset, train_labels, test_labels = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(11,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mae', 'mse']
)

model.fit(train_dataset, train_labels, epochs=100, batch_size=64, verbose=1)

loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
    print("You passed the challenge. Great job!")
else:
    print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
