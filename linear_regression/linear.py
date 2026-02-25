import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

# 1. Downloading the data
print("Downloading the data...")
ds, info = tfds.load('titanic', split='train', with_info=True, as_supervised=True)


def prepare_data(dataset):
    data = []
    for features, label in tfds.as_numpy(dataset):
        row = {
            'age': float(features['age']),
            'fare': float(features['fare']),
            'pclass': float(features['pclass']),
            'sex': 1.0 if features['sex'] == b'female' else 0.0,
            'survived': int(label)
        }
        data.append(row)
    return pd.DataFrame(data)


def prepare_data_v2(dataset):
    data = []
    for features, label in tfds.as_numpy(dataset):
        row = {
            'age': float(features['age']),
            'fare': float(features['fare']),
            'pclass': float(features['pclass']),
            'sex': 1.0 if features['sex'] == b'female' else 0.0,
            'sibsp': float(features['sibsp']),
            'parch': float(features['parch']),
            'survived': int(label)
        }
        emb = features['embarked']
        row['emb_C'] = 1.0 if emb == b'C' else 0.0
        row['emb_Q'] = 1.0 if emb == b'Q' else 0.0
        row['emb_S'] = 1.0 if emb == b'S' else 0.0

        data.append(row)
    df = pd.DataFrame(data)
    df['age'] = df['age'] / 80.0
    df['fare'] = df['fare'] / 512.0
    return df


full_df = prepare_data_v2(ds)

train_df = full_df.sample(frac=0.8, random_state=42)
eval_df = full_df.drop(train_df.index)

y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')

# 2. Building linear regression model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(9,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 3. Compiling model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Training
print("\nStarting training of the linear regression model...")
model.fit(train_df, y_train, epochs=40, batch_size=32, verbose=1)
