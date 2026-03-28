import tensorflow as tf
import re
import numpy as np

from tensorflow.keras.layers import TextVectorization

train_dataset_path = '../../resources/nn_sms/train-data.tsv'
valid_dataset_path = '../../resources/nn_sms/valid-data.tsv'

train_text = open(train_dataset_path, 'rb').read().decode(encoding='utf-8')
valid_text = open(valid_dataset_path, 'rb').read().decode(encoding='utf-8')

pattern = r'(ham|spam)\s+(.*?)(?=\s+ham|\s+spam|$)'
train_matches = re.findall(pattern, train_text, re.DOTALL)
train_data = [m[1].strip() for m in train_matches]
train_labels = [m[0].strip() for m in train_matches]

valid_matches = re.findall(pattern, valid_text, re.DOTALL)
valid_data = [m[1].strip() for m in valid_matches]
valid_labels = [m[0].strip() for m in valid_matches]

train_labels_vec = np.array([1 if v == 'spam' else 0 for v in train_labels])
valid_labels_vec = np.array([1 if v == 'spam' else 0 for v in valid_labels])

vectorize_layer = TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=30
)

model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Embedding(10000, output_dim=32),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

vectorize_layer.adapt(train_data)

train_x = np.array(train_data, dtype=object)
valid_x = np.array(valid_data, dtype=object)

model.fit(
    train_x,
    train_labels_vec,
    epochs=10,
    batch_size=64,
    validation_data=(valid_x, valid_labels_vec),
    verbose=1
)


def predict_message(pred_text):
    input_data = np.array([pred_text], dtype=object)

    prediction_score = model.predict(input_data, verbose=0)[0][0]

    label = "spam" if prediction_score >= 0.5 else "ham"

    return [prediction_score, label]


pred_text = "how are you doing today?"

my_prediction = predict_message(pred_text)
print(my_prediction)


def test_predictions():
    test_messages = ["how are you doing today",
                     "sale today! to stop texts call 98912460324",
                     "i dont want to go. can we try it a different day? available sat",
                     "our new mobile video service is live. just install on your phone to start watching.",
                     "you have won £1000 cash! call to claim your prize.",
                     "i'll bring it tomorrow. don't forget the milk.",
                     "wow, is your arm alright. that happened to me one time too"
                     ]

    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
    passed = True

    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(msg)
        if prediction[1] != ans:
            passed = False

    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")


test_predictions()
