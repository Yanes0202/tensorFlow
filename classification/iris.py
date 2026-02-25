import tensorflow as tf
import pandas as pd
import numpy as np

# 1. Dane i stałe
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file("iris_training.csv", "../resources/")
test_path = tf.keras.utils.get_file("iris_test.csv", "../resources/")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')

# 2. Budowa modelu DNN (Deep Neural Network)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),           # 4 cechy wejściowe (Sepal/Petal)
    tf.keras.layers.Dense(30, activation='relu'), # Pierwsza warstwa (30 neuronów)
    tf.keras.layers.Dense(10, activation='relu'), # Druga warstwa (10 neuronów)
    tf.keras.layers.Dense(3, activation='softmax')# Wyjście: 3 klasy (Gatunki)
])

# 3. Kompilacja
# 'sparse_categorical_crossentropy' jest idealne, gdy labelami są liczby (0, 1, 2)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Trening
print("Trenowanie sieci neuronowej Iris...")
model.fit(train, train_y, epochs=150, batch_size=32, verbose=0)

# 5. Ewaluacja
test_loss, test_acc = model.evaluate(test, test_y, verbose=2)
print(f'\nDokładność klasyfikacji: {test_acc:.2%}')


def predict_iris(model, species_names):
    print("\n--- Iris Species Predictor ---")
    print("Wpisz parametry kwiatka (np. 5.1, 3.5, 1.4, 0.2):")

    features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    user_input = []

    for feature in features:
        while True:
            val = input(f"{feature}: ")
            try:
                # Używamy float, bo wymiary kwiatów rzadko są liczbami całkowitymi
                user_input.append(float(val))
                break
            except ValueError:
                print("To nie jest poprawna liczba. Spróbuj ponownie.")

    # Model oczekuje paczki danych (batch), więc zamieniamy [1, 2, 3, 4] na [[1, 2, 3, 4]]
    prediction_data = np.array([user_input])

    # Wykonujemy predykcję
    predictions = model.predict(prediction_data, verbose=0)

    # Wybieramy indeks z najwyższym prawdopodobieństwem
    class_id = np.argmax(predictions[0])
    probability = predictions[0][class_id]

    print("\n" + "=" * 30)
    print(f'Predykcja: "{species_names[class_id]}"')
    print(f'Pewność: {100 * probability:.1f}%')
    print("=" * 30)


# Wywołanie funkcji (zakładając, że model i SPECIES już istnieją)
predict_iris(model, SPECIES)