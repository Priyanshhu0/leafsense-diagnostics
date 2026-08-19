import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# ---------- Paths ----------
TRAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "train")
VAL_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "val")
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "model", "plant_disease_model.keras")
GRAPH_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "accuracy_graph.png")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20


def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    if not os.path.exists(TRAIN_DIR) or len(os.listdir(TRAIN_DIR)) == 0:
        raise FileNotFoundError(
            f"No training data found in {TRAIN_DIR}. "
            "Create a subfolder per class and put leaf images inside "
            "(see the README for dataset instructions)."
        )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    print("Classes found:", train_generator.class_indices)

    num_classes = len(train_generator.class_indices)
    model = build_model(num_classes)
    model.summary()

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

    checkpoint = ModelCheckpoint(
        MODEL_OUTPUT_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    print("\nStarting training...\n")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    best_accuracy = max(history.history["val_accuracy"])
    print(f"\nBest validation accuracy: {best_accuracy * 100:.2f}%")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(GRAPH_OUTPUT_PATH)
    print(f"Accuracy graph saved to: {GRAPH_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
