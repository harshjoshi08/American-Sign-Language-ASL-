import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# ==============================
# 0. BASIC SETTINGS
# ==============================
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Paths to your 2 trained models (full .h5 files)
model1_path = r"C:\Harsh Works\code\American Sign Language\asl_training_outputs\asl_model_cnn.h5"
model2_path = r"C:\Harsh Works\code\American Sign Language\asl_training_outputs\asl_model_mobilenetv2.h5"

print("Model 1 path exists? ", os.path.exists(model1_path), "->", model1_path)
print("Model 2 path exists? ", os.path.exists(model2_path), "->", model2_path)

# In training you used image size 128x128
MODEL_INPUT_SIZE = (128, 128)  # H, W
num_classes = 6  # ["Hello","I Love You","Okay","Please","Thank you","Yes"]

offset = 20
imgSize = 300  # size of white canvas

# Label order must match training dataset class_names
labels = ["Hello", "I Love You", "Okay", "Please", "Thank you", "Yes"]

# ==============================
# 1. RECREATE ARCHITECTURES
# ==============================

# Data augmentation (same as training)
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)

def build_model_1(input_shape=(MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3), num_classes=6):
    """Simple CNN (baseline) – must match training architecture."""
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        data_augmentation,

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    # For inference we don't need to compile, but it's fine if we do
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_model_2(input_shape=(MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3), num_classes=6):
    """MobileNetV2 transfer learning model – must match training architecture."""
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # as in training

    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    x = data_augmentation(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ==============================
# 2. BUILD MODELS + LOAD WEIGHTS
# ==============================
model1 = None
model2 = None

if os.path.exists(model1_path):
    print("Building architecture for Model 1 (CNN)...")
    model1 = build_model_1(num_classes=num_classes)
    print("Loading weights for Model 1 from:", model1_path)
    model1.load_weights(model1_path)
    print("✅ Model 1 weights loaded.")

if os.path.exists(model2_path):
    print("Building architecture for Model 2 (MobileNetV2)...")
    model2 = build_model_2(num_classes=num_classes)
    print("Loading weights for Model 2 from:", model2_path)
    model2.load_weights(model2_path)
    print("✅ Model 2 weights loaded.")

if model1 is None and model2 is None:
    raise RuntimeError("No valid model files found. Check your .h5 paths.")

# Choose starting model
if model2 is not None:
    current_model = model2
    current_model_name = "Model 2 (MobileNetV2)"
else:
    current_model = model1
    current_model_name = "Model 1 (CNN)"

# ==============================
# 3. HELPER: PREDICT FUNCTION
# ==============================
def predict_with_model(imgWhite, model):
    """
    Takes 300x300 imgWhite, resizes to 128x128,
    normalizes (Rescaling layer handles it) by just casting to float32,
    and returns (index, confidence).
    """
    imgInput = cv2.resize(imgWhite, MODEL_INPUT_SIZE)
    imgInput = imgInput.astype("float32")  # DO NOT divide by 255 – Rescaling layer does that
    imgInput = np.expand_dims(imgInput, axis=0)  # (1, H, W, 3)

    preds = model.predict(imgInput, verbose=0)[0]  # (num_classes,)
    index = int(np.argmax(preds))
    confidence = float(preds[index])
    return index, confidence

# ==============================
# 4. REAL-TIME LOOP
# ==============================
while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        # safety check
        if imgCrop.size != 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # 🔮 predict with the currently selected model
            index, confidence = predict_with_model(imgWhite, current_model)
            gesture = labels[index]

            print(f"{current_model_name} → {gesture} ({confidence:.2f})")

            # UI: banner above hand
            cv2.rectangle(
                imgOutput,
                (x - offset, y - offset - 80),
                (x - offset + 450, y - offset - 20),
                (0, 255, 0),
                cv2.FILLED,
            )

            cv2.putText(
                imgOutput,
                f"{current_model_name}: {gesture} ({confidence:.2f})",
                (x - offset + 10, y - offset - 35),
                cv2.FONT_HERSHEY_COMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )

            # bounding box
            cv2.rectangle(
                imgOutput,
                (x - offset, y - offset),
                (x + w + offset, y + h + offset),
                (0, 255, 0),
                4,
            )

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # show which model is currently active (top-left)
    cv2.putText(
        imgOutput,
        f"Using: {current_model_name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1) & 0xFF

    # press '1' -> Model 1 (CNN)
    if key == ord('1'):
        if model1 is not None:
            current_model = model1
            current_model_name = "Model 1 (CNN)"
            print(">>> Switched to Model 1 (CNN) <<<")
        else:
            print("Model 1 not available (file missing).")

    # press '2' -> Model 2 (MobileNetV2)
    if key == ord('2'):
        if model2 is not None:
            current_model = model2
            current_model_name = "Model 2 (MobileNetV2)"
            print(">>> Switched to Model 2 (MobileNetV2) <<<")
        else:
            print("Model 2 not available (file missing).")

    # ESC to quit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
