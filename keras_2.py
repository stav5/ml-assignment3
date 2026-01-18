from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ---------------------------
#  1) Load Data
# ---------------------------
(X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

X_full = np.concatenate([X_train_full, X_test_full], axis=0)
y_full = np.concatenate([y_train_full, y_test_full], axis=0)

# Calculate the sizes for 70% train, 30% test
train_size = int(0.7 * len(X_full))
test_size = len(X_full) - train_size

np.random.seed(123) ##
indices = np.random.permutation(len(X_full))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train, y_train = X_full[train_indices], y_full[train_indices]
X_test, y_test = X_full[test_indices], y_full[test_indices]

x_train = ((X_train.reshape(-1, 28 * 28).astype('float32') / 255.) - .5) * 2
x_test = ((X_test.reshape(-1, 28 * 28).astype('float32') / 255.) - .5) * 2

# One-hot encode labels
num_classes = 10
y_train_onehot = np.eye(num_classes)[y_train]
y_test_onehot = np.eye(num_classes)[y_test]

# ---------------------------
#  2) Build the Keras Model
# ---------------------------
model = Sequential()
model.add(Dense(input_dim=28*28, units=500, activation='sigmoid'))
model.add(Dense(units=500, activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='mse',
              optimizer=SGD(learning_rate=0.1),
              metrics=['accuracy'])

# ---------------------------
#  3) Train the Model
# ---------------------------
history = model.fit(
    x_train,
    y_train_onehot,
    batch_size=100,
    epochs=50,
    validation_split=0.1,
    verbose=1
)

# ---------------------------
#  4) Evaluate on Test Set
# ---------------------------
score = model.evaluate(x_test, y_test_onehot, verbose=0)
test_loss = score[0]
test_acc = score[1]

y_pred_probs = model.predict(x_test)
macro_auc_ovr = roc_auc_score(y_test_onehot, y_pred_probs, average='macro', multi_class='ovr')

# ---------------------------
#  5) Save Metrics to TXT
# ---------------------------
model_name = "keras"
report_file = f"{model_name}_report.txt"

with open(report_file, 'w') as f:
    # Final test metrics
    f.write(f"=== Final Test Results ===\n")
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_acc}\n")
    f.write(f"Macro-AUC (One-vs-Rest): {macro_auc_ovr}\n\n")

    # Final epoch metrics from training
    f.write(f"=== Final Training Epoch ===\n")
    f.write(f"Train Loss (epoch {len(history.history['loss'])}): "
            f"{history.history['loss'][-1]}\n")
    f.write(f"Validation Loss (epoch {len(history.history['val_loss'])}): "
            f"{history.history['val_loss'][-1]}\n")
    f.write(f"Train Accuracy (epoch {len(history.history['accuracy'])}): "
            f"{history.history['accuracy'][-1]}\n")
    f.write(f"Validation Accuracy (epoch {len(history.history['val_accuracy'])}): "
            f"{history.history['val_accuracy'][-1]}\n")

print("=== Final Test Results ===")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
print("Macro-AUC (One-vs-Rest):", macro_auc_ovr)

# ---------------------------
#  6) Save Plots
# ---------------------------
# (a) Loss & Accuracy Curves

# MSE Curve
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train MSE')
plt.plot(history.history['val_loss'], label='Validation MSE')
plt.title('Training & Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.savefig(f'{model_name}_training_MSE_curve.png', dpi=300)

# Accuracy Curve
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'{model_name}_training_accuracy_curve.png', dpi=300)

# plt.show()

# (b) ROC Curves per class
y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve for Each Class')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.savefig(f'{model_name}_roc_curve_each_class.png', dpi=300)
# plt.show()

# (c) Macro-Averaged ROC Curve
fpr_macro, tpr_macro, _ = roc_curve(y_test_bin.ravel(), y_pred_probs.ravel())
macro_roc_auc = auc(fpr_macro, tpr_macro)

plt.figure(figsize=(8, 6))
plt.plot(fpr_macro, tpr_macro,
         label=f'Macro-Averaged ROC (AUC = {macro_roc_auc:.3f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Macro-Averaged ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig(f'{model_name}_macro_roc_curve.png', dpi=300)
# plt.show()



# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras.optimizers import SGD
# from keras.datasets import mnist
# import numpy as np

# # -----------------------
# # Reproducibility
# # -----------------------
# SEED = 42
# np.random.seed(SEED)

# # -----------------------
# # Load MNIST and make a 70/30 split from the FULL set (train+test)
# # -----------------------
# (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

# X_full = np.concatenate([X_train_full, X_test_full], axis=0)
# y_full = np.concatenate([y_train_full, y_test_full], axis=0)

# total_samples = len(X_full)
# train_size = int(0.7 * total_samples)
# test_size = total_samples - train_size

# indices = np.random.permutation(total_samples)
# train_idx = indices[:train_size]
# test_idx  = indices[train_size:]

# X_train, y_train_int = X_full[train_idx], y_full[train_idx]
# X_test,  y_test_int  = X_full[test_idx],  y_full[test_idx]

# print("Split summary")
# print(f"  Total samples: {total_samples}")
# print(f"  Train samples: {train_size} ({train_size/total_samples:.0%})")
# print(f"  Test  samples: {test_size} ({test_size/total_samples:.0%})")
# print(f"  X_train: {X_train.shape}, y_train: {y_train_int.shape}")
# print(f"  X_test : {X_test.shape},  y_test : {y_test_int.shape}")

# # -----------------------
# # Preprocessing
# #   1) Flatten 28x28 -> 784
# #   2) Scale pixels to [-1, 1]  (not just [0,1])
# # -----------------------
# x_train = X_train.reshape(-1, 28 * 28).astype("float32")
# x_test  = X_test.reshape(-1, 28 * 28).astype("float32")

# x_train = ((x_train / 255.0) - 0.5) * 2.0
# x_test  = ((x_test  / 255.0) - 0.5) * 2.0

# print("Preprocessing check")
# print(f"  x_train dtype: {x_train.dtype}, range: [{x_train.min():.3f}, {x_train.max():.3f}]")
# print(f"  x_test  dtype: {x_test.dtype},  range: [{x_test.min():.3f}, {x_test.max():.3f}]")

# # -----------------------
# # Labels: one-hot encoding (0..9 -> length-10 vector)
# # -----------------------
# num_classes = 10
# y_train = np.eye(num_classes)[y_train_int]
# y_test  = np.eye(num_classes)[y_test_int]

# print("Labels check")
# print(f"  y_train one-hot: {y_train.shape}, y_test one-hot: {y_test.shape}")

# # -----------------------
# # Model (Lecture Solution 1 style)
# # 784 -> 500(sigmoid) -> 500(sigmoid) -> 10(softmax)
# # -----------------------
# model = Sequential()
# model.add(Dense(input_dim=28 * 28, units=500))
# model.add(Activation("sigmoid"))
# model.add(Dense(units=500))
# model.add(Activation("sigmoid"))
# model.add(Dense(units=10))
# model.add(Activation("softmax"))

# model.compile(
#     loss="mse",
#     optimizer=SGD(learning_rate=0.1),
#     metrics=["accuracy"]
# )

# print("\nModel summary")
# model.summary()

# # -----------------------
# # Training
# # -----------------------
# print("\nTraining")
# history = model.fit(
#     x_train, y_train,
#     batch_size=100,
#     epochs=20,
#     verbose=1
# )

# # -----------------------
# # Evaluation
# # -----------------------
# print("\nEvaluation on test set")
# test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
# print(f"  Test loss: {test_loss:.4f}")
# print(f"  Test accuracy: {test_acc:.4f}")
