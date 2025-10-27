import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- Import your data preparation script ---
import data_preparation as dp

# --- 1. MODEL & TRAINING CONFIGURATION ---
# These can be imported from dp or redefined here
IMG_SIZE = dp.IMG_SIZE
NUM_CLASSES = dp.NUM_CLASSES

# Training Hyperparameters
EPOCHS_PHASE_1 = 10  # Epochs for training just the new head
EPOCHS_PHASE_2 = 10  # Additional epochs for fine-tuning the full model
LR_PHASE_1 = 0.001
LR_PHASE_2 = 1e-5    # 0.00001 (very low learning rate for fine-tuning)

# --- 2. MODEL BUILDING ---

def build_model(num_classes):
    """
    Builds the ResNet-50 transfer learning model.
    """
    # 1. Load Base Model (ResNet-50 pre-trained on ImageNet)
    base_model = ResNet50(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,  # Exclude the final 1000-class classifier
        weights='imagenet'
    )

    # 2. Freeze the Base Model
    base_model.trainable = False

    # 3. Add Custom Classifier "Head"
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Flattens the features
    x = Dense(1024, activation='relu')(x) # A dense layer
    x = Dropout(0.5)(x)                  # Regularization
    predictions = Dense(num_classes, activation='softmax')(x)

    # 4. Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model

def plot_history(history_dict, fine_tune_epoch_start):
    """Plots training and validation accuracy and loss."""
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.axvline(fine_tune_epoch_start, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.axvline(fine_tune_epoch_start, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.legend()

    plt.savefig('training_history.png')
    print("\nSaved training history plot to 'training_history.png'")

# --- 3. MAIN TRAINING & EVALUATION SCRIPT ---

def main():
    # --- Step 1: Get Data from your preparation script ---
    # This is the key link:
    train_ds, val_ds, test_ds, test_labels = dp.main()
    
    if train_ds is None:
        print("❌ Data loading failed. Aborting training.")
        return

    # --- Step 2: Build the model ---
    print("\n--- 🏗️ Building Model ---")
    model, base_model = build_model(NUM_CLASSES)
    model.summary()
    
    # Define metrics
    METRICS = [
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint("dr_best_model.keras", save_best_only=True, monitor="val_loss"),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    # --- Step 3: Phase 1 Training (Train the Head) ---
    print(f"\n--- 🧠 Training Phase 1: Head Only (Epochs: {EPOCHS_PHASE_1}) ---")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_PHASE_1),
        loss='categorical_crossentropy',
        metrics=METRICS
    )
    
    history = model.fit(
        train_ds,
        epochs=EPOCHS_PHASE_1,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # --- Step 4: Phase 2 Training (Fine-Tuning) ---
    print(f"\n--- 🛠️ Training Phase 2: Fine-Tuning (Epochs: {EPOCHS_PHASE_2}) ---")
    
    base_model.trainable = True # Unfreeze the base
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_PHASE_2), # Use low LR
        loss='categorical_crossentropy',
        metrics=METRICS
    )
    
    total_epochs = EPOCHS_PHASE_1 + EPOCHS_PHASE_2
    
    history_fine_tune = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1], # Start from where we left off
        validation_data=val_ds,
        callbacks=callbacks
    )

    # --- Step 5: Final Evaluation on Test Set ---
    print("\n--- 📊 Final Evaluation on TEST Set ---")
    
    print("Loading best model weights from 'dr_best_model.keras'...")
    model.load_weights("dr_best_model.keras")
    
    test_loss, test_acc, test_auc, test_prec, test_rec = model.evaluate(test_ds)
    print("\nTest Set Performance:")
    print(f"  Loss:       {test_loss:.4f}")
    print(f"  Accuracy:   {test_acc:.4f}")
    print(f"  AUC:        {test_auc:.4f}")
    print(f"  Precision:  {test_prec:.4f}")
    print(f"  Recall:     {test_rec:.4f}")
    
    # --- Step 6: Classification Report & Confusion Matrix ---
    print("\nGenerating Classification Report and Confusion Matrix...")
    
    y_pred_probs = model.predict(test_ds)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(test_labels, axis=1)
    
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_classes, y_pred_classes))
    
    # --- Step 7: Plot History ---
    # Combine history objects
    full_history = {}
    for key in history.history.keys():
        full_history[key] = history.history[key] + history_fine_tune.history[key]
    
    plot_history(full_history, fine_tune_epoch_start=EPOCHS_PHASE_1)

    print("\n--- ✅ Project Complete ---")

if __name__ == "__main__":
    main()