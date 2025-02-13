def train_model(model, train_data, val_data, epochs, batch_size, model_save_path):
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint]
    )

    return history

def load_trained_model(model_save_path):
    from keras.models import load_model
    return load_model(model_save_path)