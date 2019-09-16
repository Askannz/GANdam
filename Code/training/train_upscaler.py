import os
from pathlib import Path
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from models import make_upscaler
from data import load_upscaler_data

BATCH_SIZE = 16
LR = 0.000001
NB_EPOCHS = 100000
EARLY_STOPPING_PATIENCE = 20

def main():

    upscaler = make_upscaler()
    samples_data, labels_data = load_upscaler_data()

    assert samples_data.shape[0] == labels_data.shape[0]

    os.makedirs("../../Generated/training/upscaler_models/final", exist_ok=True)

    optimizer = Adam(LR, 0.9)
    upscaler.compile(loss="mean_squared_error", optimizer=optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=1, restore_best_weights=True)

    try:
        upscaler.fit(samples_data, labels_data, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, validation_split=0.05, callbacks=[early_stopping])

    except KeyboardInterrupt:
        saved_folder_path = Path("../../Generated/training/upscaler_models/")
        os.makedirs(saved_folder_path, exist_ok=True)
        print("\nSaving models to %s" % str(saved_folder_path))
        upscaler.save(str(saved_folder_path / "final" / "upscaler.h5"))

if __name__ == "__main__":
    main()
