import numpy as np
np.random.seed(6)
from tensorflow import set_random_seed
set_random_seed(6)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.optimizers
from keras.models import load_model
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, LearningRateScheduler, CSVLogger
import os
import math


MODEL_NUM = 'lstm-L14_0.0001_try2'
SAVE = False
LOAD = False
NUM_EPOCHS = 25
HIDDEN_UNITS = 500
LEARNING_RATE = 0.0001
DROP = 0.0
LAYERS = 1


max_len = 1300
DATA_DIR = '/group/project/'

## LOAD DATA
print("Loading data...")
# Load TRAINING data
x_train = np.memmap(DATA_DIR+'extracted_feats/train-feats-rnn',
            dtype='float32', mode='r', shape=(3014, max_len, 90))
# x_train = sequence.pad_sequences(train_fp, maxlen=max_len)
y_train = np.memmap(DATA_DIR+'extracted_feats/train-targets-rnn',
            dtype='float32', mode='r', shape=(3014,))


# Load DEV data
x_dev = np.memmap(DATA_DIR+'extracted_feats/dev-feats-rnn',
            dtype='float32', mode='r', shape=(1710, max_len, 90))
# x_dev = sequence.pad_sequences(dev_fp, maxlen=max_len)
y_dev = np.memmap(DATA_DIR+'extracted_feats/dev-targets-rnn',
            dtype='float32', mode='r', shape=(1710,))

# Load EVAL data
evalA_fp = np.memmap(DATA_DIR+'extracted_feats/eval-feats-A-rnn-2',
            dtype='float32', mode='r', shape=(6653, max_len, 90))
evalB_fp = np.memmap(DATA_DIR+'extracted_feats/eval-feats-B-rnn-2',
            dtype='float32', mode='r', shape=(6653, max_len, 90))

print("Done!")

## LOAD MODEL
if LOAD:
    model = load_model('./models/lstm-L14_0.0001/lstm-L14_0.0001_25-0.24.hdf5')
    print("Loaded model {} from disk".format(MODEL_NUM))

## CREATE MODEL
else:
    model = Sequential()
    if LAYERS == 1:
        model.add(LSTM(HIDDEN_UNITS, dropout=DROP))
    elif LAYERS == 3:
        model.add(LSTM(HIDDEN_UNITS, return_sequences=True, input_shape=(max_len, 90), dropout=DROP))
        model.add(LSTM(HIDDEN_UNITS, return_sequences=True, dropout=DROP))
        model.add(LSTM(HIDDEN_UNITS, dropout=DROP))
    model.add(Dense(1, activation='sigmoid'))
    optim = keras.optimizers.Adam(lr=LEARNING_RATE, amsgrad=True)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

## DEFINE CALLBACKS
def lr_decay(epoch):
    initial_lr = LEARNING_RATE
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr
lrate = LearningRateScheduler(lr_decay)
########
csv_logger = CSVLogger('./train_logs/lstm-{}.csv'.format(MODEL_NUM), append=True)
########
class WriteDevScores(Callback):
    def on_epoch_end(self, epoch, logs={}):
        print("Writing dev scores for model {}...".format(MODEL_NUM))
        scores = model.predict(x_dev)
        with open("./scores/lstm-{}_ep{}_dev_scores.csv".format(MODEL_NUM, epoch), 'w') as f:
            for s in scores:
                f.write(str(float(s)) + '\n')
dev_writer = WriteDevScores()
########
class WriteEvalScores(Callback):
	def on_epoch_end(self, epoch, logs={}):
        print("Writing eval scores...")
        scores_A = model.predict(evalA_fp)
        scores_B = model.predict(evalB_fp)
        with open("./scores/lstm-{}_eval_scores.csv".format(MODEL_NUM), 'w') as f:
            for s in scores_A:
                f.write(str(float(s)) + '\n')
            for s in scores_B:
                f.write(str(float(s)) + '\n')
eval_writer = WriteEvalScores()
########
checkpoint = ModelCheckpoint(DATA_DIR+'/lstm-{}_{{epoch:02d}}-{{val_loss:.2f}}.hdf5'.format(MODEL_NUM))
########
if SAVE:
    callbacks = [lrate, csv_logger, dev_writer, eval_writer, checkpoint]
else:
    callbacks = [lrate, csv_logger, dev_writer, eval_writer]

## TRAIN MODEL
print("Training model {}...".format(MODEL_NUM))
history = model.fit(x_train, y_train, validation_data=(x_dev, y_dev), epochs=NUM_EPOCHS,
          callbacks=callbacks, batch_size=20, verbose=2)

## WRITE EVAL SCORES
# print("Writing eval scores...")
# scores_A = model.predict(evalA_fp)
# scores_B = model.predict(evalB_fp)
# with open("./scores/lstm-{}_eval_scores.csv".format(MODEL_NUM), 'w') as f:
#     for s in scores_A:
#         f.write(str(float(s)) + '\n')
#     for s in scores_B:
#         f.write(str(float(s)) + '\n')
# print("Done!")

print("Finished training model lstm-{}".format(MODEL_NUM))
