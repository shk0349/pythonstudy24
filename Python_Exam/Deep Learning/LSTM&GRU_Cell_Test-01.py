import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.version_utils import callbacks

tf.keras.utils.set_random_seed(42)

(train_input, train_target), (test_input, test_target) = keras.datasets.imdb.load_data(num_words = 500)

train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size = 0.2)

train_seq = keras.preprocessing.sequence.pad_sequences(train_input, maxlen = 100)
val_seq = keras.preprocessing.sequence.pad_sequences(val_input, maxlen = 100)
test_seq = keras.preprocessing.sequence.pad_sequences(test_input, maxlen = 100)
rmsprop = keras.optimizers.RMSprop(learning_rate = 1e-4)

# LSTM 1개 적용
model1 = keras.Sequential()
model1.add(keras.layers.Embedding(500, 16, input_shape = (100, )))
model1.add(keras.layers.LSTM(8))
model1.add(keras.layers.Dense(1, activation = 'sigmoid'))
model1.summary()
print("=" * 50)

model1.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-1-model.keras', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 5, restore_best_weights = True)
history1 = model1.fit(train_seq, train_target, epochs = 100, batch_size = 64, validation_data = (val_seq, val_target), callbacks = [checkpoint_cb, early_stopping_cb])

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
plt.show()

# LSTM 1개 + Dropout 적용
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_shape = (100, )))
model2.add(keras.layers.LSTM(8, dropout = 0.3))
model2.add(keras.layers.Dense(1, activation = 'sigmoid'))
model2.summary()
print("=" * 50)

model2.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-1-dropout-model.keras', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 5, restore_best_weights = True)
history2 = model2.fit(train_seq, train_target, epochs = 100, batch_size = 64, validation_data = (val_seq, val_target), callbacks = [checkpoint_cb, early_stopping_cb])

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
plt.show()

# LSTM 2개 + Dropout 적용
model3 = keras.Sequential()
model3.add(keras.layers.Embedding(500, 16, input_shape = (100, )))
model3.add(keras.layers.LSTM(8, dropout = 0.3, return_sequences = True))
model3.add(keras.layers.LSTM(8, dropout = 0.3))
model3.add(keras.layers.Dense(1, activation = 'sigmoid'))
model3.summary()
print("=" * 50)

model3.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-2-dropout-model.keras', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 5, restore_best_weights = True)
history3 = model3.fit(train_seq, train_target, epochs = 100, batch_size = 64, validation_data = (val_seq, val_target), callbacks = [checkpoint_cb, early_stopping_cb])

plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.plot(history3.history['accuracy'])
plt.plot(history3.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
plt.show()

# GRU 적용
model4 = keras.Sequential()
model4.add(keras.layers.Embedding(500, 16, input_shape = (100, )))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1, activation = 'sigmoid'))
model4.summary()
print("=" * 50)

model4.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-gru-1-model.keras', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 5, restore_best_weights = True)
history4 = model4.fit(train_seq, train_target, epochs = 100, batch_size = 64, validation_data = (val_seq, val_target), callbacks = [checkpoint_cb, early_stopping_cb])

plt.plot(history4.history['loss'])
plt.plot(history4.history['val_loss'])
plt.plot(history4.history['accuracy'])
plt.plot(history4.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
plt.show()

# GRU Data 적용 Test
gru_model = keras.models.load_model('best-gru-1-model.keras')
gru_model.evaluate(test_seq, test_target)