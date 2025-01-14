import tensorflow as tf
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(42)

(train_input, train_target), (test_input, test_target) = keras.datasets.imdb.load_data(num_words = 300)

train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size = 0.2)

lengths = np.array([len(x) for x in train_input])
print("리뷰 길이의 평균, 최대, 최소, 중간값")
print(np.mean(lengths), np.max(lengths), np.min(lengths), np.median(lengths))
print("=" * 50)

plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

train_seq = keras.preprocessing.sequence.pad_sequences(train_input, maxlen = 100, truncating = 'pre')
val_seq = keras.preprocessing.sequence.pad_sequences(val_input, maxlen = 100)
lengthspd = np.array([len(x) for x in train_seq])
print("리뷰 길이의 평균, 최대, 최소, 중간값")
print(np.mean(lengthspd), np.max(lengthspd), np.min(lengthspd), np.median(lengthspd))
print("=" * 50)

plt.hist(lengthspd)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

model = keras.Sequential()
model.add(keras.layers.SimpleRNN(8, input_shape = (100, 300), activation = 'tanh'))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

model.summary()
print("=" * 50)

train_oh = keras.utils.to_categorical(train_seq)
val_oh = keras.utils.to_categorical(val_seq)
model.summary()

# RMSprop 적용
rmsprop = keras.optimizers.RMSprop(learning_rate = 1e-4)
model.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.keras', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 5, restore_best_weights = True)
history = model.fit(train_oh, train_target, epochs = 100, batch_size = 64, validation_data = (val_oh, val_target), callbacks = [checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss'])
plt.show()

model2 = keras.Sequential()
model2.add(keras.layers.Embedding(300, 16, input_shape = (100, )))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation = 'sigmoid'))
model2.summary()
print("=" * 50)

model2.compile(optimizer = rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history2 = model2.fit(train_seq, train_target, epochs=100, batch_size=64, validation_data=(val_seq, val_target),callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss'])
plt.show()

# Adam 적용
model3 = keras.Sequential()
model3.add(keras.layers.SimpleRNN(8, input_shape = (100, 300), activation = 'tanh'))
model3.add(keras.layers.Dense(1, activation = 'sigmoid'))
model3.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-adam-model.keras', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 5, restore_best_weights = True)
history3 = model3.fit(train_oh, train_target, epochs = 100, batch_size = 64, validation_data = (val_oh, val_target), callbacks = [checkpoint_cb, early_stopping_cb])

plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss'])
plt.show()

model4 = keras.Sequential()
model4.add(keras.layers.Embedding(300, 16, input_shape = (100, )))
model4.add(keras.layers.SimpleRNN(8))
model4.add(keras.layers.Dense(1, activation = 'sigmoid'))
model4.summary()
print("=" * 50)

model4.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-adam-model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history4 = model4.fit(train_seq, train_target, epochs=100, batch_size=64, validation_data=(val_seq, val_target),callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history4.history['loss'])
plt.plot(history4.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss'])
plt.show()

# Adagrad 적용
model5 = keras.Sequential()
model5.add(keras.layers.SimpleRNN(8, input_shape = (100, 300), activation = 'tanh'))
model5.add(keras.layers.Dense(1, activation = 'sigmoid'))
model5.compile(optimizer = 'adagrad', loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-adagrad-model.keras', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 5, restore_best_weights = True)
history5 = model5.fit(train_oh, train_target, epochs = 100, batch_size = 64, validation_data = (val_oh, val_target), callbacks = [checkpoint_cb, early_stopping_cb])

plt.plot(history5.history['loss'])
plt.plot(history5.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss'])
plt.show()

model6 = keras.Sequential()
model6.add(keras.layers.Embedding(300, 16, input_shape = (100, )))
model6.add(keras.layers.SimpleRNN(8))
model6.add(keras.layers.Dense(1, activation = 'sigmoid'))
model6.summary()
print("=" * 50)

model6.compile(optimizer = 'adagrad', loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-adagrad-model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history6 = model6.fit(train_seq, train_target, epochs=100, batch_size=64, validation_data=(val_seq, val_target),callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history6.history['loss'])
plt.plot(history6.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss'])
plt.show()