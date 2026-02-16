import os
import random
from matplotlib import pyplot as plt
import tensorflow as tf
import librosa


CAPUCHIN_FOLDER = 'Parsed_Capuchinbird_Clips'
NOT_CAPUSHIN_FOLDER = 'Parsed_Not_Capuchinbird_Clips'

CAPUCHIN_FILE = os.path.join('data', CAPUCHIN_FOLDER, 'XC3776-3.wav')
NOT_CAPUCHIN_FILE = os.path.join('data', NOT_CAPUSHIN_FOLDER, 'afternoon-birds-song-in-forest-0.wav')

def load_wav_16k_mono(filename):
    # load wav file
    file_contents = tf.io.read_file(filename)
    
    #decode by channels - assume mono, if stereo, take only one channel
    wav, sample_rate=tf.audio.decode_wav(file_contents, desired_channels=1)
    
    # remove trailing axis
    wav=tf.squeeze (wav, axis=-1)
    sample_rate=tf.cast(sample_rate, dtype=tf.int64)

    # Prefer librosa for reliable resampling on macOS
    import librosa
    
    # librosa.load returns numpy array at desired sample rate
    y, sr = librosa.load(filename, sr=16000, mono=True)
    return tf.convert_to_tensor(y, dtype=tf.float32)

    desired_rate = 16000
    if tf.equal(sample_rate, desired_rate):
        return wav

wave=load_wav_16k_mono(CAPUCHIN_FILE)
nwave=load_wav_16k_mono(NOT_CAPUCHIN_FILE)

# plt.plot(wave)
# plt.plot(nwave)
# plt.title('Capuchin vs Not#  Capuchin')
# plt.show()

POS=os.path.join('data', CAPUCHIN_FOLDER)
NEG = os.path.join('data', NOT_CAPUSHIN_FOLDER)

pos_files = sorted(tf.io.gfile.glob(os.path.join('data', CAPUCHIN_FOLDER, '*.wav')))
neg_files = sorted(tf.io.gfile.glob(os.path.join('data', NOT_CAPUSHIN_FOLDER, '*.wav')))

# Build a generator that yields precomputed spectrograms and labels (numpy arrays).
# This avoids passing tf.Tensor file paths into Python-only audio libraries.
all_files = pos_files + neg_files
all_labels = [1.0] * len(pos_files) + [0.0] * len(neg_files)

lengths = []
for file in pos_files:
    tensor_wave = load_wav_16k_mono(file)
    lengths.append(len(tensor_wave))

mean = tf.math.reduce_mean(lengths)
min = tf.math.reduce_min(lengths)
max = tf.math.reduce_max(lengths)

print(mean, min, max)


def _generator():
    import numpy as _np
    import librosa as _librosa

    target_len = 48000
    for path, lbl in zip(all_files, all_labels):
        # ensure str path
        if isinstance(path, bytes):
            path = path.decode('utf-8')

        # load and resample with librosa (numpy array)
        y, _ = _librosa.load(path, sr=16000, mono=True)

        if y.shape[0] > target_len:
            y = y[:target_len]
        elif y.shape[0] < target_len:
            y = _np.pad(y, (0, target_len - y.shape[0]), mode='constant')

        # compute STFT with TensorFlow to match original frame/window handling
        import tensorflow as _tf

        wav_tf = _tf.convert_to_tensor(y, dtype=_tf.float32)
        spec_tf = _tf.signal.stft(wav_tf, frame_length=320, frame_step=32, fft_length=512)
        spec_tf = _tf.abs(spec_tf)
        spec = spec_tf.numpy().astype(_np.float32)  # (time, freq) == (1491, 257)
        spec = _np.expand_dims(spec, axis=2)
        yield spec, _np.array(lbl, dtype=_np.float32)


# test_wave=load_wav_16k_mono(CAPUCHIN_FILE)
# spec = np.abs(librosa.stft(test_wave.numpy(), n_fft=320, hop_length=32))
# spec.shape()

def preprocess(file, label):

    file_path=""
    if tf.is_symbolic_tensor(file):
        # print("Is Symbolic Tensor")
        # symbolic_tensor produced by layers applied to `inp`
        model = tf.keras.Model(inputs=file)
  
        file_path = model(tf.convert_to_tensor(file))[0].numpy()
    # else:
    #     print("Not Symbolic Tensor")
    
    #     file_path = file[0]
    
    file_path = file

    # print('*****************************************************')
    # print(file_path, label)
    # print('*****************************************************')   
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000] # Truncate to 3 seconds (48,000 samples at 16kHz)
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32, fft_length=512)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2) # for our neural network, which is designed for images, so 3 dimensions
    return spectrogram, label

def test_random_file(pos=True):
    if pos: 
        dataset = pos_files
    else:
        dataset = neg_files

    import random
    filepath, label = random.choice(list(zip(dataset, all_labels)))
    print("file: ", filepath, "label: ", label )
    spec, label=preprocess(filepath, label)

    plt.figure(figsize=(10,10))
    plt.imshow(tf.transpose(spec)[0])
    plt.show()

   
# test_random_file(True)

# Build two independent datasets (train and validation) from pre-split file lists.
# This avoids sharing a single repeated generator and prevents dataset exhaustion.
total_files_and_lables =  list(zip(all_files, all_labels))

import random
random.shuffle(total_files_and_lables)

numb_files = len(total_files_and_lables)
numb_training = int(numb_files * 0.7)

train_files = total_files_and_lables[:numb_training]
test_files = total_files_and_lables[numb_training:]

# for f, l in train_files:
#     print(f, l)

# exit

batch_size = 16

import math
import builtins as _builtins

def make_generator(files_and_labels):
    def gen():
        import numpy as _np
        import librosa as _librosa
        import tensorflow as _tf

        target_len = 48000
        for path, lbl in files_and_labels:
            if isinstance(path, bytes):
                path = path.decode('utf-8')

            y, _ = _librosa.load(path, sr=16000, mono=True)

            if y.shape[0] > target_len:
                y = y[:target_len]
            elif y.shape[0] < target_len:
                y = _np.pad(y, (0, target_len - y.shape[0]), mode='constant')

            wav_tf = _tf.convert_to_tensor(y, dtype=_tf.float32)
            spec_tf = _tf.signal.stft(wav_tf, frame_length=320, frame_step=32, fft_length=512)
            spec_tf = _tf.abs(spec_tf)
            spec = spec_tf.numpy().astype(_np.float32)
            spec = _np.expand_dims(spec, axis=2)
            yield spec, _np.array(lbl, dtype=_np.float32)

    return gen

# import pandas as pd
# pd.DataFrame(train_files, columns=['file', 'label']).to_csv('train_files.csv', index=False)
# pd.DataFrame(test_files, columns=['file', 'label']).to_csv('test_files.csv', index=False)
# exit()

# Train dataset: cache, shuffle, batch, repeat, prefetch
train = tf.data.Dataset.from_generator(
    make_generator(train_files),
    output_types=(tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([None, None, 1]), tf.TensorShape([])),
)

train = train.cache()
train = train.batch(batch_size)
train = train.repeat()
train = train.prefetch(8)

# Validation dataset: no shuffle, batch, repeat and prefetch
test = tf.data.Dataset.from_generator(
    make_generator(test_files),
    output_types=(tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([None, None, 1]), tf.TensorShape([])),
)
test=test.cache()
test = test.batch(batch_size)
test = test.repeat()
test = test.prefetch(8)

# import pandas as pd
# pd.DataFrame(train, columns=['file', 'label']).to_csv('train_files.csv', index=False)
# pd.DataFrame(test, columns=['file', 'label']).to_csv('test_files.csv', index=False)
# exit()

# compute steps from file counts and batch size
steps_per_epoch = _builtins.max(1, math.ceil(len(train_files) / batch_size))
validation_steps = _builtins.max(1, math.ceil(len(test_files) / batch_size))

# grab a single batch for inspection
samples, labels = train.as_numpy_iterator().next()
print('Samples.shape:', samples.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

def train_model(train, test, save=False):
    model=Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(1491, 257, 1)))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'  ))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    model.summary()

    hist = model.fit(
        train,
        epochs=4,
        validation_data=test,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    if save:
        model.save('capuchin_classifier.h5')
    return model, hist

def load_model(model_name):
    model=tf.keras.models.load_model(model_name)
    return model

def test_results(n):
    model = load_model('capuchin_classifier.h5')
    for i in range(n):
        x_test, y_test = test.as_numpy_iterator().next()
    yhat = model.predict(x_test)
    yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
    print(yhat)
    print(y_test.astype(int))

def plot_loss():
    plt.title('Loss')
    plt.plot(hist.history['loss'], 'r')
    plt.plot(hist.history['val_loss'], 'b')
    plt.show()

def plot_precision():
    plt.title('Precision')
    plt.plot(hist.history['precision'], 'r')
    plt.plot(hist.history['val_precision'], 'b')
    plt.show()

def plot_recall():
    plt.title('Recall')
    plt.plot(hist.history['recall'], 'r')
    plt.plot(hist.history['val_recall'], 'b')
    plt.show()