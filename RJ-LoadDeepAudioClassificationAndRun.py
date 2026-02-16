import csv
import os
import random
from matplotlib import pyplot as plt
import tensorflow as tf
import librosa

def load_model(model_name):
    model=tf.keras.models.load_model(model_name)
    return model

def load_mp3_16k_as_wav_mono(filename):
    y, sr = librosa.load(filename, sr=16000, mono=True)
    tensor = tf.convert_to_tensor(y, dtype=tf.float32)
    sample_rate = sr
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav=librosa.resample(y, orig_sr=sample_rate, target_sr=16000)
    return wav

def preprocess_wav_from_mp3(sample, index):
    wav = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32, fft_length=512)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2) # for our neural network, which is designed for images, so 3 dimensions
    return spectrogram


def convert_longer_clip_to_audio_slices(wav):
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_wav_from_mp3)
    audio_slices = audio_slices.batch(64)
    return audio_slices


def slice_forest_recordings(filename):
    mp3=os.path.join('data','Forest Recordings', filename)
    wav=load_mp3_16k_as_wav_mono(mp3)
    audio_slices = convert_longer_clip_to_audio_slices(wav)
    # samples = audio_slices
    # print("Len of Audio_Slices:", len(audio_slices))
    # print("Samples:",audio_slices)
    return audio_slices


def predict_slices(model, audio_slices, tolerance):
    yhat=model.predict(audio_slices)
    yhat = [1 if prediction > tolerance else 0 for prediction in yhat]
    return yhat
    # print(yhat)

def predict_file(model, filename, tolerance=0.99):
    audio_slices = slice_forest_recordings(filename)
    prediction = predict_slices(model, audio_slices, tolerance)
    # print('Lenght:', len(prediction))
    # print('Prediction:', prediction)

    from itertools import groupby
    yhat = [key for key, group in groupby(prediction)]
    calls = tf.math.reduce_sum(yhat).numpy()
    print(f"Predicted Capuchin Calls: {calls}")
    return calls

def predict_files(model, tolerance=0.99):
    all_predictions = {}
    for file in os.listdir(os.path.join('data', 'Forest Recordings')):
        print(f"Processing file: {file}")
        calls =predict_file(model, file, tolerance)
        all_predictions[file] = calls
        # print('---')
        # print(f'File: {file} - Predicted Capuchin Calls: {calls}')
        # print()

    all_predictions = dict(sorted(all_predictions.items()))
    print("All Predictions:")
    with open('capuchin_birds.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Recording', 'Predicted Capuchin Calls'])

        for file, calls in all_predictions.items():
            print(f"{file}: {calls} calls")
            writer.writerow([file, calls])
    
