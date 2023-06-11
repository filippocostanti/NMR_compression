from variables import *
from models import *
import numpy as np
import pandas as pd
import tensorflow as tf

############################## DATASET IMPORTATION #########################################
#dataset = pd.read_csv('dataset_downsample_spatial.csv').values
folder = 'filtered_spectra'
n_ppm = 262144
dataset = np.ones((len(string_name), n_ppm))
i = 0
for name in string_name:
    spectrum = pd.read_csv(folder + "/filtered_" + name + ".txt")
    dataset[i, :] = spectrum['assey'].values
    i += 1
for i in range(dataset.shape[0]):
    dataset[i][:] = dataset[i][:] / np.max(dataset[i])

############################## ZERO FILTRATION #########################################
embed_size = 1408  # 704, 1408, 2816, 14080
layers = 3
#for embed_size in np.arange(100, 1600, 100):
non_zero_indices = np.argwhere(dataset)
first_nonzero_index = np.min(non_zero_indices[:, 1])
last_nonzero_index = np.max(non_zero_indices[:, 1])
dataset = dataset[:, first_nonzero_index:last_nonzero_index + (n_ppm % 1300)]
dataset = dataset[:, :220000]
n_ppm = dataset.shape[1]
# dataset = dataset[:, :262000]

############################## VARIABLES DECLARATION #########################################
#passo = n_ppm // embed_size
step_decomposition = 5000
aug_rows_per_spectrum = n_ppm//step_decomposition
neurons = (step_decomposition, 1024, int(embed_size/aug_rows_per_spectrum))
lrate = 0.0001


############################## TRAIN-TEST SPLIT #########################################
train_perc = 0.8
fin = int(dataset.shape[0] * train_perc)
idxg = np.arange(dataset.shape[0])
np.random.shuffle(idxg)
train_index = idxg[:fin]
test_index = idxg[fin:]

dataset_train = dataset[train_index]
dataset_test = dataset[test_index]
augmented_dataset = np.zeros((dataset.shape[0] * aug_rows_per_spectrum, step_decomposition))
augmented_dataset_train = np.ones((dataset_train.shape[0] * aug_rows_per_spectrum, step_decomposition))
augmented_dataset_test = np.zeros((dataset_test.shape[0] * aug_rows_per_spectrum, step_decomposition))
rif = np.arange(0, n_ppm, step_decomposition)

############################## DATASET AUGMENTATION #########################################
k = 0
for i in range(dataset_train.shape[0]):
    for j in range(rif.shape[0]):
        if j == rif.shape[0] - 1:
            augmented_dataset_train[k] = np.concatenate((dataset_train[i, rif[j]:],
                                                        np.zeros(step_decomposition-dataset_train[i, rif[j]:].shape[0])))
            k += 1
        else:
            augmented_dataset_train[k] = dataset_train[i, rif[j]:rif[j + 1]]
            k += 1
k = 0
for i in range(dataset_test.shape[0]):
    for j in range(rif.shape[0]):
        if j == rif.shape[0] - 1:
            augmented_dataset_test[k] = np.concatenate((dataset_test[i, rif[j]:],
                                                        np.zeros(step_decomposition-dataset_test[i, rif[j]:].shape[0])))
            k += 1
        else:
            augmented_dataset_test[k] = dataset_test[i, rif[j]:rif[j + 1]]
            k += 1
k = 0
for i in range(dataset.shape[0]):
    for j in range(rif.shape[0]):
        if j == rif.shape[0] - 1:
            augmented_dataset[k] = np.concatenate((dataset[i, rif[j]:],
                                                   np.zeros(step_decomposition-dataset[i, rif[j]:].shape[0])))
            k += 1
        else:
            augmented_dataset[k] = dataset[i, rif[j]:rif[j + 1]]
            k += 1

############################## AUTOENCODER #########################################
'''model = Autoencoder(neurons)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lrate))
model.fit(augmented_dataset_train, augmented_dataset_train,
          epochs=100,
          shuffle=True,
          validation_data=(augmented_dataset_test, augmented_dataset_test))
'''
augmented_dataset_train2 = np.resize(augmented_dataset_train, (augmented_dataset_train.shape[0], step_decomposition, 1))
augmented_dataset_test2 = np.resize(augmented_dataset_test, (augmented_dataset_test.shape[0], step_decomposition, 1))
kernel = 7
#grid = np.arange(6, 10, 1)
#for kernel in grid:
filters = 8
layers = 4
if layers == 3:
    embed = 1760
    if filters == 4:
        fil = 16
    elif filters == 8:
        fil = 32
elif layers == 4:
    embed = 352
    if filters == 4:
        fil = 32
    elif filters == 8:
        fil = 64
elif layers == 5:
    embed = 176
    if filters == 4:
        fil = 64
    elif filters == 8:
        fil = 128

model = ConvolutionalAutoencoder(fil, int(kernel), (step_decomposition, 1))
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lrate))
model.fit(augmented_dataset_train2, augmented_dataset_train2,
          epochs=100,
          shuffle=True)#,
          #validation_data=(augmented_dataset_test2, augmented_dataset_test2))

loss_error = model.evaluate(augmented_dataset_test2)
#loss_error = model.evaluate(augmented_dataset_test)
print('----------------------------------\n'
      + str(loss_error) +
      '\n----------------------------------\n')

dim_reduction = model.get_layer('Encoder')


############################## NEW DATASET RECONSTRUCTION #########################################
reconstructed_dataset = np.zeros((dataset.shape[0], neurons[1] * aug_rows_per_spectrum))
rec_range = np.arange(0, augmented_dataset.shape[0] + aug_rows_per_spectrum, aug_rows_per_spectrum)
#for i in range(dataset.shape[0]):
    #redu = np.zeros((aug_rows_per_spectrum, len(rec_range)))


'''reconstructed_dataset = np.zeros((dataset.shape[0], 44 * neurons[2]))
rec_range = np.arange(0, augmented_dataset.shape[0] + aug_rows_per_spectrum, aug_rows_per_spectrum)
k = 0
for j in range(len(rec_range)):
    if j != 0:
        reconstructed_dataset[k] = np.concatenate((dim_reduction.predict(augmented_dataset[rec_range[j-1]:rec_range[j], :])), axis=None)
        k += 1'''


reconstructed_dataset = np.zeros((dataset.shape[0], embed, filters)) # 176, 352, 1760
rec_range = np.arange(0, augmented_dataset.shape[0] + aug_rows_per_spectrum, aug_rows_per_spectrum)
k = 0
for j in range(len(rec_range)):
    if j != 0:
        for l in range(reconstructed_dataset.shape[2]):
            reconstructed_dataset[k, :, l] = np.concatenate((dim_reduction.predict(augmented_dataset[rec_range[j-1]:rec_range[j], :])[:, :, l]), axis=None)
        k += 1


name = 'dt_com_' + str(filters) + 'F_' + str(layers) + 'L_' + str(kernel) + 'K'
#name = 'mlp_' + str(embed_size) + 'E_' + str(layers) + 'L'
np.save('datasets2/' + name + '.npy', reconstructed_dataset)
model.save('models2/' + name)
#del model
print('##########################################################################################################\n'
      '##########################################################################################################\n'
      '##########################################################################################################\n')
