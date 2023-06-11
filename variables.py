import numpy as np
from setup import *



f1 = open(path_name, 'r')
string_name = f1.read()
f1.close()

string_name = string_name.split("\n")

f2 = open(path_vitality, 'r')
string_vitality = f2.read()
f2.close()

string_vitality = string_vitality.split("\n")
vitality = np.array(string_vitality)
vitality = vitality.astype(float)

target = np.zeros(len(vitality))
sm_target = np.zeros((len(vitality), 2))
count = 0
for i in range(len(vitality)):
    if vitality[i] >= 70:
        target[i] = 1
        sm_target[i, 0] = 1
    else:
        sm_target[i, 1] = 1

pos = np.sum(target)
neg = len(target) - pos

point_name = []
for ele in string_name:
    name_to_print = ele.replace('cellule', '')
    point_name.append(name_to_print)

old_experiments = ['cellulesane', 'cellulesane2', 'cellulesaneN', 'cellulesaneN2',
                    'celluleAb2uM24h_COLN', 'celluleAb2uM24h_COLN2', 'celluleAb2uM24h_EON',
                    'celluleAb2uM24h_EON2', 'celluleAb2uM24h_EONn', 'celluleAb2uM24h_EONn2',
                    'celluleAb2uM24h_EONnn', 'celluleAb2uM24h_EONnn2', 'celluleAb2uM24h_EONnnn',
                    'celluleAb2uM24h_EONnnn2', 'celluleAb2uM24h_EstrN', 'celluleAb2uM24h_EstrN2',
                    'celluleAb2uM24h_EstrNn', 'celluleAb2uM24h_EstrNn2', 'celluleAb2uM24h_COLNn',
                    'celluleAb2uM24h_COLNn2', 'celluleAb2uM24hN', 'celluleAb2uM24hN2',
                    'celluleAb5uM24h_Cu',
                    'celluleAb5uM24h_EO', 'celluleAb5uM24h_EO2', 'celluleAb5uM24h_EOCu',
                    'celluleAb5uM24h_Estr',
                    'celluleAb5uM24h_Estr2', 'celluleAb5uM24h_EstrCu', 'celluleAb5uM24h_RA',
                    'celluleAb5uM24h_RA2',
                    'celluleAb5uM24h_RACu', 'celluleAb5uM24h', 'celluleAb5uM24h2']

to_exclude = ['cellule24h_GAL0.5_1', 'cellule24h_GAL0.5_2', 'cellule24h_GAL0.5_3', 'cellule24h_GAL0.25_1',
           'cellule24h_GAL0.25_2', 'cellule24h_GAL0.25_3', 'cellule24h_GAL0.05_1', 'cellule24h_GAL0.05_2',
           'cellule24h_LYC0.008_4', 'cellule24h_LYC0.008_5', 'cellule72h_LYC0.008_1', 'cellule72h_LYC0.008_2',
           'cellule96h_LYC0.008_1', 'cellule96h_LYC0.008_2', 'cellule24h_LYC0.016_1', 'cellule24h_LYC0.016_2',
           'cellule24h_LYC0.016_3', 'cellule24h_LYC0.008_1', 'cellule24h_LYC0.008_2', 'cellule24h_LYC0.008_3']
