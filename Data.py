# Retrieves data from nfdump fil

# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
import sqlite3

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import os 
from pandas import read_csv

#Replacing textfields in dataset with unique numbers
def Dataprocessing(data):
    #Replace textfield protocol with numbers
    column_values = data[["pr"]].values.ravel()
    unique_values =  pd.unique(column_values)

    for idx,i in enumerate(unique_values):
        data.pr[data.pr == i] = idx + 1

    #Replace flg with numbers
    column_values = data[["flg"]].values.ravel()
    unique_values =  pd.unique(column_values)

    for idx,i in enumerate(unique_values):
        data.flg[data.flg == i] = idx + 100
    return data

#Get data from firewall dataset
fw_zone = read_csv('fw_zone.csv', delimiter=';')

#Replace flg with unique numbers
column_values = fw_zone[["base_interface"]].values.ravel()
unique_values =  pd.unique(column_values)

for idx,i in enumerate(unique_values):
    fw_zone.base_interface[fw_zone.base_interface == i] = idx


#Get data from VM dataset
vm_data = read_csv('vm_data.csv', delimiter=';')
vm_data = vm_data.drop(columns=['vm_name','vm_network','vm_os'])

#Read data
csv = read_csv('Data25jan.csv', delimiter=',', low_memory=False)

#Selecting all or a fraction of the data
samples = csv.sample(frac =.01)
#samples = csv

#select relevant data from nfdump file
data = samples.iloc[:,[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]

data = Dataprocessing(data)

#Make the db in memory
conn = sqlite3.connect(':memory:')

#write the tables
vm_data.to_sql('VM', conn, index=False)
fw_zone.to_sql('FW', conn, index=False)
data.to_sql('Data', conn, index=False)

#SQL
comb = 'select vm.src_ip as src_ip_aton, vm.id as src_vm_id, vm.vm_vlan as src_vlan, vm.os_id as src_os_id, dst_vm.src_ip as dst_ip_aton, dst_vm.id as dst_vm_id, dst_vm.vm_vlan as dst_vlan, dst_vm.os_id as dst_os_id, data.sp as src_port, data.dp as dst_port, data.pr as protocol, data.flg, data.fwd, data.stos, data.ipkt, data.ibyt, data.opkt, fw.base_interface FROM VM as vm JOIN Data as data ON vm.sa = data.sa JOIN VM as dst_VM ON dst_VM.sa = data.da JOIN FW as fw ON vm.src_ip BETWEEN fw.aton_start AND fw.aton_stop AND dst_port < 1024'
df = pd.read_sql_query(comb, conn)

#write data to file              
df.to_csv('dataset_25jan_fraction.csv',index=False)


print("END")