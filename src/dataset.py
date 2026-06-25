# Importing libraries
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# Loading Data for preparation 

def load_and_prepare_data(train_path, test_path, rul_path, sequence_length=30): 
    # Standard names of columns in Nasa DATA
    columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    # 1. Here we are loading data
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=columns)
    
    # coputing the RUL for train data
    # We are obtaining the max cycle of the engine
    max_cycle = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index() 
    max_cycle.columns = ['unit_number', 'max_cycle']
    # we are merging the max cycle with our main data 
    train_df = train_df.merge(max_cycle, on='unit_number', how='left') # Merging max cycle with unit number
    # We have applied formula for getting the Remaining Useful Life(RUL) of engine 
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
    # Limiting the RUL cycles to 125 if are grater than it
    train_df['RUL'] = train_df['RUL'].clip(upper=125)  
    train_df.drop(columns=['max_cycle'], inplace=True) # Dropping the max cycle column
    
    # Loading data and RUL
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=columns)
    rul_df = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['true_RUL'])
    rul_df['unit_number'] = rul_df.index + 1
    
    # Here we are computing RUL for each row in test data 
    test_max_cycle = test_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    test_max_cycle.columns = ['unit_number', 'max_cycle']
    # Merging the max cycle to test data 
    test_df = test_df.merge(test_max_cycle, on='unit_number', how='left')
    test_df = test_df.merge(rul_df, on='unit_number', how='left')
    # Finding Remaining Useful Life(RUL) in test data
    test_df['RUL'] = test_df['true_RUL'] + (test_df['max_cycle'] - test_df['time_in_cycles'])
    test_df['RUL'] = test_df['RUL'].clip(upper=125)
    test_df.drop(columns=['max_cycle', 'true_RUL'], inplace=True)
    
    # Here we are doing feature scaling in scikit-learn 
    feature_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    scaler = MinMaxScaler()  # We are using MinMaxScalar()
    # training and transforming the train_df columns
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    # We just transforming the test_df columns
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Generating the 3d temporal sequences
    X_train, y_train = create_sequences(train_df, sequence_length, feature_cols)
    X_test, y_test = create_sequences(test_df, sequence_length, feature_cols)
    
    return X_train, y_train, X_test, y_test, scaler, feature_cols



# Defing function for creating sequences
def create_sequences(df, sequence_length, feature_cols):
    X, y = [], [] # Empty lists
    for unit in df['unit_number'].unique(): # Looking for unique unique engine(unit) values 
        unit_df = df[df['unit_number'] == unit] # unit number == unit
        if len(unit_df) < sequence_length: # if any unit has <30 cycles skip
            continue
        data = unit_df[feature_cols].values #Input
        labels = unit_df['RUL'].values # Lables/given values
        # If any unit has ran for 100 cycles (it will - sequence(30) +1 ) 
        # it will make a window of 71
        for i in range(len(unit_df) - sequence_length + 1):
            X.append(data[i : i + sequence_length]) # Adding input from index to the last sequence length
            y.append(labels[i + sequence_length - 1]) # Adding labels
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32) # Changing dataype to float of arrays
# Creating class for CMAPS and giving Dataset as a parameter
class CMAPSSDataset(Dataset): 
    def __init__(self, X, y): # Working with our X and y
        # Converting the X and y in Pytorch tensors
        self.X = torch.tensor(X, dtype=torch.float32) 
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self): # Function to give length of X
        return len(self.X)
        
    def __getitem__(self, idx): # Function to get item index from X and y 
        return self.X[idx], self.y[idx] 