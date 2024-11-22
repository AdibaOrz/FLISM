"""
Dataset-specific metadata and hyperparameters
"""

PAMAP2_Opt = {
    # 1. Metadata
    'name' : 'pamap2',
    'num_users': 8,
    'num_classes': 12,
    'input_dim': 18,
    'num_modalities': 6,
    'dim_lists': [[j + 3 * i for j in range(3)] for i in range(6)],
    'sampling_rate': 100,
    'window_len': 2 * 100, # 2sec

    # 2. Training hyperparams (optimized for FedAvg)
    "optim": "sgd",
    "local_ep": 5,
    "local_bs": 32,
    "local_lr": 0.01,
    "local_wd": 0.001,
    "global_rnd": 100,
    "client_selection_rate": 0.5, # 4 clients

    # 3. Data columns
    "feature_columns": [
                'hand_acc_x', 'hand_acc_y', 'hand_acc_z', 'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
                'chest_acc_x', 'chest_acc_y', 'chest_acc_z', 'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
                'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z', 'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z'],
    "all_columns_ordered": [
                'timestamp', 'label',
                'hand_acc_x', 'hand_acc_y', 'hand_acc_z', 'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
                'chest_acc_x', 'chest_acc_y', 'chest_acc_z', 'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
                'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z', 'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
                'user_idx'],

    # 4. AutoFed
    "dim_per_modalities" : {i : 3 for i in range(6)},
}


RealWorldHAR_Opt = {
    # 1. Metadata
    'name' : 'realworldhar',
    'num_users': 15,
    'num_classes': 8,
    'input_dim': 30,
    'num_modalities': 10,
    'dim_lists': [[j + 3 * i for j in range(3)] for i in range(10)],
    'sampling_rate': 50,
    'window_len': 3 * 50, # 3 seconds

    # 2. Training hyperparams (optimized for FedAvg)
    "optim": "sgd",
    "local_ep": 5,
    "local_bs": 32,
    "local_lr": 0.01,
    "local_wd": 0.001,
    "global_rnd": 100,
    "client_selection_rate": 0.3,  # 4 clients

    # 3. Data columns
    "feature_columns": [
            'chest_acc_x', 'chest_acc_y', 'chest_acc_z', 'head_acc_x', 'head_acc_y', 'head_acc_z',
            'shin_acc_x', 'shin_acc_y', 'shin_acc_z', 'upperarm_acc_x', 'upperarm_acc_y', 'upperarm_acc_z',
            'waist_acc_x', 'waist_acc_y', 'waist_acc_z', 'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
            'head_gyro_x', 'head_gyro_y', 'head_gyro_z', 'shin_gyro_x', 'shin_gyro_y', 'shin_gyro_z',
            'upperarm_gyro_x', 'upperarm_gyro_y', 'upperarm_gyro_z', 'waist_gyro_x', 'waist_gyro_y', 'waist_gyro_z'],
    "all_columns_ordered": [
            'index',
            'chest_acc_x', 'chest_acc_y', 'chest_acc_z', 'head_acc_x', 'head_acc_y', 'head_acc_z',
            'shin_acc_x', 'shin_acc_y', 'shin_acc_z', 'upperarm_acc_x', 'upperarm_acc_y', 'upperarm_acc_z',
            'waist_acc_x', 'waist_acc_y', 'waist_acc_z','chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
            'head_gyro_x', 'head_gyro_y', 'head_gyro_z', 'shin_gyro_x', 'shin_gyro_y', 'shin_gyro_z',
            'upperarm_gyro_x', 'upperarm_gyro_y', 'upperarm_gyro_z', 'waist_gyro_x', 'waist_gyro_y', 'waist_gyro_z',
            'label', 'user_idx']
}


SleepEDF20_Opt = {
    # 1. Metadata
    'name' : 'sleepedf20',
    'num_classes': 5,
    'input_dim': 5,
    'num_modalities': 5,
    'dim_lists': [[j + i for j in range(1)] for i in range(5)],
    'sampling_rate': 100,
    'window_len': 30 * 100, # 30 sec

    # 2. Training hyperparams (optimized for FedAvg)
    "optim": "sgd",
    "local_ep": 5,
    "local_bs": 32,
    "local_lr": 0.01,
    "local_wd": 0.001,
    "global_rnd": 100,
    "client_selection_rate": 0.3,  # 4 clients

    # 3. Data columns
    "feature_columns": ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental'],
    "all_columns_ordered": [
            'time', 'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental',
            'user_idx', 'label'],

    # 4. AutoFed
    "dim_per_modalities" : {i : 1 for i in range(5)},
}


WESAD_Opt = {
    # 1. Metadata
    'name': 'wesad',
    'num_users': 15,
    'num_classes': 3,
    'input_dim': 10,
    'num_modalities': 10,
    'dim_lists': [[j + i for j in range(1)] for i in range(10)],
    'sampling_rate': 4,
    'window_len': 10 * 4,  # 10 seconds

    # 2. Training hyperparams (optimized for FedAvg)
    "optim": "sgd",
    "local_ep": 5,
    "local_bs": 32,
    "local_lr": 0.01,
    "local_wd": 0.001,
    "global_rnd": 200,
    "client_selection_rate": 0.3,  # 4 clients

    # 3. Data columns
    "feature_columns": [
            'chest_acc', 'chest_ecg', 'chest_emg', 'chest_eda', 'chest_temp', 'chest_resp',
            'wrist_acc', 'wrist_bvp', 'wrist_eda', 'wrist_temp'],
    "all_columns_ordered": [
            'chest_acc', 'chest_ecg', 'chest_emg', 'chest_eda', 'chest_temp', 'chest_resp',
            'wrist_acc', 'wrist_bvp', 'wrist_eda', 'wrist_temp',
            'label', 'user_id']
}

Simulated_Opt = {
    # 1. Metadata
    'name' : 'simulated',
    'num_classes': 3,
}


DATASET_OPTS = {
    'pamap2': PAMAP2_Opt,
    'realworldhar': RealWorldHAR_Opt,
    'sleepedf20': SleepEDF20_Opt,
    'wesad': WESAD_Opt,
    'simulated': Simulated_Opt,
}

MOON_TRAINING_PARAMS = {
    # * Important * For MOON, adam works better than sgd, therefore use adam for all datasets.
    "wesad": {
        "optim": "adam",
        "local_ep": 5,
        "local_bs": 64,
        "local_lr": 0.001,
        "local_wd": 0.001,
        "global_rnd": 200,
        "client_selection_rate": 0.3,
    },
    "sleepedf20": {
        "optim": "adam",
        "local_ep": 5,
        "local_bs": 64,
        "local_lr": 0.001,
        "local_wd": 0.001,
        "global_rnd": 100,
        "client_selection_rate": 0.3,
    },
    "realworldhar": {
        "optim": "adam",
        "local_ep": 5,
        "local_bs": 64,
        "local_lr": 0.001,
        "local_wd": 0.001,
        "global_rnd": 100,
        "client_selection_rate": 0.3,
    },
    "pamap2": {
        "optim": "adam",
        "local_ep": 5,
        "local_bs": 64,
        "local_lr": 0.001,
        "local_wd": 0.001,
        "global_rnd": 100,
        "client_selection_rate": 0.5,
    }
}


def get_dataset_opts(dataset_name):
    """ Get the dataset options (metadata, training hyperparams, data columns) for the given dataset."""
    assert dataset_name in ['pamap2', 'realworldhar', 'sleepedf20', 'wesad', 'simulated']
    return DATASET_OPTS[dataset_name]

def get_moon_training_params(dataset_name):
    """ Get the training hyperparameters for the MOON model for the given dataset."""
    assert dataset_name in ['pamap2', 'realworldhar', 'sleepedf20', 'wesad']
    return MOON_TRAINING_PARAMS[dataset_name]
