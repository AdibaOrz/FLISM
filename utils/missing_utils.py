import torch
import random
import numpy as np



def get_modality_indices_to_drop(args, num_total_modalities, verbose=True):
    if not args.simulated:
        num_modalities_to_drop = random.randint(num_total_modalities // 2, num_total_modalities - 1) # drop at least half of the modalities
        drop_modality_indices = np.random.choice(list(range(num_total_modalities)), size=num_modalities_to_drop, replace=False)
        drop_modality_indices = drop_modality_indices.tolist()
    else:
        num_modalities_to_drop = torch.randint(num_total_modalities // 2, num_total_modalities, (1,)).item()
        drop_modality_indices = torch.randperm(num_total_modalities)[:num_modalities_to_drop].tolist()

    if verbose:
        print(f"Drop {num_modalities_to_drop}: {drop_modality_indices}")
    return drop_modality_indices


def get_drop_list_for_all_users(args, unique_user_list):
    user_drop_dict = {}
    if not args.simulated:
        num_total_modalities = args.dataset_opts['num_modalities']
    else:
        num_total_modalities = args.simulated_number_of_modalities

    num_users_to_apply_drop = int(len(unique_user_list) * args.drop_p)
    selected_clients_for_drop = np.random.choice(unique_user_list, size=num_users_to_apply_drop, replace=False)

    print(f'for drop percentage of {args.drop_p *100}, '
          f'num_clients applied for drop: {num_users_to_apply_drop}, selected clients: {selected_clients_for_drop}')
    for user_id in unique_user_list:
        if user_id in selected_clients_for_drop:
            mod_indices_drop_curr_user = get_modality_indices_to_drop(args, num_total_modalities)
            user_drop_dict[user_id] = mod_indices_drop_curr_user
        else:
            user_drop_dict[user_id] = []
    return user_drop_dict


def modality_idx_to_name(dataset_name, modality_idx):
    if dataset_name == "wesad":
        mod_idx_name_dict = {
            0: 'chestAcc',
            1: 'chestECG',
            2: 'chestEMG',
            3: 'chestEDA',
            4: 'chestTemp',
            5: 'chestResp',
            6: 'wristAcc',
            7: 'wristBVP',
            8: 'wristEDA',
            9: 'wristTemp'
        }
    elif dataset_name == "pamap2":
        mod_idx_name_dict = {
            0: 'hand_acc',
            1: 'hand_gyro',
            2: 'chest_acc',
            3: 'chest_gyro',
            4: 'ankle_acc',
            5: 'ankle_gyro'
        }
    elif dataset_name == "realworldhar":
        mod_idx_name_dict = {
            0: 'chest_acc',
            1: 'head_acc',
            2: 'shin_acc',
            3: 'upperarm_acc',
            4: 'waist_acc',
            5: 'chest_gyro',
            6: 'head_gyro',
            7: 'shin_gyro',
            8: 'upperarm_gyro',
            9: 'waist_gyro'
        }
    elif dataset_name == "sleepedf20":
        mod_idx_name_dict = {
            0: 'EEG Fpz-Cz',
            1: 'EEG Pz-Oz',
            2: 'EOG horizontal',
            3: 'Resp oro-nasal',
            4: 'EMG submental',
        }
    else:
        raise NotImplementedError(f"dataset_name {dataset_name} not implemented yet")
    return mod_idx_name_dict[modality_idx]



def drop_modalities(dataset_name, input_data, modality_indices_to_drop):
    if dataset_name == "realworldhar": # contains both acc and gyro
        dropped_input = drop_modalities_for_realworldhar(input_data, modality_indices_to_drop)
    elif dataset_name == "pamap2": # contains both acc and gyro
        dropped_input = drop_modalities_for_pamap2(input_data, modality_indices_to_drop)
    elif dataset_name == "wesad": # contains only acc
        dropped_input = drop_modalities_for_wesad(input_data, modality_indices_to_drop)
    elif dataset_name == "sleepedf20": # contains only acc
        dropped_input = drop_modalities_for_sleepedf20(input_data, modality_indices_to_drop)
    else:
        raise ValueError(f"Dataset name {dataset_name} not supported")
    return dropped_input


def drop_modalities_for_realworldhar(input_data, modality_indices_to_drop):
    for idx in modality_indices_to_drop:
        input_data[:, 3 * idx: 3 * (idx + 1), :] = 0
    return input_data


def drop_modalities_for_pamap2(input_data, modality_indices_to_drop):
    for idx in modality_indices_to_drop:
        input_data[:,3 * idx : 3 * (idx + 1), :] = 0
    return input_data


def drop_modalities_for_wesad(input_data, modality_indices_to_drop):
    for idx in modality_indices_to_drop:
            input_data[:,idx, :] = 0
    return input_data


def drop_modalities_for_sleepedf20(input_data, modality_indices_to_drop):
    for idx in modality_indices_to_drop:
        input_data[:,idx, :] = 0
    return input_data
