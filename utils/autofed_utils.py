import re

def extract_indices(s):
    pattern = r'from_(\d+)_to_(\d+)'
    match = re.match(pattern, s)
    if match:
        in_idx, out_idx = match.groups()
        return int(in_idx), int(out_idx)
    else:
        return None, None


def get_min_loss_for_dropped_indices(autoencoder_loss_dict, modality_indices_to_drop, modality_indices_not_to_drop):
    results = {}

    for drop_index in modality_indices_to_drop:
        # Filter the dict for keys corresponding to from_x_to_drop_index
        filtered_dict = {k: v for k, v in autoencoder_loss_dict.items() if
                         k.endswith(f'_to_{drop_index}') and int(k.split('_')[1]) in modality_indices_not_to_drop}

        # Find the key with the smallest value
        if filtered_dict:
            min_key = min(filtered_dict, key=filtered_dict.get)
            results[drop_index] = min_key

    return results

def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None