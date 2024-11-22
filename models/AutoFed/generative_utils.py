def generate_model_dict(num_modalities):
    model_dict = {}
    index = 0
    for i in range(num_modalities):
        for j in range(num_modalities):
            if i != j:
                key = f'from_{i}_to_{j}'
                model_dict[index] = key
                index += 1
    return model_dict