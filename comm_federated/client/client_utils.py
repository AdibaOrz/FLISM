def calculate_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size_bytes = total_params * 4  # Assuming 32-bit (4 bytes) floating point numbers
    total_size_kb = total_size_bytes / 1024  # Convert bytes to megabytes
    return total_size_kb

def calculate_number_of_parameter(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params