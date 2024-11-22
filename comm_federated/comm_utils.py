import os
import json

def log_communication_result_to_json(total_communication_cost, total_number_of_trained_parameters, args):
    log_file_path = f'{args.json_log_path}/communication/{args.final_filename}_{args.note}/{args.final_general_filename}'
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    full_path_to_json_file = f"{log_file_path}/test_results.json"
    all_test_results = {
        'total_communication_cost': total_communication_cost,
        'total_number_of_trained_parameters': total_number_of_trained_parameters,
    }
    with open(full_path_to_json_file, 'w') as f:
        json.dump(all_test_results, f, indent=4)
    print(f"====Saved the results to {full_path_to_json_file}====")