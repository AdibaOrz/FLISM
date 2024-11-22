import warnings
import json
from utils.general_utils import set_seed
from utils.missing_utils import get_drop_list_for_all_users
from comm_federated.server.server_utils import get_server
from models.model_utils import get_models
from comm_arg_parser import parse_arguments
from configs import get_dataset_opts
from data_loader.data_loader import load_train_test_datasets

warnings.filterwarnings("ignore")

def main():
    # 0. Initial set-up
    args = parse_arguments()
    print(args)
    set_seed(args.seed) # Set seed
    # print(args)

    # 1. Load data
    if not args.simulated:
        args.dataset_opts = get_dataset_opts(args.dataset)
        _, _, user_groups_train, user_groups_test = load_train_test_datasets(args.dataset)
        unique_user_list = list(user_groups_train.keys())

    # If we simulate the dataset
    else:
        number_of_clients = args.simulated_clients_number
        unique_user_list = [str(i) for i in range(number_of_clients)]
        user_groups_train, user_groups_test = None, None
    user_mod_drop_dict = get_drop_list_for_all_users(args, unique_user_list)

    # 2. Get Speed Configuration
    speed_distri = None
    try:
        with open('comm_federated/speed_distribution.json', 'r') as f:
            speed_distri = json.load(f)
    except FileNotFoundError as e:
        print(f'Can not find a speed distribution file')

    # 3. Run method
    model = get_models(args.method, args.dataset, args)
    server = get_server(model, unique_user_list, user_mod_drop_dict, speed_distri, user_groups_train, user_groups_test, args)
    server.run()


if __name__ == '__main__':
    main()