import torch
import warnings
from arg_parser import parse_arguments
from utils.general_utils import set_seed
from models.model_utils import get_models
from federated.server.server_utils import get_server
from utils.missing_utils import get_drop_list_for_all_users
from data_loader.data_loader import load_train_test_datasets

warnings.filterwarnings("ignore")


def main():
    args = parse_arguments()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # 1. Load data
    train_set, test_set, train_user_groups, test_user_groups= load_train_test_datasets(args.dataset)

    # 2. Define users with missing modalities
    unique_user_list = list(train_user_groups.keys())
    user_mod_drop_dict = get_drop_list_for_all_users(args, unique_user_list)

    # 3. Run method
    model = get_models(args.method, args.dataset, args)
    server = get_server(model, train_set, test_set, train_user_groups, test_user_groups, user_mod_drop_dict, device, args)
    # 4. Train and test
    server.run()



if __name__ == '__main__':
    main()