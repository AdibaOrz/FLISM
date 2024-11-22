from federated.server.FedAvg_Server import FedAvg_Server
from federated.server.FedProx_Server import FedProx_Server
from federated.server.AutoFed_Server import AutoFed_Server
from federated.server.FedMultiModal_Server import FedMultiModal_Server
from federated.server.Harmony_Server import Harmony_Server
from federated.server.MOON_Server import MOON_Server
from federated.server.FLISM_Server import FLISM_Server



def get_server(model, train_set, test_set, train_user_groups, test_user_groups, user_mod_drop_dict, device, args):
    method = args.method
    if method == 'fedavg':
        server = FedAvg_Server(model, train_set, test_set, train_user_groups, test_user_groups, user_mod_drop_dict, device, args)
    elif method == 'fedprox':
        server = FedProx_Server(model, train_set, test_set, train_user_groups, test_user_groups, user_mod_drop_dict, device, args)
    elif method == 'moon':
        server = MOON_Server(model, train_set, test_set, train_user_groups, test_user_groups, user_mod_drop_dict, device, args)
    elif method == 'harmony':
        server = Harmony_Server(train_set, test_set, train_user_groups, test_user_groups, user_mod_drop_dict, device, args)
    elif method == 'fedmultimodal':
        server = FedMultiModal_Server(model, train_set, test_set, train_user_groups, test_user_groups, user_mod_drop_dict, device, args)
    elif method == 'autofed':
        server = AutoFed_Server(model, train_set, test_set, train_user_groups, test_user_groups, user_mod_drop_dict, device, args)
    elif method == 'flism':
        server = FLISM_Server(model, train_set, test_set, train_user_groups, test_user_groups, user_mod_drop_dict, device, args)
    else:
        raise ValueError(f"Method {method} not supported")
    return server