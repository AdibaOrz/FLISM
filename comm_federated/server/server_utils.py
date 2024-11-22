from comm_federated.server.Comm_FedAvg_Server import FedAvg_Server
from comm_federated.server.Comm_FedProx_Server import FedProx_Server
from comm_federated.server.Comm_AutoFed_Server import AutoFed_Server
from comm_federated.server.Comm_FedMultiModal_Server import FedMultiModal_Server
from comm_federated.server.Comm_Harmony_Server import Harmony_Server
from comm_federated.server.Comm_MOON_Server import MOON_Server
from comm_federated.server.Comm_FLISM_Server import FLISM_Server



def get_server(model, unique_user_list, user_mod_drop_dict, speed_distri, user_groups_train, user_groups_test, args):
    method = args.method
    if method == 'fedavg':
        server = FedAvg_Server(model, unique_user_list, speed_distri, args)
    elif method == 'fedprox':
        server = FedProx_Server(model, unique_user_list, speed_distri, args)
    elif method == 'moon':
        server = MOON_Server(model, unique_user_list, speed_distri, args)
    elif method == 'harmony':
        server = Harmony_Server(unique_user_list, user_mod_drop_dict, speed_distri, args)
    elif method == 'fedmultimodal':
        server = FedMultiModal_Server(model, unique_user_list, user_mod_drop_dict, speed_distri, args)
    elif method == 'autofed':
        main_model = model["model"]
        generative_model = model["generative_model"]
        server = AutoFed_Server(main_model, generative_model, unique_user_list, user_groups_train, user_groups_test, speed_distri, args)
    elif method == 'flism':
        server = FLISM_Server(model, unique_user_list, speed_distri, args)
    else:
        raise ValueError(f"Method {method} not supported")
    return server