from models.WESAD_Model import WESAD_Model
from models.PAMAP2_Model import PAMAP2_Model
from models.Simulated_Model import Simulated_Model
from models.SleepEDF20_Model import SleepEDF20_Model
from models.RealWorldHAR_Model import RealWorldHAR_Model

# FedMultiModal
from models.FedMultimodal.WESAD_Model import WESAD_Dynamic_Model
from models.FedMultimodal.PAMAP2_Model import PAMAP2_Dynamic_Model
from models.FedMultimodal.Simulated_Model import Simulated_Dynamic_Model
from models.FedMultimodal.SleepEDF20_Model import SleepEDF20_Dynamic_Model
from models.FedMultimodal.RealWorldHAR_Model import RealWorldHAR_Dynamic_Model

# AutoFed
from models.AutoFed.PAMAP2_Generative_Model import PAMAP2_Dynamic_Autoencoder
from models.AutoFed.SleepEDF20_Generative_Model import SleepEDF20_Dynamic_Autoencoder


def get_models(method_name, dataset_name, args):
    if method_name in ['fedavg', 'fedprox', 'moon', 'flism']:
        model = get_general_models(dataset_name, args)
    elif method_name == 'fedmultimodal':
        model = get_fedmultimodal_model(dataset_name, args)
    elif method_name == 'autofed':
        main_model = get_general_models(dataset_name, args)
        generative_model = get_generative_models(dataset_name)
        model = {"model" : main_model,
                 "generative_model": generative_model}
    elif method_name == 'harmony':
        model = None # dynamic model is obtained inside the server
    else:
        raise ValueError(f"Method {method_name} not supported")
    return model


def get_general_models(dataset_name, args):
    if dataset_name == 'pamap2':
        return PAMAP2_Model()
    elif dataset_name == 'realworldhar':
        return RealWorldHAR_Model()
    elif dataset_name == 'sleepedf20':
        return SleepEDF20_Model()
    elif dataset_name == 'wesad':
        return WESAD_Model()
    elif dataset_name == 'simulated':
        return Simulated_Model(args)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

def get_fedmultimodal_model(dataset_name, args):
    if dataset_name == 'pamap2':
        model = PAMAP2_Dynamic_Model()
    elif dataset_name == 'realworldhar':
        model = RealWorldHAR_Dynamic_Model()
    elif dataset_name == 'sleepedf20':
        model = SleepEDF20_Dynamic_Model()
    elif dataset_name == 'wesad':
        model = WESAD_Dynamic_Model()
    elif dataset_name == 'simulated':
        model = Simulated_Dynamic_Model(args)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return model

def get_generative_models(dataset_name):
    if dataset_name == 'pamap2':
        model = PAMAP2_Dynamic_Autoencoder()
    elif dataset_name == 'sleepedf20':
        model = SleepEDF20_Dynamic_Autoencoder()

    return model
