"""Utility function for Harmony models."""


def get_dynamic_model(modality_indices, device, args):
    if args.dataset == "wesad":
        import models.harmony_models.WESAD_Harmony_Models as models
        dynamic_model = models.Dynamic_Model(modality_indices, device, args.verbose)
    elif args.dataset == "realworldhar":
        import models.harmony_models.RealWorldHAR_Harmony_Models as models
        dynamic_model = models.Dynamic_Model(modality_indices, device, args.verbose)
    elif args.dataset == "pamap2":
        import models.harmony_models.PAMAP2_Harmony_Models as models
        dynamic_model = models.Dynamic_Model(modality_indices,device, args.verbose)
    elif args.dataset == "sleepedf20":
        import models.harmony_models.SleepEDF20_Harmony_Models as models
        dynamic_model = models.Dynamic_Model(modality_indices, device, args.verbose)
    elif args.dataset == "simulated":
        import models.harmony_models.Simulated_Harmony_Models as models
        dynamic_model = models.Dynamic_Model(modality_indices, device, args)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
    return dynamic_model