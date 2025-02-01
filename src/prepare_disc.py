def prepare_cv_ensamble_mps(ensemble,net_disc):  
    print('placing conv_1 of discriminator on cpu for mps compatible training')
    cv_list = ensemble.split('+')
    for idx, cv_model in enumerate(net_disc.cv_ensemble.models):
        if cv_list[idx] == 'clip':
            cv_model.model.conv1.to("cpu")
        elif cv_list[idx] == 'dino':
            cv_model.model.patch_embed.proj.to("cpu")
        elif cv_list[idx] == 'swin':
            print("mvoing conv swin on cpu")
            cv_model.model.patch_embed.to("cpu")
    return net_disc