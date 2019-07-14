class DefaultConfigs(object):
    """
    global parameters for training
    """
    # 1.string parameters
    train_data = "D:\\Documents\\project\\deepfakeface\\deepfake_database\\deepfake_database\\train_test\\"
    test_data = ""
    val_data = "D:\\Documents\\project\\deepfakeface\\deepfake_database\\deepfake_database\\validation\\"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    gpus = "1"
    model_name = "XceptionNet"
    logs = "./logs/"

    # 2.numeric parameters
    epochs = 16
    batch_size = 16
    img_height = 256
    img_weight = 256
    num_classes = 2
    seed = 888
    lr = 1e-3
    lr_decay = 1e-4
    weight_decay = 1e-4
    stepLR_size = 2


configTrain = DefaultConfigs()
