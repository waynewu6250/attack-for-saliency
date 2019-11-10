class Config:

    path = "" #"attack-for-saliency/"
    data_path = path+"data/train_images.pkl"
    label_path = path+"data/train_labels.pkl"
    val_path = path+"data/val_images.pkl"
    val_label_path = path+"data/val_labels.pkl"
    dic_path = path+"data/label2id.pkl"

    model_path = path+"checkpoints/model-4.pth"
    save_model = 1

    attack_model = "fast_attack"
    netg_path = None
    lr1 = 2e-4
    beta1 = 0.5
    inf = 100
    gnf = 64
    
    epochs = 10
    batch_size = 32
    lr = 1e-4

    # For attack model
    alpha = 0.04

opt = Config()