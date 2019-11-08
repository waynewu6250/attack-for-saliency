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
    
    epochs = 10
    batch_size = 32
    lr = 1e-4

    # For attack model
    alpha = 0.4

opt = Config()