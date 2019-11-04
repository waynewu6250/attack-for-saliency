class Config:

    data_path = "./data/train_images.pkl"
    label_path = "./data/train_labels.pkl"
    val_path = "./data/val_images.pkl"
    val_label_path = "./data/val_labels.pkl"

    model_path = None
    
    epochs = 10
    batch_size = 64
    lr = 1e-3

opt = Config()