def save_model(model, file_name):
    if not os.path.exists(cfg.SAVE_MODEL_PATH):
        os.makedirs(cfg.SAVE_MODEL_PATH)
    torch.save(model.state_dict(), os.path.join(cfg.SAVE_MODEL_PATH, file_name))


def load_model_state_dict(model, file_name):
    path = os.path.join(cfg.SAVE_MODEL_PATH, f'{file_name}.pth')
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
