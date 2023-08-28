def image_feature_vector(dataloader, model):
    # 모델의 마지막 층을 제거하여 특징 추출
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    imaga_list, label_list = [], []
    for idx, (image, label) in enumerate(tqdm(dataloader)):
        image, label = image.to(CFG.DEVICE), label.to(CFG.DEVICE)
        with torch.inference_mode():
            prediction = model(image)
        imaga_list.append(prediction.cpu().detach().numpy().reshape(-1))
        label_list.append(label.cpu().detach().numpy())

    return imaga_list, label_list
