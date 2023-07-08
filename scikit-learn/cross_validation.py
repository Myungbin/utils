def cross_validate_models(folds, X, y, model):
    """
    지정된 모델을 사용하여 주어진 데이터셋에 대해 교차 검증을 수행합니다.
    Args:
        folds (list): 교차 검증에서 각 폴드의 인덱스를 나타내는 튜플의 리스트입니다.
        X (DataFrame): 데이터셋의 입력 피처입니다.
        y (Series): 데이터셋의 타겟 변수입니다.
        model: 평가할 모델입니다.

    Returns:
        list: 각 폴드에 대해 훈련된 모델의 리스트입니다.
    """
    models = []

    for idx, (train_idx, val_idx) in enumerate(folds):
        print(f'===================================={idx+1}============================================')
        x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(x_train, y_train)

        y_pred = model.predict(x_val)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"{idx + 1} Fold MAE = {mae}")
        models.append(model)
        print(f'================================================================================\n\n')

    return models


if __name__ == "__main__":
    model = XGBRegressor(**params, tree_method='gpu_hist',
                         gpu_id=0, random_state=404)
    folds = list(StratifiedKFold(
        n_splits=9, shuffle=True, random_state=404).split(X, y))

    models = cross_validate_models(model, folds, X, y)
    predictions = sum(model.predict(test) for model in models) / len(folds)
