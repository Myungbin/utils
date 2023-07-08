def get_stacking_model(model, X, y, test, n_fold):
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    folds = []

    for train_idx, val_idx in skf.split(X, y):
        folds.append((train_idx, val_idx))

    fold_model = {}

    for f in range(n_fold):
        print(
            f'===================================={f+1}============================================')
        train_idx, val_idx = folds[f]

        x_train, x_val, y_train, y_val = X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]

        model.fit(x_train, y_train)

        y_pred = model.predict(x_val)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"{f + 1} Fold MAE = {mae}")
        fold_model[f] = model
        print(f'================================================================================\n\n')

    train_submission = pd.read_csv('./jeju_data/train_id_target.parquet')
    test_submossion = pd.read_csv('./jeju_data/sample_submission.csv')

    for fold in range(n_fold):
        train_submission['target'] += fold_model[fold].predict(X)/n_fold
        test_submossion['target'] += fold_model[fold].predict(
            test)/n_fold

    return train_submission['target'], test_submossion['target']


XGB = XGBRegressor(**xgb_param)
Cat = CatBoostRegressor(**cat_param)
lgbm = LGBMRegressor(**lgb_param)


xgb_train_pred, xgb_test_pred = get_stacking_model(XGB, X, y, test, 7)
Cat_train_pred, Cat_test_pred = get_stacking_model(Cat, X, y, test, 7)
lgbm_train_pred, lgbm_test_pred = get_stacking_model(lgbm, X, y, test, 7)

Stack_final_X_train = pd.concat([xgb_train_pred, Cat_train_pred, lgbm_train_pred], axis=1)
Stack_final_X_test = pd.concat([xgb_test_pred, Cat_test_pred, lgbm_test_pred], axis=1)

meta_model = LinearRegression()
meta_model.fit(Stack_final_X_train, y)
stack_final = meta_model.predict(Stack_final_X_test)

