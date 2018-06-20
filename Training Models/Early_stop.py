from sklearn.base import clone 

#prepare the data


def early_stop(X_train, y_train):
    
    poly_scaler = Pipline([
        ("poly_features", PolynomialFeatures(degree = 90, include_bias=False)), 
        ("std_scalar", StandardScalar()) ])
    X_train_poly_scaled = poly_scaler.fit_transform(X_train)
    X_val_poly_scaled = poly_scaler.transform(X_val)

    sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None, learning_rate="constant", eta=0.0005)

    minimum_val_error = float("inf")
    best_epoch = None
    best_model = None
    
    for epoch in range(1000):
        sgd_reg.fit(X_train_poly_scaled, y_train) #continues where it left off
        y_val_predict = sgd_reg.predict(X_val_poly_scaled)
        val_error = mean_squared_error(y_val_predict, y_val)
        
        if val_error < minimum_val_error:
            minimum_val_error = val_error
            best_epoch = epoch
            best_model = clone(sgd_reg)
            