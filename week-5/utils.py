# method to test the model for particular sample
def predict(model_obj, client):
    X = model_obj.dv.transform([client])
    y_pred = float(round(model_obj.model.predict_proba(X)[0,1],3))
    return y_pred