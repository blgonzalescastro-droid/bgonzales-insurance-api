import joblib
import numpy as np
import sklearn

model = joblib.load('./model/model.pkl')
sc_x = joblib.load('./model/scaler_x.pkl')
sc_y = joblib.load('./model/scaler_y.pkl')

age= int (input("Ingrese la edad de la persona:  "))
age_sc = sc_x.transform(np.array([[age]]))

insurance_sc = model.predict(age_sc)
insurance = sc_y.inverse_transform(insurance_sc)
print(f'El costo del seguro es de {insurance[0][0]:.2f} para una persona de {age} anos')
