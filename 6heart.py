import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
    'heartdisease'
]

heartDisease = pd.read_csv('heart.csv', names=names)
heartDisease = heartDisease.replace('?', np.nan)

model = BayesianModel([
    ('age', 'trestbps'), 
    ('age', 'fbs'), 
    ('sex', 'trestbps'), 
    ('exang', 'trestbps'),
    ('trestbps', 'heartdisease'),
    ('fbs', 'heartdisease'),
    ('heartdisease', 'restecg'), 
    ('heartdisease', 'thalach'),
    ('heartdisease', 'chol')
])


model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)


HeartDisease_infer = VariableElimination(model)
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 37, 'sex': 0})
print(q)
