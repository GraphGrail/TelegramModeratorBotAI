import os
from commentEvaluator import CommentEvaluator
from keras.models import load_model
import pickle

script_dir = os.path.dirname(__file__)
with open(os.path.join(script_dir, 'model_files/token.pickle'), 'rb') as handle:
    tok = pickle.load(handle)

model = load_model(os.path.join(script_dir, 'model_files/rus_weights2.hdf5'))
evaluator = CommentEvaluator(model, tok)

while True:
    text = input('Введите текст для проверки: ')
    print(evaluator.analyze(text))
    print()
