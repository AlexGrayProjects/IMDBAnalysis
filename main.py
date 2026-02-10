<<<<<<< Updated upstream
import model
import torch
from torch.utils.data import DataLoader
from torch import nn
import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize
import pickle
import re


with open('word_bank.pkl', 'rb') as f:
    word_bank = pickle.load(f)
def data_pipeline(review):
    remove_html = lambda x : re.sub(r'<.*?>', '', x).lower()
    review = remove_html(review)
    # Simple function to prepare review for mopel use
    tokens = word_tokenize(review)
    output = []
    for token in tokens:
        if token in word_bank.keys():
            output.append(word_bank[token])
        else:
            output.append(0)
    n = len(output)
    output = torch.tensor(output).reshape(1, n)
    return output
        

if __name__ == '__main__':
    PATH = 'model.pt'
    model = model.LSTM(max(word_bank.values()) + 1, 256)
    model.load_state_dict(torch.load(PATH, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        while True:
            choice = input(
                '''Use this LSTM to predict whether IMDB reviews are positive or negative.
                (Enter) Predict sentiment of IMDB review 
                (Q) Exit
                ''')
            if choice == 'Q':
                print('Goodbye!')
                break
            review = input('Enter review: ')
            prediction = model(data_pipeline(review)).item()
            threshold = 0.5
            print('Prediction value : {:.2f}'.format(prediction))
            if prediction > threshold:
                print('The model predicts a positive review')
            else:
                print('The model predicts a negative review')

=======
import model
import torch
from torch.utils.data import DataLoader
from torch import nn
import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize
import pickle
import re


with open('word_bank.pkl', 'rb') as f:
    word_bank = pickle.load(f)
def data_pipeline(review):
    remove_html = lambda x : re.sub(r'<.*?>', '', x).lower()
    review = remove_html(review)
    # Simple function to prepare review for mopel use
    tokens = word_tokenize(review)
    output = []
    for token in tokens:
        if token in word_bank.keys():
            output.append(word_bank[token])
        else:
            output.append(0)
    n = len(output)
    output = torch.tensor(output).reshape(1, n)
    return output
        

if __name__ == '__main__':
    PATH = 'model.pt'
    model = model.LSTM(max(word_bank.values()) + 1, 256)
    model.load_state_dict(torch.load(PATH, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        while True:
            choice = input(
                '''Use this LSTM to predict whether IMDB reviews are positive or negative.
                (Enter) Predict sentiment of IMDB review 
                (Q) Exit
                ''')
            if choice == 'Q':
                print('Goodbye!')
                break
            review = input('Enter review: ')
            prediction = model(data_pipeline(review)).item()
            threshold = 0.5
            print('Prediction value : {:.2f}'.format(prediction))
            if prediction > threshold:
                print('The model predicts a positive review')
            else:
                print('The model predicts a negative review')

>>>>>>> Stashed changes
        