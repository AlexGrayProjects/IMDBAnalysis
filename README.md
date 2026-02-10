# IMDBAnalysis
Using freely available data from kaggle.com, we create an LSTM model to perform a classic NLP task of sentiment analysis on IMDB reviews. An LSTM is a specialized RNN which allows for memory to be saved over multiple states. Doing this provides the model with more contextual information from earlier in the sequence, allowing it to pass through the network using the cell state (C state in model), which decides whether to remember or forget the previous input. 

## Results
As we care about both positive and negative accuracy, an F score would not be totally appropriate in this instance. Instead, we use Mathews Correlation Coefficient (MCC), which gives an output between -1 and 1. A high score represents total agreement with the predicted and actual outputs, whereas a zero represents no correlation. We acheive an MCC of 0.9095, showing a good model with strong predictive capabilities.

