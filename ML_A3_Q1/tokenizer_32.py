
import pickle

def save_tokenizer(tokenizer, path='ML_A3_Q1/tokenizer_32.pkl'):
    # Save the tokenizer to a file
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(path='ML_A3_Q1/tokenizer_32.pkl'):
    # Load the tokenizer from a file
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer
