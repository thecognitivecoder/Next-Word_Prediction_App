import pickle
def save_tokenizer(tokenizer, path='ML_A3_Q1/tokenizer_64_holmes.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(path='ML_A3_Q1/tokenizer_64_holmes.pkl'):
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer
