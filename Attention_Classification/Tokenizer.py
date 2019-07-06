
import json
import collections
import pickle



class Tokenizer:

    def __init__(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

    def load_vocab(self, vocab_file):
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    def tokenize(self, text):
        sentences = text.split('. ')
        text_sents_words = [sent.split() for sent in sentences]
        return text_sents_words

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(1)
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens






if __name__ == '__main__':
    tokenizer = Tokenizer(vocab_file=config['vocab_file'])
    tokens = ['i', 'love', 'you']
    print(tokenizer.convert_tokens_to_ids(tokens))


