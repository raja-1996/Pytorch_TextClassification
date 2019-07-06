import pandas as pd
from Attention_Classification.Tokenizer import Tokenizer
import torch

class InputExample:

    def __init__(self, id, text, label):
        self.id = id
        self.text = text
        self.label =label


class InputFeature:

    def __init__(self, id, text_ids, label, word_lengths, sent_length):
        self.id = id
        self.text_ids = text_ids
        self.label = label
        self.word_lengths = word_lengths
        self.sent_length = sent_length

class Dataprocessor:
    
    def __init__(self, config):

        self.tokenizer = Tokenizer(vocab_file=config['vocab_file'])
        self.max_sents = config['max_sents']
        self.max_words = config['max_words']

        self.example_func_dict = {
            'train' : self.get_train_examples,
            'valid' : self.get_dev_examples,
            'test' : self.get_test_examples
        }

    def read_csv(self, filepath, mode):
        df = pd.read_csv(filepath)
        sentences = df['text']
        if mode in ['train', 'valid']:
            label = df['labels']
            data = list(zip(sentences, label))
        elif mode in ['test']:
            label = None
            data = [(each, None) for each in sentences]
        return data

    def create_examples(self, data):
        examples = []
        for i, dat in enumerate(data):
            text, label = dat
            example = InputExample(
                id=i,
                text=text,
                label=label
            )
            examples.append(example)
        return examples

    def get_train_examples(self, filepath='data/train.csv'):
        data = self.read_csv(filepath, 'train')
        examples = self.create_examples(data)
        return examples

    def get_dev_examples(self, filepath='data/valid.csv'):
        data = self.read_csv(filepath, 'valid')
        examples = self.create_examples(data)
        return examples

    def get_test_examples(self, filepath='data/test.csv'):
        data = self.read_csv(filepath, 'test')
        examples = self.create_examples(data)
        return examples

    def convert_examples_to_features(self, examples):
        features = []
        label_list = self.get_labels()

        for i, example in enumerate(examples):
            text = example.text
            label = example.label

            text_tokenized = self.tokenizer.tokenize(text)

            tokens = []
            word_lengths, sent_lengths = [], []
            for words in text_tokenized:
                ids = self.tokenizer.convert_tokens_to_ids(words)
                ids = ids[:self.max_words]
                word_lengths.append(len(ids))
                while len(ids) < self.max_words:
                    ids.append(0)
                tokens.append(ids)

            tokens = tokens[:self.max_sents]
            word_lengths = word_lengths[:self.max_sents]
            sent_length = len(tokens)
            while len(tokens) < self.max_sents:
                temp_sent = [ 0 for _ in range(self.max_words)]
                word_lengths.append(len(temp_sent))
                tokens.append(temp_sent)

            label = label_list.index(label) if label else None
            feature = InputFeature(
                id=i,
                text_ids=tokens,
                label=label,
                word_lengths=word_lengths,
                sent_length=sent_length
            )

            features.append(feature)
        return features

    def get_features(self, features, mode):
        text_ids = torch.tensor([f.text_ids for f in features])
        word_lengths = torch.tensor([f.word_lengths for f in features])
        sent_lengths = torch.tensor([f.sent_length for f in features])

        if mode in ['train', 'valid']:
            labels = torch.tensor([f.label for f in features])
        elif mode in ['test']:
            labels = None

        return text_ids, labels, word_lengths, sent_lengths

    def get_examples(self, mode):
        example_func = self.example_func_dict[mode]
        examples = example_func()
        return examples

    def get_labels(self):
        return ['rec.autos', 'comp.sys.mac.hardware', 'comp.graphics', 'sci.space', 'talk.politics.guns', 'sci.med', 'comp.sys.ibm.pc.hardware', 'comp.os.ms-windows.misc', 'rec.motorcycles', 'talk.religion.misc', 'misc.forsale', 'alt.atheism', 'sci.electronics', 'comp.windows.x', 'rec.sport.hockey', 'rec.sport.baseball', 'soc.religion.christian', 'talk.politics.mideast', 'talk.politics.misc', 'sci.crypt']
        # return ['sports', 'business']
        # return ['politics', 'entertainment']


    
    