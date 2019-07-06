from Attention_Classification.Tokenizer import Tokenizer
from Attention_Classification.attention_config import attention_config as config


def test_tokenizer():
    print()
    print('Testing Tokenizer !!!')

    tokenizer = Tokenizer(vocab_file=config['vocab_file'])
    tokens = ['i', 'love', 'you3000']

    print(tokenizer.convert_tokens_to_ids(tokens))

if __name__ == '__main__':
    test_tokenizer()


