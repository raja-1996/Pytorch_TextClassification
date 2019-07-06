

attention_config = {

    'num_epochs': 50,
    'batch_size' : 30,
    'rank': 0,
    'model_dir': 'Attention_Classification/Models',
    'model_name': 'best_model.pth',

    'stats_dir': 'Attention_Classification/Stats',
    'stats_name': 'stats.pkl',

    'results_dir': 'Results/',

    'vocab_size': 123123,
    'embed_size' : 300,
    'hidden_size': 128,
    'rnn_type': 'GRU',
    'n_layers': 1,
    'dropout': 0.4,
    'bidir': True,

    'max_sents': 15,
    'max_words': 40,

    'num_classes' : -1,

    'load_pretrained_word_embeddings': True,
    'weight_matrix_path': 'Attention_Classification/Embeddings/embedding_matrix.npy',
    'vocab_file': 'Attention_Classification/Embeddings/word_index.pkl',

}