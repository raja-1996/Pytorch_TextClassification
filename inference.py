
from Attention_Classification.HierarchicalAttention import HierarchicalAttention
from Attention_Classification.utils import get_inference_dataloader, get_device_n_gpus
from Attention_Classification.DataProcessor import Dataprocessor
import torch

class NewsClassification:
    
    def __init__(self, config, device='cpu'):
        

        self.processor = Dataprocessor(config)
        self.labels_set = self.processor.get_labels()
        num_classes = len(self.labels_set)
        config['num_classes'] = num_classes

        self.device, n_gpu = get_device_n_gpus(rank=-2)
        config['device'] = self.device
        config['n_gpu'] = n_gpu
        print('Device {}, n_gpu {}'.format(self.device, n_gpu))
        
        self.model = HierarchicalAttention(config).to(self.device)
        
    def predict(self, documents):
        data = [(each, None) for each in documents]
        test_loader = get_inference_dataloader(self.processor, data)
        classes = []
        ids, words, sents = [], [], []
        for i, batch in enumerate(test_loader):
            batch = [each.to(self.device) for each in batch]
            input_ids, word_lengths, sent_lengths = batch

            scores, word_attn_scores, sent_attn_scores = self.model(input_ids, word_lengths, sent_lengths)
            idxs = torch.argmax(scores, dim=-1)
            classes_ = [self.labels_set[idx] for idx in idxs.tolist()]
            classes.extend(classes_)
            ids.extend(input_ids.tolist())
            words.extend(word_attn_scores.tolist())
            sents.extend(sent_attn_scores.tolist())

        result = list(zip(documents, classes))
        attn_scores = list(zip(ids, words, sents))

        return result, attn_scores

