import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    emb = np.random.uniform(-0.08, 0.08, (len(vocab), emb_size))
    
    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab:
                word_id = vocab[word]
                emb_vec = np.array([float(x) for x in parts[1:]])
                if len(emb_vec) == emb_size:
                    emb[word_id] = emb_vec
    
    return emb


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        # Embedding layer
        self.embeddings = nn.Embedding(len(self.vocab), self.args.emb_size, 
                                       padding_idx=self.vocab.pad_id)
        
        # Dropout layers
        self.word_dropout = nn.Dropout(self.args.word_drop)
        self.emb_dropout = nn.Dropout(self.args.emb_drop)
        self.hid_dropout = nn.Dropout(self.args.hid_drop)
        
        # Feedforward layers
        self.hidden_layers = nn.ModuleList()
        
        # First hidden layer: from embedding to hidden
        self.hidden_layers.append(nn.Linear(self.args.emb_size, self.args.hid_size))
        
        # Additional hidden layers: from hidden to hidden
        for _ in range(self.args.hid_layer - 1):
            self.hidden_layers.append(nn.Linear(self.args.hid_size, self.args.hid_size))
        
        # Output layer: from hidden to tag scores
        self.output_layer = nn.Linear(self.args.hid_size, self.tag_size)
        
        # Activation function
        self.activation = nn.ReLU()

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        init_range = 0.08
        
        nn.init.uniform_(self.embeddings.weight, -init_range, init_range)
        
        for layer in self.hidden_layers:
            nn.init.uniform_(layer.weight, -init_range, init_range)
            nn.init.uniform_(layer.bias, -init_range, init_range)
        
        nn.init.uniform_(self.output_layer.weight, -init_range, init_range)
        nn.init.uniform_(self.output_layer.bias, -init_range, init_range)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        emb_matrix = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        
        self.embeddings.weight.data.copy_(torch.from_numpy(emb_matrix))

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        # Apply word dropout
        if self.training and self.args.word_drop > 0:
            mask = torch.rand(x.shape, device=x.device) > self.args.word_drop
            x = x * mask.long()
        
        # Get word embeddings: [batch_size, seq_length, emb_size]
        embeds = self.embeddings(x)
        embeds = self.emb_dropout(embeds)
        
        # Pool embeddings
        mask = (x != self.vocab.pad_id).float().unsqueeze(2)  # [batch_size, seq_length, 1]
        
        if self.args.pooling_method == "avg":
            # Average pooling
            sum_embeds = (embeds * mask).sum(dim=1)  # [batch_size, emb_size]
            seq_lengths = mask.sum(dim=1)  # [batch_size, 1]
            h = sum_embeds / (seq_lengths + 1e-10)
        elif self.args.pooling_method == "sum":
            # Sum pooling
            h = (embeds * mask).sum(dim=1)
        elif self.args.pooling_method == "max":
            # Max pooling
            embeds_masked = embeds.masked_fill(mask == 0, -1e9)
            h = embeds_masked.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.args.pooling_method}")
        
        # Pass through feedforward layers
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.activation(h)
            h = self.hid_dropout(h)
        
        scores = self.output_layer(h)
        
        return scores
