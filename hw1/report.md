# CS639 Assignment 1 Report
**Handan Hu**

ID: 9087303443

## Improvements

### 1. Pre-trained Word Embeddings (GloVe)

- Download link: https://nlp.stanford.edu/data/glove.6B.zip
- Used `glove.6B.300d.txt` for the SST dataset

- **SST Dataset:** WITH GloVe embeddings
  - Result: ~2% accuracy improvement
  - Analysis: SST contains formal sentences similar to GloVe's training data (Wikipedia and news articles)
  
- **CFIMDB Dataset:** WITHOUT GloVe embeddings
  - Result: Better performance without pre-trained embeddings
  - Analysis: Movie reviews may contain informal language and domain-specific expressions that differ from GloVe's training corpus.

### 2. Adam Optimizer

Replaced the default Adagrad optimizer with Adam, which typically leads to faster convergence and more stable training.

- Optimizer: `torch.optim.Adam`
- Learning rate: 0.001 (lowered from 0.005 for Adam)
- Supports both optimizers via `--optimizer` flag

### 3. Early Stopping

Implemented patience-based early stopping to prevent overfitting.

- Patience: 5 evaluations
- Monitors dev set accuracy every 500 iterations
- Stops training if no improvement for 5 consecutive evaluations

## Experimental Results

### Run 1
- **SST Dev:** 0.4269
- **SST Test:** 0.4276
- **CFIMDB Dev:** 0.9429

### Run 2
- **SST Dev:** 0.4178
- **SST Test:** 0.4285
- **CFIMDB Dev:** 0.9265

### Average Performance
- **SST Dev:** 0.4224 (+2.73% over baseline)
- **SST Test:** 0.4281 (+1.59% over baseline)
- **CFIMDB Dev:** 0.9347 (+1.23% over baseline)

