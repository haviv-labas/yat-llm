# Yet Another Mini-LLM

## Notes / Completed TODOs

- [x] TODO: Separate loss calculation
- [x] TODO: Add block size as a hyperparameter
- [x] TODO: Check validation data is used correctly
- [x] TODO: Add GP optimiser
- [x] TODO: Add early stopping
- [x] TODO: Unify and move configs to one file
- [x] TODO: Speedup runtimes / deal with multiproc exceptions
- [x] TODO: Split training functionality away from model.py and Feedforward


## Useful commands

To run the solution, please use the provided Makefile - everything should run easily via the streamlit app with:

```
make run-streamlit-app
```

You can also run the above via the included Docker image. (optional)


## Future Work

### Using Word Embeddings Instead of Character-Level Modeling

- **Granularity and Context:** Character-level models capture granular data and can generate new words or handle misspelled words better. Word embeddings, however, provide richer semantic information, capturing relationships between words more effectively.

- **Vocabulary Size:** Word embeddings require managing a larger vocabulary, which can increase complexity and resource requirements. Character-level models have a much smaller vocabulary (the character set).

- **Training Data Requirements:** Word embeddings often need more data to capture the semantic relationships effectively.

- **Generalization:** Word embeddings can generalize better for different tasks (like translation, classification) due to their semantic richness.

#### Alternatives:

Subword Tokenization (like Byte Pair Encoding - BPE) balances between word-level and character-level, capturing common subword units. It's effective for languages with rich morphology and handling out-of-vocabulary words.

#### Alternative Measure for Model Quality

- **Why Character-level Accuracy is Bad:** If the accuracy measure is something like character-level accuracy, it doesn't capture the linguistic quality of the generated text.

- **Perplexity:** This is a common alternative, especially in language models. It measures how well a probability model predicts a sample and is better at capturing the quality of text generation.


### Informed Stopping

- **Early Stopping Based on Validation Performance:** Instead of a set number of iterations, use a validation set to monitor the model's performance. Stop training when the model's performance on the validation set starts to degrade or doesn't improve for a specified number of iterations (patience parameter).

- **Dynamic Learning Rate Adjustments:** Reduce the learning rate when the rate of improvement slows down.

## Requirements
- Python 3
- Pip

## Installation
- Create a virtual enviornment (recommended name is `venv` so that it is git ignored by default)
- Activate the environment
- Run `pip install -r requirements.txt`

