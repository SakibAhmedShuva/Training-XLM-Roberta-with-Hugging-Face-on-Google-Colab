# Training XLM-RoBERTa with Hugging Face on Google Colab

This repository contains code for fine-tuning XLM-RoBERTa for Named Entity Recognition (NER) using the Hugging Face Transformers library. The implementation is specifically designed to run on Google Colab.

## Overview

This project demonstrates how to:
- Fine-tune XLM-RoBERTa on the CoNLL-2003 dataset for NER
- Process and tokenize text data for NER tasks
- Implement custom evaluation metrics
- Save and load the trained model
- Perform inference on new text

## Requirements

The following packages are required:
```bash
pip install transformers
pip install datasets
pip install seqeval
pip install evaluate
pip install torch
```

## Dataset

The project uses the CoNLL-2003 dataset, which is automatically downloaded through the Hugging Face datasets library. The dataset contains text annotated with the following entity types:
- PER (Person)
- ORG (Organization)
- LOC (Location)
- MISC (Miscellaneous)

## Model Architecture

- Base Model: `xlm-roberta-base`
- Task: Token Classification (NER)
- Training Framework: Hugging Face Transformers

## Training

The training process includes:
- Custom tokenization with label alignment
- Evaluation metrics calculation (Precision, Recall, F1-score)
- Training arguments optimization
- Model checkpointing

Key training parameters:
```python
learning_rate=2e-5
per_device_train_batch_size=16
num_train_epochs=1
weight_decay=0.01
```

## Usage

1. Open the `xlm_hf.ipynb` notebook in Google Colab
2. Run the installation cells to set up the required packages
3. Execute the training cells to fine-tune the model
4. Use the inference code to make predictions on new text

Example inference:
```python
text = "Your text here"
result = perform_ner(text, loaded_model, loaded_tokenizer, id2label)
print(result)
```

## Model Outputs

The model returns predictions in JSON format:
```json
{
  "ENTITY_TYPE": [
    {
      "token": "word",
      "label": "B-ENTITY_TYPE",
      "confidence": "95.67%"
    }
  ]
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- Hugging Face for their Transformers library
- The XLM-RoBERTa team for the pre-trained model
- CoNLL-2003 dataset creators

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
