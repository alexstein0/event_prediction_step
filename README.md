# Event Prediction Pipeline

This repository provides a pipeline for training and evaluating models for event prediction. The workflow is divided into five key steps, ranging from data downloading to model evaluation.

## Workflow

The repository runs things in the following steps:

1. **Download the Data**  
   Downloads and prepares the dataset for the pipeline.

2. **Train a Tokenizer**  
   Trains a tokenizer specific to the dataset for use during model training.

3. **Tokenize the Data**  
   Uses the trained tokenizer to tokenize the downloaded data, preparing it for training.

4. **Train the Model**  
   Trains a model on the tokenized dataset with specified parameters.

5. **Evaluate the Model**  
   Evaluates the model's performance on the test set.

## Installation

Make sure you have Python installed along with the required dependencies. You can install dependencies using:

```sh
pip install -r requirements.txt
```

## Usage

Below is an example usage of the pipeline with the `amazon_movies_5core` dataset:

### Step 1: Download the Data
```sh
python download_and_save_data.py data=amazon_movies_5core
```

### Step 2: Train the Tokenizer
```sh
python train_tokenizer.py data=amazon_movies_5core
```

### Step 3: Tokenize the Data
```sh
python tokenize_dataset.py data=amazon_movies_5core tokenizer_name=amazon_movies_5core_simple
```

### Step 4: Train the Model
```sh
python pretrain_HF.py seed=42 name=amazon_movies_reverse experiment=amazon_movies_default model.batch_size=64 model.epochs=10 model.train_test_split=.02 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=1000 model.randomize_order=True impl.save_intermediate_checkpoints=True
```

### Step 5: Evaluate the Model
```sh
python eval.py seed=42 name=amazon_movies experiment=amazon_movies_default model.batch_size=64 model.seq_length=10 model.randomize_order=True
```

## Parameters

- **Data**: The dataset to be used (`data=amazon_movies_5core`).
- **Tokenizer Name**: Name of the trained tokenizer (`tokenizer_name=amazon_movies_5core_simple`).
- **Model Training Parameters**:
  - **`seed`**: Random seed for reproducibility.
  - **`name`**: Model name.
  - **`experiment`**: Experiment identifier.
  - **`model.batch_size`**: Training batch size.
  - **`model.epochs`**: Number of training epochs.
  - **`model.train_test_split`**: Train-test split ratio.
  - **`model.seq_length`**: Sequence length for training.
  - **`model.lr`**: Learning rate.
  - **`impl.print_loss_every_nth_step`**: Interval for printing the training loss.
  - **`model.randomize_order`**: Whether to randomize the order of training data.
  - **`impl.save_intermediate_checkpoints`**: Whether to save checkpoints during training.

## Requirements

- Python 3.x
- Hugging Face Transformers
- PyTorch
- Other dependencies as listed in `requirements.txt`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
