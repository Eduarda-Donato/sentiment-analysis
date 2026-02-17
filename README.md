# Sentiment Analysis with RNN

This project implements a sentiment analysis model using a Recurrent Neural Network (RNN) in PyTorch, trained on movie review data.

## Project Structure

The notebook covers the following steps:

1.  **Data Loading and Preprocessing**: Loads the 'moviesData' dataset, selects relevant columns, renames labels, and converts sentiment labels to numerical format (0 for negative, 1 for positive).
2.  **Data Splitting**: Splits the dataset into training, validation, and test sets.
3.  **Data Balancing**: Applies undersampling to the training data to handle class imbalance.
4.  **Tokenization**: Uses a pre-trained BERT tokenizer to tokenize the review texts.
5.  **Dataset and DataLoader Creation**: Prepares the tokenized data into `torch.utils.data.Dataset` objects and creates `DataLoader` instances for batching.
6.  **Model Definition**: Defines an RNN-based `TextClassifier` model using `nn.Embedding`, `nn.RNN`, and `nn.Linear` layers.
7.  **Training and Evaluation Functions**: Implements functions for training (`train`), validation (`val`), and testing (`test`) the model, including loss calculation and accuracy metrics.
8.  **Model Training**: Trains the model for a specified number of epochs, tracks loss and accuracy, and saves the best performing model based on validation accuracy.
9.  **Results Visualization**: Plots the training and validation loss over epochs.
10. **Final Evaluation**: Evaluates the trained model on the test set.
11. **Example Predictions**: Demonstrates how to use the trained model to predict sentiment on new, unseen text examples.

## Setup and Usage

To run this notebook, ensure you have the necessary libraries installed:

```bash
pip install pandas numpy tqdm imblearn torch transformers datasets
```

**Key Steps to Run:**

1.  **Load Dataset**: The dataset is loaded using `load_dataset("AbrilCota/moviesData")`.
2.  **Define Model Parameters**: Adjust `EMBED_DIM`, `HIDDEN_DIM`, `N_LAYERS`, `LR`, and `EPOCHS` as needed.
3.  **Execute Cells**: Run all cells sequentially in the notebook.

## Model Details

-   **Architecture**: Simple RNN with an embedding layer and a linear output layer.
-   **Tokenizer**: `bert-base-uncased` tokenizer.
-   **Loss Function**: `nn.BCEWithLogitsLoss` for binary classification.
-   **Optimizer**: `torch.optim.Adam`.

## Results

After training for 5 epochs, the model achieved approximately:

-   **Validation Accuracy**: 76.03%
-   **Test Accuracy**: 75.7%

Individual model checkpoints are saved as `best_model_epoch_X.pt` where `X` corresponds to the epoch with the best validation accuracy.
