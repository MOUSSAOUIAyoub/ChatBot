import torch
import transformers
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the data from the CSV file
data = pd.read_csv('data.csv')

# Train-test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Define the number of classes
num_classes = len(train_data['intent'].unique())

# Define the learning rate
learning_rate = 2e-5

# Define the number of epochs
num_epochs = 3

# Define the batch size
batch_size = 32

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize the text data
train_tokens = tokenizer.batch_encode_plus(train_data['text'].values.tolist(), padding=True, truncation=True)
test_tokens = tokenizer.batch_encode_plus(test_data['text'].values.tolist(), padding=True, truncation=True)

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(torch.tensor(train_tokens['input_ids']), torch.tensor(train_tokens['attention_mask']), torch.tensor(train_data['intent'].values))
test_dataset = TensorDataset(torch.tensor(test_tokens['input_ids']), torch.tensor(test_tokens['attention_mask']), torch.tensor(test_data['intent'].values))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Set up the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
loss_fn = torch.nn.CrossEntropyLoss()

# Define the function to fine-tune the BERT model
def fine_tune_bert():
    # Set the model to training mode
    model.train()

    # Fine-tune the model for the specified number of epochs
    for epoch in range(num_epochs):
        train_loss = 0.0

        # Loop over the training data in batches
        for batch in train_dataloader:
            # Get the inputs and labels
            batch_inputs, batch_masks, batch_labels = batch

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_inputs, attention_mask=batch_masks, labels=batch_labels)
            loss = outputs.loss
            train_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

        # Compute the average training loss for this epoch
        train_loss /= len(train_dataloader)


       


        

# Define a function to predict the intent of a given utterance
def predict_intent(utterance):
    inputs = tokenizer.encode_plus(
        utterance,
        add_special_tokens=True,
        return_tensors='pt'
    )

    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    _, predicted = torch.max(outputs[0], dim=1)

    return predicted.numpy()[0]

