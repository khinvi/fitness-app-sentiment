import pandas as pd
import numpy as np
import re
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define constants
MAX_SEQUENCE_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FitnessAppReviewDataset:
    """Class to load and preprocess the dataset of fitness app reviews."""
    
    def __init__(self, data_path):
        """Initialize the dataset.
        
        Args:
            data_path: Path to the CSV file containing the reviews
        """
        self.data = pd.read_csv(data_path)
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Clean and preprocess text.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Join tokens back into string
        text = ' '.join(tokens)
        
        return text
    
    def get_sentiment_label(self, rating):
        """Convert numerical rating to sentiment label.
        
        Args:
            rating: Numerical rating (1-5)
            
        Returns:
            Sentiment label (0: Negative, 1: Neutral, 2: Positive)
        """
        if rating >= 4:
            return 2  # Positive
        elif rating >= 2:
            return 1  # Neutral
        else:
            return 0  # Negative
    
    def prepare_data(self):
        """Preprocess all reviews and create features/labels.
        
        Returns:
            X: List of preprocessed review texts
            y: List of sentiment labels
        """
        X = []
        y = []
        
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Preprocessing data"):
            text = row['review_text']  # Adjust column name if needed
            rating = row['rating']     # Adjust column name if needed
            
            # Preprocess text
            clean_text = self.preprocess_text(text)
            
            # Get sentiment label
            sentiment = self.get_sentiment_label(rating)
            
            X.append(clean_text)
            y.append(sentiment)
        
        return X, y
    
    def split_data(self, X, y, test_size=0.15, val_size=0.15):
        """Split data into training, validation, and test sets.
        
        Args:
            X: List of preprocessed review texts
            y: List of sentiment labels
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split off test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Then split remaining data into train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=42, stratify=y_train_val
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class BaselineModels:
    """Class for implementing baseline models."""
    
    def __init__(self):
        """Initialize baseline models."""
        self.tfidf = TfidfVectorizer(max_features=20000)
        self.log_reg = LogisticRegression(max_iter=1000, solver='lbfgs')
        self.naive_bayes = MultinomialNB(alpha=1.0)
    
    def prepare_features(self, X_train, X_val, X_test):
        """Transform text data to TF-IDF features.
        
        Args:
            X_train, X_val, X_test: Text data for train, validation, and test sets
            
        Returns:
            X_train_tfidf, X_val_tfidf, X_test_tfidf: TF-IDF transformed features
        """
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        X_val_tfidf = self.tfidf.transform(X_val)
        X_test_tfidf = self.tfidf.transform(X_test)
        
        return X_train_tfidf, X_val_tfidf, X_test_tfidf
    
    def train_logistic_regression(self, X_train_tfidf, y_train):
        """Train logistic regression model.
        
        Args:
            X_train_tfidf: TF-IDF features for training
            y_train: Labels for training
        """
        print("Training Logistic Regression model...")
        self.log_reg.fit(X_train_tfidf, y_train)
    
    def train_naive_bayes(self, X_train_tfidf, y_train):
        """Train Naive Bayes model.
        
        Args:
            X_train_tfidf: TF-IDF features for training
            y_train: Labels for training
        """
        print("Training Naive Bayes model...")
        self.naive_bayes.fit(X_train_tfidf, y_train)
    
    def evaluate_model(self, model, X_test_tfidf, y_test):
        """Evaluate model performance.
        
        Args:
            model: Trained model
            X_test_tfidf: TF-IDF features for testing
            y_test: Labels for testing
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred
        }


class TextCNN(nn.Module):
    """Text CNN model for sentiment classification."""
    
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        """Initialize TextCNN model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            n_filters: Number of CNN filters
            filter_sizes: List of filter sizes
            output_dim: Number of output classes
            dropout: Dropout rate
            pad_idx: Padding index
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, 
                      out_channels=n_filters, 
                      kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        """Forward pass through the model.
        
        Args:
            text: Input text tensor
            
        Returns:
            Output predictions
        """
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        
        return self.fc(cat)


class AdvancedModels:
    """Class for implementing advanced models (TextCNN and DistilBERT)."""
    
    def __init__(self):
        """Initialize advanced models and tokenizers."""
        self.tokenizer = None
        self.textcnn = None
        self.distilbert = None
        self.device = DEVICE
        
    def prepare_glove_embeddings(self, vocab, embedding_dim=100):
        """Load GloVe embeddings for TextCNN.
        
        Args:
            vocab: Vocabulary dictionary mapping words to indices
            embedding_dim: Dimension of embeddings
            
        Returns:
            Embedding matrix
        """
        print(f"Loading GloVe embeddings (dim={embedding_dim})...")
        
        # This function would download GloVe embeddings if they don't exist,
        # but for simplicity of this example, we'll just create a mock function
        
        # In a real implementation, you would:
        # 1. Download GloVe embeddings from https://nlp.stanford.edu/projects/glove/
        # 2. Load them into a dictionary
        # 3. Create an embedding matrix for words in your vocabulary
        
        # Mock embedding matrix with random values
        vocab_size = len(vocab)
        embedding_matrix = np.random.rand(vocab_size, embedding_dim)
        
        # Set padding token embedding to zeros
        pad_idx = vocab['<pad>']
        embedding_matrix[pad_idx] = np.zeros(embedding_dim)
        
        return torch.FloatTensor(embedding_matrix)
    
    def build_vocab(self, texts, min_freq=2):
        """Build vocabulary from texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency for words to be included
            
        Returns:
            vocab: Dictionary mapping words to indices
            word_counts: Dictionary of word frequencies
        """
        word_counts = {}
        
        for text in texts:
            for word in text.split():
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1
        
        vocab = {'<pad>': 0, '<unk>': 1}
        idx = 2
        
        for word, count in word_counts.items():
            if count >= min_freq:
                vocab[word] = idx
                idx += 1
        
        return vocab, word_counts
    
    def text_to_indices(self, texts, vocab, max_length):
        """Convert texts to index sequences.
        
        Args:
            texts: List of text strings
            vocab: Vocabulary dictionary
            max_length: Maximum sequence length
            
        Returns:
            List of index sequences
        """
        indices = []
        
        for text in texts:
            words = text.split()
            seq = [vocab.get(word, vocab['<unk>']) for word in words]
            
            # Pad or truncate to max_length
            if len(seq) < max_length:
                seq = seq + [vocab['<pad>']] * (max_length - len(seq))
            else:
                seq = seq[:max_length]
                
            indices.append(seq)
        
        return indices
    
    def train_textcnn(self, X_train, y_train, X_val, y_val, vocab_size, 
                      embedding_dim=100, n_filters=100, filter_sizes=[3, 4, 5], 
                      output_dim=3, dropout=0.5, pad_idx=0):
        """Train TextCNN model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            n_filters: Number of CNN filters
            filter_sizes: List of filter sizes
            output_dim: Number of output classes
            dropout: Dropout rate
            pad_idx: Padding index
            
        Returns:
            Trained TextCNN model
        """
        print("Training TextCNN model...")
        
        # Build model
        self.textcnn = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_filters=n_filters,
            filter_sizes=filter_sizes,
            output_dim=output_dim,
            dropout=dropout,
            pad_idx=pad_idx
        ).to(self.device)
        
        # Prepare data
        X_train_tensor = torch.LongTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.LongTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Calculate class weights for imbalanced dataset
        class_counts = np.bincount(y_train)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(self.textcnn.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(EPOCHS):
            self.textcnn.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                batch_texts, batch_labels = batch
                
                optimizer.zero_grad()
                predictions = self.textcnn(batch_texts)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Update learning rate
            scheduler.step()
            
            # Validation
            self.textcnn.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch_texts, batch_labels = batch
                    
                    predictions = self.textcnn(batch_texts)
                    loss = criterion(predictions, batch_labels)
                    
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(predictions, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            
            # Save best model weights in memory
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.textcnn.state_dict().copy()
                print("Saved best model state.")
        
        # Load best model state
        self.textcnn.load_state_dict(best_model_state)
        return self.textcnn

    def train_distilbert(self, X_train, y_train, X_val, y_val):
        """Train DistilBERT model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Trained DistilBERT model
        """
        print("Training DistilBERT model...")
        
        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', 
            num_labels=3
        ).to(self.device)
        
        # Tokenize data
        train_encodings = self.tokenizer(
            X_train, 
            truncation=True, 
            padding='max_length', 
            max_length=MAX_SEQUENCE_LENGTH, 
            return_tensors='pt'
        )
        
        val_encodings = self.tokenizer(
            X_val, 
            truncation=True, 
            padding='max_length', 
            max_length=MAX_SEQUENCE_LENGTH, 
            return_tensors='pt'
        )
        
        # Create datasets
        class SentimentDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item

            def __len__(self):
                return len(self.labels)
        
        train_dataset = SentimentDataset(train_encodings, torch.tensor(y_train))
        val_dataset = SentimentDataset(val_encodings, torch.tensor(y_val))
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # Calculate class weights for imbalanced dataset
        class_counts = np.bincount(y_train)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(self.device)
        
        # Define optimizer and scheduler
        optimizer = AdamW(self.distilbert.parameters(), lr=LEARNING_RATE)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(EPOCHS):
            self.distilbert.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.distilbert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.distilbert.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.distilbert(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    val_loss += loss.item()
                    
                    logits = outputs.logits
                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            
            # Save best model state in memory
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.distilbert.state_dict().copy()
                print("Saved best model state.")
        
        # Load best model state
        self.distilbert.load_state_dict(best_model_state)
        return self.distilbertd == labels.sum().item()

    def evaluate_textcnn(self, model, X_test, y_test):
        """Evaluate TextCNN model.
        
        Args:
            model: Trained TextCNN model
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        
        X_test_tensor = torch.LongTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch_texts, batch_labels = batch
                
                predictions = model(batch_texts)
                _, predicted = torch.max(predictions, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print("TextCNN Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(all_labels, all_predictions, 
                                    target_names=['Negative', 'Neutral', 'Positive']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': all_predictions
        }
    
    def evaluate_distilbert(self, model, X_test, y_test):
        """Evaluate DistilBERT model.
        
        Args:
            model: Trained DistilBERT model
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        
        # Tokenize test data
        test_encodings = self.tokenizer(
            X_test, 
            truncation=True, 
            padding='max_length', 
            max_length=MAX_SEQUENCE_LENGTH, 
            return_tensors='pt'
        )
        
        # Create dataset
        class SentimentDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item

            def __len__(self):
                return len(self.labels)
        
        test_dataset = SentimentDataset(test_encodings, torch.tensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print("DistilBERT Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(all_labels, all_predictions, 
                                    target_names=['Negative', 'Neutral', 'Positive']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': all_predictions
        }

def generate_dummy_data(output_file, num_samples=1000):
    """Generate dummy data for testing.
    
    Args:
        output_file: Path to save the generated CSV
        num_samples: Number of samples to generate
    """
    print(f"Generating {num_samples} dummy reviews for testing...")
    
    # Sample reviews
    positive_reviews = [
        "Amazing app! Helps me stay active.",
        "Great workout tracking, love the interface.",
        "Best fitness app I've ever used. Highly recommend!",
        "Very helpful for my daily training routine.",
        "Excellent features and user-friendly design."
    ]
    
    neutral_reviews = [
        "It's okay, but the UI needs work.",
        "Decent app, some bugs occasionally.",
        "Works fine for basic tracking.",
        "Average app, nothing special.",
        "Does what it says, but nothing extra."
    ]
    
    negative_reviews = [
        "Terrible app, constant crashes!",
        "Waste of money, doesn't work properly.",
        "Very disappointed, many bugs.",
        "Useless for serious athletes.",
        "Poor design and unreliable tracking."
    ]
    
    # Generate data
    app_names = ['FitTrack', 'NutriLog', 'WorkoutPro', 'DietBuddy', 'FitnessCoach']
    data = []
    
    for _ in range(num_samples):
        rating = np.random.choice(
            [1, 2, 3, 4, 5], 
            p=[0.1, 0.1, 0.1, 0.3, 0.4]  # Class imbalance as in the original dataset
        )
        
        if rating >= 4:
            review = np.random.choice(positive_reviews)
        elif rating >= 2:
            review = np.random.choice(neutral_reviews)
        else:
            review = np.random.choice(negative_reviews)
        
        app_name = np.random.choice(app_names)
        
        data.append({
            'app_name': app_name,
            'rating': rating,
            'review_text': review
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Dummy data saved to {output_file}")
    
    return df


def main():
    """Main function to run the sentiment analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fitness App Review Sentiment Analysis')
    parser.add_argument('--data_path', type=str, default='fitness_app_reviews.csv',
                       help='Path to the dataset CSV file')
    parser.add_argument('--model', type=str, default='all',
                       choices=['logistic_regression', 'naive_bayes', 'textcnn', 'distilbert', 'all'],
                       help='Model to train and evaluate')
    parser.add_argument('--generate_data', action='store_true',
                       help='Generate dummy data for testing')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Generate dummy data if requested
    if args.generate_data:
        generate_dummy_data(args.data_path, args.num_samples)
        print("Dummy data generated. Exiting.")
        return
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    dataset = FitnessAppReviewDataset(args.data_path)
    X, y = dataset.prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.split_data(X, y)
    
    print(f"Data split complete. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train and evaluate models
    if args.model in ['logistic_regression', 'naive_bayes', 'all']:
        # Baseline models
        baseline = BaselineModels()
        X_train_tfidf, X_val_tfidf, X_test_tfidf = baseline.prepare_features(X_train, X_val, X_test)
        
        if args.model in ['logistic_regression', 'all']:
            print("\n===== Logistic Regression =====")
            baseline.train_logistic_regression(X_train_tfidf, y_train)
            lr_metrics = baseline.evaluate_model(baseline.log_reg, X_test_tfidf, y_test)
        
        if args.model in ['naive_bayes', 'all']:
            print("\n===== Naive Bayes =====")
            baseline.train_naive_bayes(X_train_tfidf, y_train)
            nb_metrics = baseline.evaluate_model(baseline.naive_bayes, X_test_tfidf, y_test)
    
    if args.model in ['textcnn', 'distilbert', 'all']:
        # Advanced models
        advanced = AdvancedModels()
        
        if args.model in ['textcnn', 'all']:
            print("\n===== TextCNN =====")
            # Build vocabulary
            vocab, _ = advanced.build_vocab(X_train)
            vocab_size = len(vocab)
            print(f"Vocabulary size: {vocab_size}")
            
            # Convert text to indices
            X_train_indices = advanced.text_to_indices(X_train, vocab, MAX_SEQUENCE_LENGTH)
            X_val_indices = advanced.text_to_indices(X_val, vocab, MAX_SEQUENCE_LENGTH)
            X_test_indices = advanced.text_to_indices(X_test, vocab, MAX_SEQUENCE_LENGTH)
            
            # Train and evaluate TextCNN
            textcnn = advanced.train_textcnn(
                X_train_indices, y_train, 
                X_val_indices, y_val,
                vocab_size=vocab_size
            )
            
            textcnn_metrics = advanced.evaluate_textcnn(textcnn, X_test_indices, y_test)
        
        if args.model in ['distilbert', 'all']:
            print("\n===== DistilBERT =====")
            # Train and evaluate DistilBERT
            distilbert = advanced.train_distilbert(X_train, y_train, X_val, y_val)
            
            distilbert_metrics = advanced.evaluate_distilbert(distilbert, X_test, y_test)
    
    print("\nSentiment Analysis complete!")


if __name__ == "__main__":
    main()