# Fitness App Review Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning project that classifies sentiment in fitness and nutrition app reviews from the Google Play Store. This implementation demonstrates various approaches from simple baselines to advanced deep learning models.

## Overview

This project implements sentiment classification for fitness app reviews with the following features:

- **Data Preprocessing**: Clean and prepare text data for model training
- **Baseline Models**: Logistic Regression and Naive Bayes with TF-IDF features
- **Advanced Models**: TextCNN with GloVe embeddings and DistilBERT fine-tuning
- **Comprehensive Evaluation**: Performance metrics including accuracy, precision, recall, and F1 scores
- **Exploratory Analysis**: Notebook for data visualization and insights

The implementation is based on research by Arnav Khinvasara analyzing user satisfaction with mobile fitness applications.

## Project Structure

```
fitness-app-sentiment/
├── README.md                     # Project documentation
├── requirements.txt              # Required packages
├── sentiment_analysis.py         # Main implementation
├── data/
│   ├── fitness_app_reviews.csv   # Dataset (example or generated)
│   └── glove.6B.100d.txt         # GloVe embeddings (to be downloaded)
├── models/                       # Saved model checkpoints
│   ├── textcnn_model.pt
│   └── distilbert_model.pt
└── notebooks/
    └── exploratory_analysis.ipynb # Data exploration notebook
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fitness-app-sentiment.git
cd fitness-app-sentiment

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download GloVe embeddings (optional, for TextCNN)
mkdir -p data
wget -O data/glove.6B.zip http://nlp.stanford.edu/data/glove.6B.zip
unzip data/glove.6B.zip -d data/
```

## Usage

### Generate Sample Data
```bash
python sentiment_analysis.py --generate_data --num_samples 2000
```

### Train and Evaluate Models

Train and evaluate all models:
```bash
python sentiment_analysis.py --model all
```

Train and evaluate a specific model:
```bash
python sentiment_analysis.py --model logistic_regression
python sentiment_analysis.py --model naive_bayes
python sentiment_analysis.py --model textcnn
python sentiment_analysis.py --model distilbert
```

### Use Your Own Dataset
```bash
python sentiment_analysis.py --data_path path/to/your/dataset.csv
```

## Dataset Format

The CSV dataset should have the following columns:
- `app_name`: Name of the fitness app
- `rating`: Numerical rating (1-5)
- `review_text`: The text of the review

## Results

| Model              | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 84.6%    | 85.0%     | 87.4%  | 85.4%    |
| Naive Bayes        | 83.3%    | 83.3%     | 87.0%  | 83.4%    |
| TextCNN            | 88.2%    | 86.5%     | 88.3%  | 87.5%    |
| DistilBERT         | 87.0%    | 88.0%     | 87.0%  | 88.0%    |

## Implementation Details

### Baseline Models
- **Logistic Regression**: Uses TF-IDF features with a maximum of 20,000 features and the LBFGS solver
- **Naive Bayes**: Uses multinomial Naive Bayes with Laplace smoothing (α = 1)

### Advanced Models
- **TextCNN**: 
  - Pre-trained GloVe embeddings (100-dimensional)
  - Three convolutional filters with kernel sizes of 3, 4, and 5
  - AdamW optimizer with a learning rate scheduler
  - Weighted loss function to handle class imbalance

- **DistilBERT**:
  - Fine-tuned pre-trained DistilBERT model
  - Maximum sequence length of 128 tokens
  - AdamW optimizer with a learning rate of 5e-5
  - 10 training epochs

## Challenges and Future Work

- **Class Imbalance**: The dataset has significantly more positive reviews (70%) than neutral (20%) or negative (10%) ones, affecting model performance.
- **Handling Ambiguity**: Models struggle with reviews containing mixed sentiments or sarcasm.
- **Potential Improvements**:
  - Collect more negative and neutral reviews to balance the dataset
  - Implement more sophisticated techniques for handling mixed sentiments
  - Explore zero-shot or few-shot learning approaches

## References

This project is based on research by Arnav Khinvasara and includes techniques from the following papers:

1. Ahn, H., Park, E. Motivations for user satisfaction of mobile fitness applications: An analysis of user experience based on online review comments. *Humanit Soc Sci Commun* **10**, 3 (2023). https://doi.org/10.1057/s41599-022-01452-6
2. Aslam N, Ramay WY, Xia K et al. (2020) - Convolutional Neural Network Based Classification of App Reviews
3. Hedegaard S, Simonsen JG (2013) - Extracting Usability and User Experience Information from Online User Reviews

## License

MIT