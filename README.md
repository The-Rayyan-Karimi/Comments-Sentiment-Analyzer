# Comment Sentiment Analyzer

A Python application that analyzes the sentiment of user comments using Hugging Face's pre-trained models. The application allows users to input up to 5 comments and provides sentiment analysis results (Positive, Negative, or Neutral) with confidence scores.

## Features

- **Interactive Input**: Enter up to 5 comments through the command line interface
- **Sentiment Analysis**: Uses the `cardiffnlp/twitter-roberta-base-sentiment-latest` model for accurate sentiment detection
- **Confidence Scores**: Shows confidence levels for each sentiment prediction
- **User-Friendly Interface**: Clean, formatted output with visual indicators
- **Error Handling**: Robust error handling for various edge cases

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The required packages are:
   - `transformers==4.35.0` - Hugging Face transformers library
   - `torch==2.1.0` - PyTorch for deep learning
   - `numpy==1.24.3` - Numerical computing library

## Usage

1. **Run the sentiment analyzer**:
   ```bash
   python sentiment_analyzer.py
   ```

2. **Enter your comments**:
   - You can enter up to 5 comments
   - Press Enter without typing anything to finish early
   - Each comment will be analyzed for sentiment

3. **View results**:
   - The program will display the sentiment (Positive ✅, Negative ❌, or Neutral ➖)
   - Confidence scores are shown as percentages
   - Results are clearly formatted for easy reading

## Example Output

```
==================================================
COMMENT SENTIMENT ANALYZER
==================================================
Enter up to 5 comments to analyze their sentiment.
Press Enter without typing anything to finish early.
==================================================

Enter comment 1: I love this product! It's amazing!
Comment 1 added: 'I love this product! It's amazing!'

Enter comment 2: This is terrible, I hate it.
Comment 2 added: 'This is terrible, I hate it.'

Enter comment 3: 
No comment entered. Moving to analysis...

============================================================
ANALYZING 2 COMMENT(S)
============================================================

Comment 1:
Text: I love this product! It's amazing!
Sentiment: ✅ Positive
Confidence: 95.23%
----------------------------------------

Comment 2:
Text: This is terrible, I hate it.
Sentiment: ❌ Negative
Confidence: 98.45%
----------------------------------------

============================================================
Analysis complete!
============================================================
```

## Model Information

The application uses the `cardiffnlp/twitter-roberta-base-sentiment-latest` model, which is:
- Pre-trained on Twitter data for sentiment analysis
- Optimized for social media and general text sentiment
- Provides three-class classification (Positive, Negative, Neutral)
- Offers high accuracy for sentiment detection

## Requirements

- Python 3.7 or higher
- Internet connection (for downloading the model on first run)
- Sufficient RAM (recommended 4GB+ for model loading)

## Troubleshooting

1. **Model download issues**: Ensure you have a stable internet connection for the first run
2. **Memory issues**: Close other applications if you encounter memory errors
3. **Import errors**: Make sure all dependencies are installed correctly using `pip install -r requirements.txt`

## License

This project is open source and available under the MIT License. #   C o m m e n t s - S e n t i m e n t - A n a l y z e r  
 #   C o m m e n t s - S e n t i m e n t - A n a l y z e r  
 