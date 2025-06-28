import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re

class CommentSentimentAnalyzer:
    """
    A sentiment analyzer that uses Hugging Face's pre-trained models to classify
    text comments as Positive, Negative, or Neutral.
    
    This class uses the 'cardiffnlp/twitter-roberta-base-sentiment-latest' model,
    which is specifically trained for sentiment analysis on social media text.
    """
    
    def __init__(self):
        """
        Initialize the sentiment analyzer by loading the pre-trained model and tokenizer.
        
        The model is downloaded from Hugging Face Hub on first run and cached locally.
        This model is trained on Twitter data and can classify text into 3 sentiment categories.
        """
        print("Loading sentiment analysis model...")
        
        # Model name from Hugging Face Hub - this is a RoBERTa model fine-tuned for sentiment analysis
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        # Load the tokenizer - converts text into tokens that the model can understand
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load the pre-trained model - this contains the learned weights for sentiment classification
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Define the output labels in the correct order (as per the model's training)
        # LABEL_0 = Negative, LABEL_1 = Neutral, LABEL_2 = Positive
        self.labels = ['Negative', 'Neutral', 'Positive']
        
        print("Model loaded successfully!")

    def preprocess(self, text):
        """
        Preprocess the input text according to the model's requirements.
        
        The CardiffNLP model was trained on Twitter data, so it expects:
        - Lowercase text
        - URLs removed
        - Usernames (@username) removed
        - Hashtags (#) removed
        - Extra whitespace normalized
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Preprocessed text ready for the model
        """
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove URLs (http:// or https:// followed by any non-whitespace characters)
        text = re.sub(r'http\S+', '', text)
        
        # Remove usernames (words starting with @)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtag symbols (keep the word, just remove #)
        text = re.sub(r'#', '', text)
        
        # Normalize whitespace - replace multiple spaces/tabs/newlines with single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of a given text using the loaded model.
        
        This method:
        1. Preprocesses the text
        2. Tokenizes it for the model
        3. Runs inference to get logits
        4. Applies softmax to get probabilities
        5. Returns the sentiment with highest probability
        
        Args:
            text (str): The text to analyze
            
        Returns:
            tuple: (sentiment_label, confidence_score)
                - sentiment_label: 'Positive', 'Negative', or 'Neutral'
                - confidence_score: float between 0 and 1
        """
        try:
            # Step 1: Preprocess the text according to model requirements
            processed_text = self.preprocess(text)
            
            # Step 2: Tokenize the text - convert to model's input format
            # return_tensors='pt' returns PyTorch tensors
            encoded_input = self.tokenizer(processed_text, return_tensors='pt')
            
            # Step 3: Run inference (prediction) with the model
            with torch.no_grad():  # Disable gradient computation for inference (faster)
                # Get model output (logits - raw scores before softmax)
                output = self.model(**encoded_input)
                
                # Extract logits and convert to numpy array
                scores = output.logits[0].numpy()
                
                # Step 4: Apply softmax to convert logits to probabilities
                # Softmax formula: exp(x_i) / sum(exp(x_j)) for all j
                scores = np.exp(scores) / np.sum(np.exp(scores))
                
                # Step 5: Find the sentiment with highest probability
                max_idx = int(np.argmax(scores))
                sentiment = self.labels[max_idx]
                confidence = float(scores[max_idx])
                
                return sentiment, confidence
                
        except Exception as e:
            # Handle any errors gracefully
            print(f"Error analyzing sentiment: {e}")
            return "Neutral", 0.0

    def get_user_comments(self):
        """
        Get up to 5 comments from the user through command line input.
        
        Users can enter fewer than 5 comments by pressing Enter without typing.
        Each comment is stored and a preview is shown.
        
        Returns:
            list: List of user-entered comments
        """
        comments = []
        
        # Display welcome message and instructions
        print("\n" + "="*50)
        print("COMMENT SENTIMENT ANALYZER")
        print("="*50)
        print("Enter up to 5 comments to analyze their sentiment.")
        print("Press Enter without typing anything to finish early.")
        print("="*50)
        
        # Loop to collect up to 5 comments
        for i in range(5):
            # Get comment from user
            comment = input(f"\nEnter comment {i+1}: ").strip()
            
            # If user enters nothing, stop collecting comments
            if not comment:
                print("No comment entered. Moving to analysis...")
                break
                
            # Add comment to list and show confirmation
            comments.append(comment)
            print(f"Comment {i+1} added: '{comment[:50]}{'...' if len(comment) > 50 else ''}'")
        
        return comments

    def analyze_comments(self, comments):
        """
        Analyze the sentiment of all provided comments and display results.
        
        For each comment, this method:
        1. Shows the original text
        2. Calls analyze_sentiment() to get sentiment and confidence
        3. Displays results with visual indicators (emojis)
        
        Args:
            comments (list): List of comments to analyze
        """
        # Check if there are any comments to analyze
        if not comments:
            print("\nNo comments to analyze!")
            return
        
        # Display analysis header
        print(f"\n{'='*60}")
        print(f"ANALYZING {len(comments)} COMMENT(S)")
        print(f"{'='*60}")
        
        # Analyze each comment individually
        for i, comment in enumerate(comments, 1):
            print(f"\nComment {i}:")
            print(f"Text: {comment}")
            
            # Get sentiment analysis results
            sentiment, confidence = self.analyze_sentiment(comment)
            
            # Create visual indicator for sentiment
            if sentiment == "Positive":
                sentiment_display = f"✅ {sentiment}"
            elif sentiment == "Negative":
                sentiment_display = f"❌ {sentiment}"
            else:
                sentiment_display = f"➖ {sentiment}"
            
            # Display results
            print(f"Sentiment: {sentiment_display}")
            print(f"Confidence: {confidence:.2%}")
            print("-" * 40)

    def run(self):
        """
        Main method to run the complete sentiment analysis workflow.
        
        This method:
        1. Gets comments from the user
        2. Analyzes each comment's sentiment
        3. Displays results
        4. Handles any errors gracefully
        """
        try:
            # Step 1: Get comments from user
            comments = self.get_user_comments()
            
            # Step 2: Analyze all comments
            self.analyze_comments(comments)
            
            # Step 3: Show completion message
            print(f"\n{'='*60}")
            print("Analysis complete!")
            print(f"{'='*60}")
            
        except KeyboardInterrupt:
            # Handle user interruption (Ctrl+C)
            print("\n\nProgram interrupted by user.")
        except Exception as e:
            # Handle any other errors
            print(f"\nAn error occurred: {e}")

def main():
    """
    Main function to start the sentiment analyzer application.
    
    Creates an instance of CommentSentimentAnalyzer and runs it.
    """
    analyzer = CommentSentimentAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main() 