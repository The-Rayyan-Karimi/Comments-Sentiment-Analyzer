#!/usr/bin/env python3
"""
Test script to verify installation and model loading.
Run this before using the main sentiment analyzer.
"""

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy version: {numpy.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if the sentiment analysis model can be loaded."""
    print("\nTesting model loading...")
    
    try:
        from transformers import pipeline
        
        print("Loading sentiment analysis model...")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None  # Get all scores instead of deprecated return_all_scores
        )
        
        print("✅ Model loaded successfully!")
        
        # Test with a sample comment
        test_comment = "I love this!"
        results = sentiment_pipeline(test_comment)
        print(f"✅ Test analysis successful for: '{test_comment}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("INSTALLATION TEST")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test model loading
        model_ok = test_model_loading()
        
        if model_ok:
            print("\n" + "=" * 50)
            print("✅ ALL TESTS PASSED!")
            print("✅ Your installation is ready!")
            print("✅ You can now run: python sentiment_analyzer.py")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("❌ MODEL LOADING FAILED")
            print("❌ Check your internet connection and try again")
            print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ IMPORT TESTS FAILED")
        print("❌ Run: pip install -r requirements.txt")
        print("=" * 50)

if __name__ == "__main__":
    main() 