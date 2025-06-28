# Setup Guide for Comment Sentiment Analyzer

## Prerequisites: Installing Python

Since Python is not installed on your system, follow these steps:

### Option 1: Install from Microsoft Store (Recommended for beginners)
1. Open Microsoft Store
2. Search for "Python 3.11" or "Python 3.12"
3. Click "Get" or "Install"
4. Wait for installation to complete
5. Restart your terminal/PowerShell

### Option 2: Install from Python.org (More control)
1. Go to https://www.python.org/downloads/
2. Download the latest Python version (3.11 or 3.12)
3. Run the installer
4. **IMPORTANT**: Check "Add Python to PATH" during installation
5. Complete the installation
6. Restart your terminal/PowerShell

### Option 3: Install using winget (Command line)
```powershell
winget install Python.Python.3.11
```

## Verifying Python Installation

After installation, verify Python is working:

```powershell
python --version
```

You should see something like: `Python 3.11.x`

## Installing Dependencies

Once Python is installed, install the required packages:

```powershell
pip install -r requirements.txt
```

## Testing the Installation

Run the test script to verify everything works:

```powershell
python test_installation.py
```

## Alternative: Using Python Online

If you prefer not to install Python locally, you can use:

1. **Google Colab**: https://colab.research.google.com/
2. **Jupyter Notebook Online**: https://jupyter.org/try
3. **Replit**: https://replit.com/

## Troubleshooting

### Python not found
- Make sure Python is added to PATH during installation
- Restart your terminal after installation
- Try using `py` instead of `python` on Windows

### pip not found
- Python 3.4+ should include pip by default
- If missing, install pip: `python -m ensurepip --upgrade`

### Permission errors
- Run PowerShell as Administrator
- Or use: `pip install --user -r requirements.txt`

### Model download issues
- Ensure stable internet connection
- The model will download automatically on first run (~500MB)

## Quick Start After Installation

1. Open PowerShell in this directory
2. Run: `python test_installation.py`
3. If successful, run: `python sentiment_analyzer.py`
4. Enter your comments and see the sentiment analysis results! 