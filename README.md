# Regime Classifier App

## Overview
This project is a Streamlit-based financial regime classification tool. It analyzes market data and identifies different regimes using machine learning techniques.

## Features
- Interactive UI built with Streamlit
- Financial data fetching using yfinance
- Data processing with pandas and numpy
- Visualization with Plotly
- Machine learning models (scikit-learn, hmmlearn)
- Optional deep learning support with PyTorch

## Project Structure
- app.py → Main application entry point
- ui.py → UI components and charting functions

## Installation

1. Clone or download the project
2. Open terminal in project folder
3. Install dependencies:

```
pip install -r requirements.txt
```

## Running the App

```
streamlit run app.py
```

Then open the browser at:
http://localhost:8501

## Common Issues

### ModuleNotFoundError
Make sure all dependencies are installed.

### Streamlit not recognized
```
pip install streamlit
```

## Notes
- Ensure both app.py and ui.py are in the same directory
- This is a web app, not a command-line script
