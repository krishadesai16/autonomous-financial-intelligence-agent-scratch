# 📈 Autonomous AI Financial & Market Intelligence Agent (From Scratch)

> **Academic Capstone:** Dual Degree B.Tech + MBA (Financial Analytics)

## Project Overview
An autonomous financial market intelligence pipeline built from scratch using PyTorch. The system extracts financial sentiment using a custom Bidirectional LSTM paired with an internal Self-Attention mechanism (zero pre-trained transformers or weights), blends textual signals with quantitative momentum features, estimates 1-Day 95% Value-at-Risk (VaR), and conducts algorithmic strategy backtests against standard benchmarks.

## Key Features
- **NLP from Scratch:** Custom tokenization, vocabulary mapping, and PyTorch BiLSTM + Self-Attention network achieving ~79% accuracy on financial consensus datasets.
- **Quantitative Risk Management (MBA):** Real-time 95% Value-at-Risk (VaR) and annualized volatility scaling to govern portfolio sizing and stop-loss levels.
- **Predictive Machine Learning:** Supervised Random Forest Classifier predicting directional price momentum.
- **Algorithmic Backtester:** 1-year performance simulation tracking Cumulative Equity vs. Buy & Hold, Sharpe Ratio, and Maximum Drawdown.
- **Interactive Web App:** Real-time Gradio dashboard with candlestick charting and sentiment distribution plots.

## Repository Contents
- `Ai_Project_from_Scratch.ipynb`: Full Colab training notebook including vocabulary building, BiLSTM architecture, and backtesting.
- `app.py`: Production-ready Gradio dashboard logic.
- `scratch_model.pt`: Trained PyTorch weights from scratch.
- `vocab.json`: Tokenizer mapping dictionary.
- `requirements.txt`: Environment dependencies.

## Live Deployment
- **Hugging Face Space:** [Live Demo](https://huggingface.co/spaces/desaikrisha16/Autonomous-Financial-AI-Agent)

## Local Setup
```bash
git clone [https://github.com/krishadesai16/autonomous-financial-intelligence-agent-scratch.git](https://github.com/krishadesai16/autonomous-financial-intelligence-agent-scratch.git)
cd autonomous-financial-intelligence-agent-scratch
pip install -r requirements.txt
python app.py
