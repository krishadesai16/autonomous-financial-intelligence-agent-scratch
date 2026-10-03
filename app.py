import os
os.environ["GRADIO_SSR_MODE"] = "false"

import json
import re
import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
import plotly.graph_objects as go
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")

# --- 1. Model & Tokenizer Definitions ---
class ScratchTokenizer:
    def __init__(self, vocab_file="vocab.json", max_seq_len=40):
        self.max_seq_len = max_seq_len
        with open(vocab_file, "r") as f:
            self.word2idx = json.load(f)
            
    def clean(self, text):
        return re.sub(r'[^a-z0-9\s]', ' ', str(text).lower()).split()

    def encode(self, text):
        ids = [self.word2idx.get(w, 1) for w in self.clean(text)]
        if len(ids) < self.max_seq_len:
            ids += [0] * (self.max_seq_len - len(ids))
        return ids[:self.max_seq_len]

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.att = nn.Linear(dim, 1)
        
    def forward(self, x):
        w = F.softmax(torch.tanh(self.att(x)), dim=1)
        return torch.sum(w * x, dim=1)

class FinancialLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=64, output_dim=3, pad_idx=0):
        super(FinancialLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_drop = nn.Dropout(0.3)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_dim * 2)
        self.fc_drop = nn.Dropout(0.55)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.emb_drop(self.emb(x))
        out, _ = self.lstm(x)
        ctx = self.attn(out)
        return self.fc(self.fc_drop(ctx))

# Load Tokenizer & Saved Scratch Weights
tokenizer = ScratchTokenizer("vocab.json")
model = FinancialLSTM(vocab_size=len(tokenizer.word2idx), emb_dim=64, hidden_dim=64, output_dim=3).to(device)
model.load_state_dict(torch.load("scratch_model.pt", map_location=device))
model.eval()

# --- 2. Analytics & Inference Engines ---
def predict_scratch_sentiment(headlines):
    if not headlines:
        return 0.0, {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
    encoded_list = [tokenizer.encode(h) for h in headlines]
    tensor_input = torch.tensor(encoded_list, dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
    neg = float(probs[:, 0].mean())
    neu = float(probs[:, 1].mean())
    pos = float(probs[:, 2].mean())
    return (pos - neg), {"positive": pos, "neutral": neu, "negative": neg}

def get_market_data(ticker="AAPL", period="2y"):
    t = yf.Ticker(ticker)
    df = t.history(period=period)
    if df.empty:
        raise ValueError(f"No price data found for {ticker}")
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()

def train_market_classifier(df):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    prob_up = clf.predict_proba(X.iloc[[-1]])[0][1]
    return acc, prob_up

def compute_risk_metrics(df):
    daily_returns = df['Close'].pct_change().dropna()
    mean_ret = daily_returns.mean()
    std_dev = daily_returns.std()
    var_95 = -(mean_ret - 1.645 * std_dev) * 100
    ann_volatility = (std_dev * np.sqrt(252)) * 100
    return round(var_95, 2), round(ann_volatility, 2)

def run_strategy_backtest(df):
    backtest_df = df.copy()
    backtest_df['Market_Return'] = backtest_df['Close'].pct_change().fillna(0)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI']
    X = backtest_df[features]
    y = backtest_df['Target']
    split_idx = int(len(backtest_df) * 0.4)
    train_X, test_X = X.iloc[:split_idx], X.iloc[split_idx:]
    train_y = y.iloc[:split_idx]
    
    clf = RandomForestClassifier(n_estimators=80, max_depth=4, random_state=42)
    clf.fit(train_X, train_y)
    
    test_signals = clf.predict(test_X)
    sim_df = backtest_df.iloc[split_idx:].copy()
    sim_df['Signal'] = test_signals
    sim_df['Strategy_Return'] = sim_df['Market_Return'] * sim_df['Signal'].shift(1).fillna(0)
    
    initial_capital = 10000.0
    sim_df['Buy_Hold_Equity'] = initial_capital * (1 + sim_df['Market_Return']).cumprod()
    sim_df['AI_Strategy_Equity'] = initial_capital * (1 + sim_df['Strategy_Return']).cumprod()
    
    strat_ret = ((sim_df['AI_Strategy_Equity'].iloc[-1] / initial_capital) - 1) * 100
    mkt_ret = ((sim_df['Buy_Hold_Equity'].iloc[-1] / initial_capital) - 1) * 100
    
    rf_daily = 0.03 / 252
    excess = sim_df['Strategy_Return'] - rf_daily
    sharpe = np.sqrt(252) * (excess.mean() / (sim_df['Strategy_Return'].std() + 1e-9))
    
    drawdown = (sim_df['AI_Strategy_Equity'] - sim_df['AI_Strategy_Equity'].cummax()) / sim_df['AI_Strategy_Equity'].cummax()
    mdd = drawdown.min() * 100
    
    return sim_df, {
        "strat_return": round(strat_ret, 2),
        "mkt_return": round(mkt_ret, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(mdd, 2)
    }

def get_recent_news(ticker="AAPL"):
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(rss_url)
    headlines = [e.title for e in feed.entries[:6]]
    if not headlines:
        headlines = [
            f"{ticker} demonstrates steady market operations across units",
            f"Financial analysts evaluate quarterly performance forecasts for {ticker}",
            f"Index movements steady ahead of sector earnings releases"
        ]
    return headlines

def generate_agent_strategy(sentiment_score, prob_up, rsi, var_95, vol):
    norm_sent = (sentiment_score + 1.0) / 2.0
    composite = (0.50 * prob_up) + (0.50 * norm_sent)
    base_sizing = 8.0 if composite >= 0.58 else (0.0 if composite <= 0.42 else 3.0)
    if vol > 40.0:
        base_sizing *= 0.6
        
    if composite >= 0.58:
        rec = "ACCUMULATE (BUY)"
        action = f"Deploy {base_sizing:.1f}% capital tranche. Set stop-loss at {var_95:.1f}%."
        risk = "MODERATE" if rsi < 70 else "HIGH (RSI Overbought Alert)"
    elif composite <= 0.42:
        rec = "REDUCE EXPOSURE (SELL)"
        action = "Trim exposure and preserve liquidity in cash."
        risk = "ELEVATED DOWNSIDE VOLATILITY"
    else:
        rec = "HOLD / NEUTRAL"
        action = "Preserve existing baseline weight."
        risk = "BALANCED"

    confidence = round(abs(composite - 0.50) * 200, 1)
    brief = (
        f"=== EXECUTIVE STRATEGY MEMORANDUM ===\n"
        f"Recommendation   : {rec}\n"
        f"Conviction Score : {confidence}%\n"
        f"Risk Rating      : {risk}\n"
        f"Allocation Size  : {base_sizing:.1f}% Capital\n\n"
        f"MBA Risk Analytics:\n"
        f"• 1-Day 95% VaR  : {var_95:.2f}%\n"
        f"• Annual Volatility: {vol:.2f}%\n"
        f"• 14-Day RSI     : {rsi:.1f}\n\n"
        f"Operational Plan : {action}"
    )
    return rec, f"{confidence}%", brief

# --- 3. Gradio Interface Construction ---
POPULAR_COMPANIES = [
    ("Apple Inc. (AAPL)", "AAPL"),
    ("NVIDIA Corporation (NVDA)", "NVDA"),
    ("Microsoft Corporation (MSFT)", "MSFT"),
    ("Tesla, Inc. (TSLA)", "TSLA"),
    ("Alphabet Inc. / Google (GOOGL)", "GOOGL"),
    ("Amazon.com, Inc. (AMZN)", "AMZN"),
    ("Meta Platforms, Inc. (META)", "META"),
    ("JPMorgan Chase (JPM)", "JPM"),
    ("Reliance Industries (RELIANCE.NS)", "RELIANCE.NS"),
    ("Tata Consultancy Services (TCS.NS)", "TCS.NS"),
    ("HDFC Bank (HDFCBANK.NS)", "HDFCBANK.NS"),
    ("Infosys Ltd (INFY.NS)", "INFY.NS"),
    ("S&P 500 ETF (SPY)", "SPY")
]

def run_agent(ticker_choice):
    ticker = (ticker_choice or "AAPL").strip().upper()
    try:
        df = get_market_data(ticker, period="2y")
        headlines = get_recent_news(ticker)
        compound_score, sent_dist = predict_scratch_sentiment(headlines)
        acc, prob_up = train_market_classifier(df)
        var_95, vol = compute_risk_metrics(df)
        rec, conf, brief = generate_agent_strategy(compound_score, prob_up, df['RSI'].iloc[-1], var_95, vol)
        sim_df, btest_kpi = run_strategy_backtest(df)
        
        p_fig = go.Figure()
        p_fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
        p_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1.5), name='SMA 20'))
        p_fig.update_layout(title=f"{ticker} Candlestick & Moving Average", template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=40, b=20))
        
        b_fig = go.Figure()
        b_fig.add_trace(go.Scatter(x=sim_df.index, y=sim_df['AI_Strategy_Equity'], mode='lines', name='AI Strategy ($)', line=dict(color='#00CC96', width=2.5)))
        b_fig.add_trace(go.Scatter(x=sim_df.index, y=sim_df['Buy_Hold_Equity'], mode='lines', name='Buy & Hold ($)', line=dict(color='#AB63FA', width=1.8, dash='dash')))
        b_fig.update_layout(title="Historical Backtest ($10K Starting Capital)", template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
        
        s_fig = go.Figure(data=[go.Pie(labels=['Bullish', 'Neutral', 'Bearish'], values=[sent_dist['positive'], sent_dist['neutral'], sent_dist['negative']], hole=.45, marker=dict(colors=['#00CC96', '#636EFA', '#EF553B']))])
        s_fig.update_layout(title="BiLSTM News Sentiment Distribution", template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
        
        kpis = (
            f"Strategy Return : {btest_kpi['strat_return']}%\n"
            f"Buy & Hold Return: {btest_kpi['mkt_return']}%\n"
            f"Sharpe Ratio    : {btest_kpi['sharpe_ratio']}\n"
            f"Max Drawdown    : {btest_kpi['max_drawdown']}%\n"
            f"1-Day 95% VaR   : {var_95:.2f}%\n"
            f"Annualized Vol  : {vol:.2f}%"
        )
        news_text = "\n".join([f"• {h}" for h in headlines])
        return rec, conf, kpis, brief, p_fig, b_fig, s_fig, news_text
    except Exception as e:
        empty = go.Figure()
        return "ERROR", "0%", "N/A", f"Error: {str(e)}", empty, empty, empty, "N/A"

with gr.Blocks() as demo:
    gr.Markdown("# 📈 Autonomous AI Financial & Market Intelligence Agent")
    gr.Markdown("**Dual BTech + MBA System** — Custom Scratch BiLSTM NLP + Quantitative Risk (VaR) Engine.")
    with gr.Row():
        with gr.Column(scale=1):
            comp_dd = gr.Dropdown(choices=POPULAR_COMPANIES, value="AAPL", label="Select Company / Market Ticker", allow_custom_value=True)
            run_btn = gr.Button("Execute Market Intelligence Agent", variant="primary")
            sig_box = gr.Textbox(label="Agent Signal")
            conf_box = gr.Textbox(label="Conviction Score")
            kpi_box = gr.Textbox(label="Quantitative Risk & Backtest KPIs", lines=7)
            news_box = gr.Textbox(label="Live News Ingested by BiLSTM", lines=4)
        with gr.Column(scale=2):
            price_plt = gr.Plot(label="Price Action & Moving Average")
            backtest_plt = gr.Plot(label="Backtest Performance Curve")
            sent_plt = gr.Plot(label="Sentiment Distribution")
            brief_box = gr.Textbox(label="Executive Investment Memorandum (MBA)", lines=7)
            
    run_btn.click(fn=run_agent, inputs=[comp_dd], outputs=[sig_box, conf_box, kpi_box, brief_box, price_plt, backtest_plt, sent_plt, news_box])
    comp_dd.change(fn=run_agent, inputs=[comp_dd], outputs=[sig_box, conf_box, kpi_box, brief_box, price_plt, backtest_plt, sent_plt, news_box])

demo.launch()
