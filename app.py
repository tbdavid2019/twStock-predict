import gradio as gr
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import io
import matplotlib as mpl
import matplotlib.font_manager as fm
import tempfile
import os
import yfinance as yf
import logging
from datetime import datetime, timedelta

# 設置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# 字體設置
def setup_font():
    try:
        url_font = "https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_"
        response_font = requests.get(url_font)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as tmp_file:
            tmp_file.write(response_font.content)
            tmp_file_path = tmp_file.name
        
        fm.fontManager.addfont(tmp_file_path)
        mpl.rc('font', family='Taipei Sans TC Beta')
    except Exception as e:
        logging.error(f"字體設置失敗: {str(e)}")
        # 使用備用字體
        mpl.rc('font', family='SimHei')

# 網路請求設置
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

def fetch_stock_categories():
    try:
        url = "https://tw.stock.yahoo.com/class/"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        main_categories = soup.find_all('div', class_='C($c-link-text)')
        
        data = []
        for category in main_categories:
            main_category_name = category.find('h2', class_="Fw(b) Fz(24px) Lh(32px)")
            if main_category_name:
                main_category_name = main_category_name.text.strip()
                sub_categories = category.find_all('a', class_='Fz(16px) Lh(1.5) C($c-link-text) C($c-active-text):h Fw(b):h Td(n)')
                
                for sub_category in sub_categories:
                    data.append({
                        '台股': main_category_name,
                        '類股': sub_category.text.strip(),
                        '網址': "https://tw.stock.yahoo.com" + sub_category['href']
                    })
        
        category_dict = {}
        for item in data:
            if item['台股'] not in category_dict:
                category_dict[item['台股']] = []
            category_dict[item['台股']].append({'類股': item['類股'], '網址': item['網址']})
        
        return category_dict
    except Exception as e:
        logging.error(f"獲取股票類別失敗: {str(e)}")
        return {}

# 股票預測模型類別保持不變...
class StockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, df):
        features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        scaled_data = self.scaler.fit_transform(df[features])
        
        X, y = [], []
        for i in range(len(scaled_data) - 1):
            X.append(scaled_data[i])
            y.append(scaled_data[i+1, [0, 3]])  # Open和Close的索引
        
        return np.array(X).reshape(-1, 1, len(features)), np.array(y)
    
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(100, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(2)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train(self, df):
        X, y = self.prepare_data(df)
        self.model = self.build_model((1, X.shape[2]))
        history = self.model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        return history
    
    def predict(self, last_data, n_days):
        predictions = []
        current_data = last_data.copy()
        
        for _ in range(n_days):
            next_day = self.model.predict(current_data.reshape(1, 1, -1), verbose=0)
            predictions.append(next_day[0])
            
            current_data = current_data.flatten()
            current_data[0] = next_day[0][0]
            current_data[3] = next_day[0][1]
            current_data = current_data.reshape(1, -1)
        
        return np.array(predictions)

# Gradio界面函數
def update_stocks(category):
    if not category or category not in category_dict:
        return []
    return [item['類股'] for item in category_dict[category]]

def get_stock_items(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        stock_items = soup.find_all('li', class_='List(n)')
        
        stocks_dict = {}
        for item in stock_items:
            stock_name = item.find('div', class_='Lh(20px) Fw(600) Fz(16px) Ell')
            stock_code = item.find('span', class_='Fz(14px) C(#979ba7) Ell')
            if stock_name and stock_code:
                full_code = stock_code.text.strip()
                display_code = full_code.split('.')[0]
                display_name = f"{stock_name.text.strip()}{display_code}"
                stocks_dict[display_name] = full_code
        
        return stocks_dict
    except Exception as e:
        logging.error(f"獲取股票項目失敗: {str(e)}")
        return {}

def update_category(category):
    stocks = update_stocks(category)
    return {
        stock_dropdown: gr.update(choices=stocks, value=None),
        stock_item_dropdown: gr.update(choices=[], value=None),
        stock_plot: gr.update(value=None)
    }

def update_stock(category, stock):
    if not category or not stock:
        return {
            stock_item_dropdown: gr.update(choices=[], value=None),
            stock_plot: gr.update(value=None)
        }
    
    url = next((item['網址'] for item in category_dict.get(category, [])
                if item['類股'] == stock), None)
    
    if url:
        stock_items = get_stock_items(url)
        return {
            stock_item_dropdown: gr.update(choices=list(stock_items.keys()), value=None),
            stock_plot: gr.update(value=None)
        }
    return {
        stock_item_dropdown: gr.update(choices=[], value=None),
        stock_plot: gr.update(value=None)
    }

def predict_stock(category, stock, stock_item):
    if not all([category, stock, stock_item]):
        return gr.update(value=None)
    
    try:
        url = next((item['網址'] for item in category_dict.get(category, [])
                   if item['類股'] == stock), None)
        if not url:
            return gr.update(value=None)
        
        stock_items = get_stock_items(url)
        stock_code = stock_items.get(stock_item, "")
        
        if not stock_code:
            return gr.update(value=None)
        
        # 下載股票數據
        df = yf.download(stock_code, period="1y")
        if df.empty:
            raise ValueError("無法獲取股票數據")
        
        # 預測
        predictor = StockPredictor()
        predictor.train(df)
        
        last_data = predictor.scaler.transform(df.iloc[-1:][['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
        predictions = predictor.predict(last_data[0], 5)
        
        # 反轉預測結果
        last_original = df[['Open', 'Close']].iloc[-1].values
        predictions_original = predictor.scaler.inverse_transform(
            np.hstack([predictions, np.zeros((predictions.shape[0], 4))])
        )[:, :2]
        all_predictions = np.vstack([last_original, predictions_original])
        
        # 創建日期指標
        dates = [datetime.now() + timedelta(days=i) for i in range(6)]
        date_labels = [d.strftime('%m/%d') for d in dates]
        
        # 繪圖
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = ['#FF9999', '#66B2FF']
        labels = ['預測開盤價', '預測收盤價']
        
        for i, (col, label, color) in enumerate(zip(['Open', 'Close'], labels, colors)):
            ax.plot(date_labels, all_predictions[:, i], label=label,
                   marker='o', color=color, linewidth=2)
            for j, value in enumerate(all_predictions[:, i]):
                ax.annotate(f'{value:.2f}', (date_labels[j], value),
                           textcoords="offset points", xytext=(0,10),
                           ha='center', va='bottom')
        
        ax.set_title(f'{stock_item} 股價預測 (未來5天)', pad=20, fontsize=14)
        ax.set_xlabel('日期', labelpad=10)
        ax.set_ylabel('股價', labelpad=10)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return gr.update(value=fig)
        
    except Exception as e:
        logging.error(f"預測過程發生錯誤: {str(e)}")
        return gr.update(value=None)

# 初始化
setup_font()
category_dict = fetch_stock_categories()
categories = list(category_dict.keys())

# Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 台股預測系統")
    with gr.Row():
        with gr.Column():
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="產業類別",
                value=None
            )
            stock_dropdown = gr.Dropdown(
                choices=[],
                label="類股",
                value=None
            )
            stock_item_dropdown = gr.Dropdown(
                choices=[],
                label="股票",
                value=None
            )
            predict_button = gr.Button("開始預測", variant="primary")
    
    with gr.Row():
        stock_plot = gr.Plot(label="股價預測圖")
    
    # 事件綁定
    category_dropdown.change(
        update_category,
        inputs=[category_dropdown],
        outputs=[stock_dropdown, stock_item_dropdown, stock_plot]
    )
    
    stock_dropdown.change(
        update_stock,
        inputs=[category_dropdown, stock_dropdown],
        outputs=[stock_item_dropdown, stock_plot]
    )
    
    predict_button.click(
        predict_stock,
        inputs=[category_dropdown, stock_dropdown, stock_item_dropdown],
        outputs=[stock_plot]
    )

# 啟動應用
if __name__ == "__main__":
    demo.launch(share=False)

