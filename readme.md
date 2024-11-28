# Stock Prediction Application

This project is a stock prediction application developed using Gradio, TensorFlow, and Plotly, with integration of Yahoo Finance for stock data fetching. It allows users to predict future stock prices by selecting various stock features, periods, and categories in the Taiwanese market. The application is designed to be deployed on Hugging Face Spaces, providing a user-friendly interface for non-technical users.

### Features
- **Real-time Stock Data**: Retrieve stock data directly from Yahoo Finance.
- **Customizable Prediction Features**: Users can select different features (e.g., Open, High, Low, Close, Volume) for prediction.
- **Dynamic Charting**: The stock prices are displayed with interactive charts using Plotly.
- **Flexible Data Range**: Users can select different data ranges (e.g., 1 year, 6 months, 3 months, 1 month).
- **Taiwanese Stock Categories**: Extract and analyze Taiwanese stock categories to help users gain insights.

### Installation
To run this project locally, you need to have Python installed and the required dependencies. You can install the dependencies using the following command:

```sh
pip install -r requirements.txt
```

### Run the Application
To launch the application, run the following command:

```sh
python app.py
```

### Deploy to Hugging Face Spaces
This application can be deployed to Hugging Face Spaces by pushing the project files to your Hugging Face repository.

### License
This project is open-sourced under the MIT license.

---

# 股票預測應用程序

這個項目是一個使用 Gradio、TensorFlow 和 Plotly 開發的股票預測應用程序，集成了 Yahoo Finance 以抓取股票數據。它允許用戶通過選擇各種股票特徵、時間範圍和台股類別來預測未來的股票價格。該應用旨在部署到 Hugging Face Spaces，為非技術用戶提供友好的界面。

### 功能特點
- **即時股票數據**：直接從 Yahoo Finance 獲取股票數據。
- **可自定義預測特徵**：用戶可以選擇不同的特徵（如 開盤價、最高價、最低價、收盤價、成交量）進行預測。
- **動態圖表顯示**：使用 Plotly 提供互動式的股價圖表顯示。
- **靈活的數據範圍選擇**：用戶可以選擇不同的數據範圍（如 1年、半年、3個月、1個月）。
- **台股類別分析**：提取並分析台灣股票類別，幫助用戶獲得更多見解。

### 安裝
要在本地運行此項目，您需要安裝 Python 和必要的依賴項。您可以使用以下命令安裝依賴項：

```sh
pip install -r requirements.txt
```

### 運行應用程序
運行以下命令以啟動應用程序：

```sh
python app.py
```

### 部署到 Hugging Face Spaces
您可以將這個應用程序部署到 Hugging Face Spaces，只需將項目文件推送到您的 Hugging Face 倉庫即可。

### 授權協議
此項目以 MIT 許可證開源。

