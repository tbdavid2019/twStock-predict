---
title: TwStock Predict
emoji: 🏃
colorFrom: pink
colorTo: blue
sdk: gradio
sdk_version: 5.7.1
app_file: app.py
pinned: false
short_description: 台灣股價預設
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


---


# Taiwan Stock Price Prediction
台股預測系統 - 使用機器學習模型預測台灣股市走勢

## 專案簡介
這是一個基於 Gradio 開發的台股預測網頁應用程式,提供以下功能:
- 即時抓取 Yahoo 財經台股資料
- 依產業類別、類股分類瀏覽
- 使用 LSTM 模型進行股價預測
- 互動式股價預測視覺化圖表
- 支援多種技術指標選擇



## 安裝方式
```bash
# Clone 專案
git clone https://github.com/tbdavid2019/twStock-predict.git

# 進入專案目錄
cd twStock-predict

# 安裝相依套件
pip install -r requirements.txt
```

## 使用方法
1. 執行應用程式
```bash
python app.py
```

2. 開啟瀏覽器訪問 http://localhost:7860

3. 操作步驟:
   - 選擇產業類別
   - 選擇類股
   - 選擇個股
   - 設定預測參數(時間範圍、技術指標)
   - 點擊「開始預測」進行預測

## 功能特色
- 即時爬取 Yahoo 財經最新股市資訊
- 支援多種技術指標選擇 (開盤價、收盤價、最高價、最低價等)
- 使用 LSTM 深度學習模型進行預測
- 預測結果視覺化呈現
- 支援不同時間範圍的歷史資料分析
- 使用者友善的網頁介面

## 系統需求
- Python 3.8 或以上版本
- 穩定的網路連線
- 至少 4GB RAM
- 支援 CUDA 的 GPU (選配,可加速模型訓練)

## 專案結構
```
twStock-predict/
│
├── app.py              # 主程式
├── requirements.txt    # 相依套件清單
├── README.md
└── .gitignore
```

## 模型說明
使用 LSTM (Long Short-Term Memory) 神經網路模型:
- 雙層 LSTM 架構
- Dropout 層防止過擬合
- Adam 優化器
- MSE 損失函數
- 預設訓練 50 個 epochs

## 注意事項
- 股市預測結果僅供參考,不構成投資建議
- 建議使用較長時間範圍的數據來提高預測準確度
- 系統需要穩定的網路連線以獲取即時數據

## License
MIT License

## 作者
[tbdavid2019](https://github.com/tbdavid2019)

