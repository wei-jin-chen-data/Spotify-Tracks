import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split df = pd.read_csv('dataset.csv', encoding='latin1')
#encoding='latin1'文字編碼設定，因歌名裡有德文、法文....，utf-8可能會報錯

df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
#將預測目標popularity轉換成數字型態，errors='coerce'FLASE值先轉換成NaN，後續用dropna()清掉

df['is_popular'] = (df['popularity'] > 50).astype(int)
#將popularity裡面數值大於50還是小於50，回傳True(1)或是Flase(0)
#目標是預測這首歌會不會紅?

feature_cols = [
    'danceability', 'energy', 'key', 'loudness', 'mode',

    'speechiness', 'acousticness', 'instrumentalness',

    'liveness', 'valence', 'tempo', 'duration_ms'
]
#12個特徵欄位存入變數feature_cols中(list)

for col in feature_cols:
    df_clean.loc[:, col] = pd.to_numeric(df_clean[col], errors='coerce')
#依序拿取feature_cols裡的元素
#修改df_clean所有列和指定欄位，並且轉成數字型態，FLASE值先轉換成NaN

df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
#計算欄位平均值並且替換掉NaN，不替換掉Keras遇到NaN會無法計算

X = df_clean[feature_cols]
y = df_clean['is_popular']
#分成特徵資料(X)和預測目標(y) print(df['is_popular'].value_counts(normalize=True))
#發現有類別不平衡的問題，非熱門75%，熱門25% X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#test_size=0.2留下20%確認model訓練成效
#random_state=42，確保切出來都是80%和20%
#stratify=y，確保取出資料熱門和非熱門，保持25%和75% current_X_for_rf = df_clean[feature_cols]

current_y_for_rf = df_clean['is_popular']
X_train_rf, _, y_train_rf, _ = train_test_split(
    current_X_for_rf, current_y_for_rf, test_size=0.2, random_state=42, stratify=current_y_for_rf
)
#訓練model階段不需要X_test和y_test，用_讓它不要回傳

for col in X_train_rf.columns:
    X_train_rf.loc[:, col] = pd.to_numeric(X_train_rf[col], errors='coerce')
X_train_rf = X_train_rf.fillna(X_train_rf.mean(numeric_only=True))
#確保每一欄都轉為數字型態，並且用平均數替換掉NaN
#為了確保隨機森林運作時不會報錯而停止(保險)

rf = RandomForestClassifier( #建立隨機森林
    n_estimators=100,    #建立100顆決策樹
    random_state=42,    #確保切出來都是80%和20%
    n_jobs=-1        #加速運算
)
rf.fit(X_train_rf, y_train_rf)
#100顆樹開始學習

importances = rf.feature_importances_
#回傳計算好的特徵重要性

feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print(" 各特徵重要性排名與得分：")
print(feature_importance_df.to_string(index=False))
#打包成兩欄，並且遞減排列，隱藏index plt.figure(figsize=(10, 6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df,
    palette='viridis'
)
plt.title('Spotify Audio Features - Random Forest Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Audio Features')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
#利用seaborn作圖 TOP_K = 8

selected_rf_features = feature_importance_df['Feature'].head(TOP_K).tolist()
print("根據隨機森林特徵重要性，篩選出的核心特徵為：")
print(selected_rf_features)
#選出前8個重要的指標 import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report X = df_clean[selected_rf_features]

y = df_clean['is_popular']
#取出重要指標(X)，取出預測目標(y) scaler = StandardScaler()
#建立標準化，因為直接餵給model，數值大的會產生大的梯度，數值小的特徵就會被忽視

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#fit只能對X_train_scaled做，計算出平均值和標準差，利用這個數值去對X_test_scaled做轉換
#比方說train平均是0.6，標準差是0.1，X_test_scaled來了一個數值0.7，(0.7-0.6)/0,1=+1，比train時高出一個標準差 model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
#relu，將負值歸零，正值保留
#activation='sigmoid'將輸出值壓縮到0到1之間的機率值

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
#loss='binary_crossentropy專門計算模型預測誤差(0到1)與真實數值誤差

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#達到最低的val_loss後，後續10輪並未創新低，觸發 EarlyStopping
#monitor='val_loss'監控驗證的loss，連續10輪都沒有降低(代表model背答案)，就停止，restore_best_weights=True自動回到val_loss最低的那輪

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)
#callbacks=[early_stop]上面定義的煞車器
#verbose=1，顯示詳細的訓練進度條與紀錄

test_loss, test_acc, test_auc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f" 測試集 Accuracy: {test_acc:.4f}")
print(f" 測試集 AUC: {test_auc:.4f}")
# 評估模型結果
