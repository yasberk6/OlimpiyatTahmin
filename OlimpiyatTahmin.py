import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

df_2016 = pd.read_csv('rio2016.csv')
df_2020 = pd.read_csv('2020.csv')

print("2016 Verileri:")
print(df_2016.head())
print("\n2020 Verileri:")
print(df_2020.head())

X = df_2016[['Gold', 'Silver', 'Bronze']]
y = df_2016['Total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = Sequential()
nn_model.add(Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(1))  # Çıktı katmanı

nn_model.compile(optimizer='adam', loss='mean_squared_error')

history = nn_model.fit(X_train_scaled, y_train, epochs=150, validation_data=(X_test_scaled, y_test), batch_size=32)

plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch Sayısı')
plt.ylabel('Kayıp')
plt.title('Model Kayıp Fonksiyonunun Eğitimi ve Doğrulaması')
plt.legend()
plt.show()

y_pred = nn_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Tahminler ve Gerçek Değerler Karşılaştırması
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Tahminler')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Gerçek Değerler')
plt.xlabel('Gerçek Toplam Madalya Sayısı')
plt.ylabel('Tahmin Edilen Toplam Madalya Sayısı')
plt.title('Gerçek ve Tahmin Edilen Toplam Madalya Sayıları (Eğitim/Test Seti)')
plt.legend()
plt.show()

selected_country = input("Lütfen bir ülkenin NOC kodunu giriniz (örneğin: USA): ").strip()

# Seçilen ülkenin 2016 verilerini alma
selected_data_2016 = df_2016[df_2016['NOC'] == selected_country]
selected_data_2020 = df_2020[df_2020['NOC'] == selected_country]

if not selected_data_2016.empty:
    X_selected = selected_data_2016[['Gold', 'Silver', 'Bronze']]
    X_selected_scaled = scaler.transform(X_selected)

    # Tahmin yapma
    y_pred_2020 = nn_model.predict(X_selected_scaled)

    print(f"{selected_country} ülkesinin 2020 Olimpiyatlarında tahmin edilen toplam madalya sayısı: {y_pred_2020[0][0]}")

    # Gerçek 2020 verisi varsa karşılaştırma yapma
    if not selected_data_2020.empty:
        actual_total = selected_data_2020['Total'].values[0]
        print(f"{selected_country} ülkesinin 2020 Olimpiyatlarındaki gerçek toplam madalya sayısı: {actual_total}")
        
        # Tahmin ve gerçek değerleri karşılaştırma
        plt.figure(figsize=(10, 6))
        plt.bar(['Tahmin', 'Gerçek'], [y_pred_2020[0][0], actual_total], color=['blue', 'orange'])
        plt.ylabel('Toplam Madalya Sayısı')
        plt.title(f"{selected_country} için 2020 Olimpiyat Tahmini ve Gerçek Değer")
        plt.show()
    else:
        print(f"{selected_country} kodlu 2020 verisi bulunamadı.")
else:
    print(f"{selected_country} kodlu 2016 verisi bulunamadı.")
