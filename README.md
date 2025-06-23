# 💻 Laptop Price Prediction with Machine Learning

Bu proje, **Yapay Zeka Uygulamaları** dersi kapsamında gerçekleştirilmiştir. Amacımız, dizüstü bilgisayarların donanım özelliklerini analiz ederek fiyatlarını tahmin eden bir makine öğrenimi modeli oluşturmaktır.
```
🔗 [Web Siteme Bakmak İçin Tıkla](https://www.hakkiharmankaya.com/)
```
## 📂 Veri Seti

Kullanılan veri seti `laptop_data.csv` dosyasıdır ve 1303 farklı ürün kaydı içermektedir. Her kayıtta marka, model, RAM, işlemci tipi, ekran çözünürlüğü, ekran kartı ve depolama gibi özellikler yer almaktadır.

## 🔧 Kullanılan Kütüphaneler

```python
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
```

## 📊 Veri Analizi ve Görselleştirme

- `df.info()`, `df.describe()` ve `df.value_counts()` ile temel istatistiksel bilgiler elde edilmiştir.
- Çubuk grafikler ile marka bazlı ürün sayısı görselleştirilmiştir.
- Eksik veri analizi yapılmış ve eksiksiz veri elde edilmiştir.
- `CPU`, `Memory`, `GPU`, `OpSys`, `Ram`, `Weight` gibi alanlar detaylı şekilde ayrıştırılmış ve dönüştürülmüştür.

## 🧠 Özellik Mühendisliği

- `Cpu` sütunu marka, seri, model ve GHz olarak ayrılmıştır.
- `Memory` sütunu kapasite ve tür (SSD, HDD vs.) olarak ayrılmıştır.
- `Gpu` sütunundan marka ve tip (entegre/harici) çıkarılmıştır.
- One-hot encoding işlemleriyle kategorik veriler modele uygun hale getirilmiştir.
- Popüler CPU modelleri kategorilere ayrılmıştır: **High**, **Medium**, **Low**.

## 🤖 Uygulanan Modeller

Aşağıdaki regresyon algoritmaları kullanılmış ve test skorları karşılaştırılmıştır:

| Model                    | R2 Score | RMSE (Hata)   |
|--------------------------|----------|---------------|
| Random Forest            | 0.81     | ~16,435 ₺     |
| Gradient Boosting        | 0.80     | ~16,620 ₺     |
| Decision Tree            | 0.74     | ~19,415 ₺     |
| Extra Trees              | 0.74     | ~19,426 ₺     |
| Linear Regression        | 0.69     | ~21,219 ₺     |

## ✅ Sonuç

Toplamda 5 farklı model test edilmiştir. En iyi sonucu **Random Forest Regressor** vermiştir. Özellikle 80+ R2 puanı ile oldukça başarılı bir tahmin gücü sağlanmıştır.

---


