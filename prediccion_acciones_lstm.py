"""
Ejercicio 1 (Tema 6) — Predicción de acción financiera con LSTM
Asignatura: Inteligencia Artificial

Descarga el valor de cierre de una acción financiera entre dos fechas
y realiza una predicción de su comportamiento en los próximos días.
Evalúa el resultado en un intervalo de tiempo (conjunto de test).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Silenciar warnings de TensorFlow/oneDNN antes de importar
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# ─────────────────────────────────────────────
# 1. DESCARGAR DATOS HISTÓRICOS DE LA ACCIÓN
# ─────────────────────────────────────────────
print("=" * 60)
print("  PREDICCIÓN DE ACCIONES CON LSTM")
print("=" * 60)

TICKER  = 'AAPL'
START   = '2020-01-01'
END     = '2022-01-01'
VENTANA = 30   # días de histórico para predecir el siguiente

data_loaded = False
if _YF_AVAILABLE:
    try:
        raw = yf.download(TICKER, start=START, end=END, progress=False)
        if len(raw) > 100:
            # Versiones recientes de yfinance devuelven MultiIndex en columnas
            # (e.g. ('Close', 'AAPL')). Lo aplanamos para quedarnos con nombres simples.
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            data = raw[['Close']].dropna()
            data_loaded = True
    except Exception:
        pass

if not data_loaded:
    print("    [INFO] yfinance no disponible — usando datos sintéticos realistas de AAPL.")
    np.random.seed(42)
    fechas = pd.date_range(start=START, end=END, freq='B')
    n = len(fechas)
    precio_inicio = 75.0
    rendimientos = np.random.normal(0.0008, 0.018, n)
    precios = precio_inicio * np.exp(np.cumsum(rendimientos))
    precios[29:49]  = precios[29:49]  * np.linspace(1.0, 0.62, 20)
    precios[49:109] = precios[49:109] * np.linspace(0.62, 1.12, 60)
    data = pd.DataFrame({'Close': precios}, index=fechas)

print(f"\n[1] Datos: {TICKER}  ({START} a {END})")
print(f"    Muestras totales:   {len(data)}")
print(f"    Precio minimo:      ${float(data['Close'].min()):.2f}")
print(f"    Precio maximo:      ${float(data['Close'].max()):.2f}")
print(f"    Precio medio:       ${float(data['Close'].mean()):.2f}")

# ─────────────────────────────────────────────
# 2. NORMALIZAR CON MinMaxScaler [0, 1]
# ─────────────────────────────────────────────
scaler = MinMaxScaler(feature_range=(0, 1))
datos_norm = scaler.fit_transform(data[['Close']].values)

print(f"\n[2] Normalizacion MinMaxScaler -> rango [{datos_norm.min():.2f}, {datos_norm.max():.2f}]")

# ─────────────────────────────────────────────
# 3. CREAR VENTANAS TEMPORALES (VENTANA = 30 DÍAS)
# ─────────────────────────────────────────────
X, y = [], []
for i in range(VENTANA, len(datos_norm)):
    X.append(datos_norm[i - VENTANA:i, 0])
    y.append(datos_norm[i, 0])

X = np.array(X)
y = np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

print(f"\n[3] Ventanas temporales creadas:")
print(f"    Forma X: {X.shape}  (muestras, pasos_tiempo=30, features=1)")
print(f"    Forma y: {y.shape}")

# ─────────────────────────────────────────────
# 4. DIVIDIR: 80% TRAIN / 20% TEST (orden temporal)
# ─────────────────────────────────────────────
split = int(len(X) * 0.80)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n[4] Division temporal (80 / 20):")
print(f"    Train: {X_train.shape[0]} muestras")
print(f"    Test:  {X_test.shape[0]} muestras")

# ─────────────────────────────────────────────
# 5. DEFINIR Y ENTRENAR EL MODELO LSTM
# ─────────────────────────────────────────────
model = Sequential()
model.add(LSTM(100,
               input_shape=(VENTANA, 1),
               return_sequences=False,
               recurrent_dropout=0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(1,  activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

print(f"\n[5] Arquitectura del modelo:")
model.summary()

print(f"\n    Iniciando entrenamiento...")
historia = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.15,
    verbose=0
)
print(f"    Entrenamiento completado ({len(historia.history['loss'])} epocas).")

# ─────────────────────────────────────────────
# 6. EVALUAR SOBRE EL CONJUNTO DE TEST
# ─────────────────────────────────────────────
y_pred_norm = model.predict(X_test, verbose=0).flatten()

y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_real = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

mae  = mean_absolute_error(y_test_real, y_pred_real)
mse  = mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)

print(f"\n[6] Evaluacion sobre el conjunto de TEST ({len(y_test_real)} muestras):")
print(f"    MAE  (Error Medio Absoluto):      ${mae:.2f}")
print(f"    RMSE (Raiz del Error Cuadratico): ${rmse:.2f}")
print(f"    Precio medio test:                ${y_test_real.mean():.2f}")
print(f"    Error relativo MAE:               {mae/y_test_real.mean()*100:.2f}%")

# ─────────────────────────────────────────────
# 7. GRAFICAS
# ─────────────────────────────────────────────
fechas_all  = data.index[VENTANA:]
fechas_test = fechas_all[split:]

precios_all_real = scaler.inverse_transform(
    datos_norm[VENTANA:, 0].reshape(-1, 1)
).flatten()

# --- Grafica 1: Error MSE entrenamiento vs validacion ---
epocas    = range(1, len(historia.history['loss']) + 1)
fin_train = historia.history['loss'][-1]
fin_val   = historia.history['val_loss'][-1]

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(epocas, historia.history['loss'],     color='steelblue', lw=2,
        label=f'Entrenamiento (final: {fin_train:.6f})')
ax.plot(epocas, historia.history['val_loss'], color='tomato', lw=2, linestyle='--',
        label=f'Validacion    (final: {fin_val:.6f})')

mejor_epoca = int(np.argmin(historia.history['val_loss'])) + 1
mejor_val   = min(historia.history['val_loss'])
ax.axvline(mejor_epoca, color='gray', lw=1, linestyle=':')
ax.annotate(f'Mejor val.\nep. {mejor_epoca}',
            xy=(mejor_epoca, mejor_val),
            xytext=(mejor_epoca + 2, mejor_val + 0.002),
            fontsize=9, color='gray',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1))

ax.set_title('Error MSE por epoca - entrenamiento vs validacion',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Epoca', fontsize=12)
ax.set_ylabel('Error MSE (escala normalizada 0-1)', fontsize=12)
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('entrenamiento_lstm.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[7] Graficas guardadas:")
print("    - entrenamiento_lstm.png")

# --- Grafica 2: Serie completa + prediccion en zona test ---
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(fechas_all, precios_all_real,
        color='steelblue', lw=1.5, alpha=0.8, label='Precio real (todo el periodo)')
ax.plot(fechas_test, y_pred_real,
        color='tomato', lw=2, linestyle='--', label='Prediccion LSTM (zona test)')
ax.axvline(fechas_test[0], color='gray', lw=1.5, linestyle=':')
ax.text(fechas_test[0], precios_all_real.min() * 1.02, '  <- Test', fontsize=9, color='gray')
ax.set_title(f'Prediccion de precio de cierre - {TICKER}  ({START} a {END})\n'
             'El 20% final (zona test) evalua la capacidad de generalizacion del modelo',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Precio de cierre (USD)', fontsize=12)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, linestyle='--', alpha=0.4)
ax.text(0.97, 0.05,
        f"MAE  = ${mae:.2f}\nRMSE = ${rmse:.2f}\nError rel. = {mae/y_test_real.mean()*100:.1f}%",
        transform=ax.transAxes, fontsize=10, color='dimgray',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray'))
plt.tight_layout()
plt.savefig('prediccion_serie_completa.png', dpi=150, bbox_inches='tight')
plt.close()
print("    - prediccion_serie_completa.png")

# --- Grafica 3: Zoom en la zona de test ---
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(fechas_test, y_test_real, 'o-', color='steelblue', lw=1.5, ms=3, label='Precio REAL')
ax.plot(fechas_test, y_pred_real, 's--', color='tomato', lw=1.5, ms=3,
        label='Precio PREDICHO por LSTM')
ax.fill_between(fechas_test, y_test_real, y_pred_real,
                alpha=0.15, color='orange', label='Zona de error')
ax.set_title(f'Zoom en el conjunto de test - {TICKER}\n'
             'Comparacion dia a dia entre valor real y prediccion LSTM',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Precio de cierre (USD)', fontsize=12)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('prediccion_zoom_test.png', dpi=150, bbox_inches='tight')
plt.close()
print("    - prediccion_zoom_test.png")

# --- Grafica 4: Distribucion del error ---
errores = y_pred_real - y_test_real
fig, ax = plt.subplots(figsize=(10, 5))
n_bins, bins, patches = ax.hist(errores, bins=40, color='steelblue',
                                 edgecolor='white', linewidth=0.4)
for patch, left in zip(patches, bins[:-1]):
    if abs(left) > mae:
        patch.set_facecolor('tomato')
        patch.set_alpha(0.7)
ax.axvline(0,    color='black',      lw=2, linestyle='-',  label='Error = 0')
ax.axvline( mae, color='darkorange', lw=2, linestyle='--', label=f'MAE = +/-${mae:.2f}')
ax.axvline(-mae, color='darkorange', lw=2, linestyle='--')
ax.set_title('Distribucion del error en el conjunto de test\n'
             'Azul = errores dentro del MAE   |   Rojo = errores mayores',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Error prediccion (USD)', fontsize=12)
ax.set_ylabel('Numero de muestras', fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.4, axis='y')
dentro = np.sum(np.abs(errores) <= mae)
ax.text(0.02, 0.92,
        f'{dentro/len(errores)*100:.1f}% dentro del MAE',
        transform=ax.transAxes, fontsize=10, color='steelblue',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='lightgray'))
plt.tight_layout()
plt.savefig('distribucion_error.png', dpi=150, bbox_inches='tight')
plt.close()
print("    - distribucion_error.png")

# ─────────────────────────────────────────────
# 8. PREDICCION DE LOS PROXIMOS DIAS
# ─────────────────────────────────────────────
N_FUTUROS = 10
ultima_ventana = datos_norm[-VENTANA:, 0].tolist()
predicciones_futuras = []

for _ in range(N_FUTUROS):
    entrada = np.array(ultima_ventana[-VENTANA:]).reshape(1, VENTANA, 1)
    pred_norm = model.predict(entrada, verbose=0)[0, 0]
    predicciones_futuras.append(pred_norm)
    ultima_ventana.append(pred_norm)

predicciones_futuras_real = scaler.inverse_transform(
    np.array(predicciones_futuras).reshape(-1, 1)
).flatten()

ultimo_precio = float(data['Close'].iloc[-1])

print(f"\n[8] Prediccion de los proximos {N_FUTUROS} dias habiles tras {END}:")
print(f"    Ultimo precio conocido: ${ultimo_precio:.2f}  ({data.index[-1].date()})")
print(f"    {'Dia':>4}  {'Precio predicho':>18}")
print(f"    {'---':>4}  {'----------------':>18}")
for i, precio in enumerate(predicciones_futuras_real, 1):
    print(f"    {i:>4}  ${precio:>17.2f}")

print("\n" + "=" * 60)
print("  Proceso completado correctamente.")
print("=" * 60)
