
# 📊 Análisis de Cancelación de Clientes – Telecom X

Este proyecto tiene como objetivo analizar la cancelación de clientes (churn) en una empresa de telecomunicaciones llamada **Telecom X**, utilizando técnicas de análisis de datos y modelos de machine learning. Se identifican patrones, se evalúa el rendimiento de modelos predictivos y se proponen estrategias de retención de clientes basadas en los resultados.

## 📌 Objetivos del Proyecto

- Explorar y limpiar los datos del cliente.
- Visualizar la distribución de cancelaciones y su relación con distintas variables.
- Aplicar técnicas de balanceo (SMOTE) para mejorar la distribución de clases.
- Entrenar y evaluar dos modelos predictivos: **Regresión Logística** y **Random Forest**.
- Interpretar las variables más influyentes en la cancelación.
- Proponer estrategias de retención basadas en el análisis.

## 📚 Librerías Utilizadas

```python
import pandas as pd
import json
import requests
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
```

## 🧠 Modelos Utilizados

1. **Regresión Logística**
   - Permite interpretar la dirección e intensidad de la influencia de cada variable.
   - Métricas:
     - Accuracy: 70%
     - F1-Score: 72%
     - Precision (clase 1): 93%
     - Recall (clase 1): 64%

2. **Random Forest**
   - Modelo de tipo ensamble basado en árboles de decisión.
   - Identifica la importancia relativa de cada variable.
   - Métricas:
     - Accuracy: 74%
     - F1-Score: 74%
     - Precision (clase 1): 83%
     - Recall (clase 1): 81%

## 📈 Principales Conclusiones

- **Tipo de contrato**, **tiempo como cliente (tenure)** son factores clave en la decisión de cancelar.
- Los clientes con contratos **mensuales** tienen mayor propensión a cancelar.
- El gasto total y mensual también influye: clientes con gastos bajos tienden a cancelar más.
- Las variables con mayor impacto en los modelos son:
  - `account.Contract_Two year`
  - `account.Contract_Month-to-month`
  - `customer.tenure`
  - `customer.gender`
  - `account.Charges.Total`

## 🔁 Estrategias de Retención Propuestas

- Ofrecer beneficios a quienes usan contrato mensual para migrar a contratos anuales.
- Crear alertas para clientes con bajo tiempo de permanencia y pocos servicios activos.
- Promocionar servicios de seguridad online y soporte técnico, los cuales se asocian con menor cancelación.
- Implementar campañas preventivas para clientes con bajo gasto total y mensual.

## 🚀 Cómo usar este repositorio

Clona este repositorio en tu máquina local:

```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
```

Abre el archivo .ipynb en Jupyter o Google Colab:

Haz clic en "Abrir en Colab" (si está configurado).

O súbelo manualmente a Google Colab.

Ejecuta todas las celdas para cargar los datos y generar los gráficos interactivos.

## 🙋‍♂️ Contacto
¿Tienes preguntas o sugerencias?

📧 mitchel.poblete@outlook.com \
👤 Autor: Mitchel Poblete\
🌐 LinkedIn : [https://www.linkedin.com/in/mitchel-poblete/]
