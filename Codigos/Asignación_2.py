#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd

# Lee el archivo CSV
df = pd.read_csv('Datos Estaciones AMB - Acualago.csv')

# Elimina la segunda fila
df = df.drop(1)

# Reemplaza los valores "nodata" por NaN
df = df.applymap(lambda x: float('nan') if str(x) == 'nodata' else x)

# Guarda el DataFrame limpio en un nuevo archivo CSV
df.to_csv('datos_limpios.csv', index=False)

print("Archivo limpio guardado como 'datos_limpios.csv'")


# In[84]:


import pandas as pd

# Cargar el archivo CSV limpio
archivo_csv_limpiado = 'datos_limpios.csv'  # Reemplaza con la ubicación real del archivo limpiado
df_limpiado = pd.read_csv(archivo_csv_limpiado)

# Imprimir las columnas del DataFrame
print("\nColumnas del DataFrame:")
print(df_limpiado.columns)

# Imprimir las primeras filas del DataFrame
print("\nPrimeras filas del DataFrame:")
print(df_limpiado.head(8042))



# In[85]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV limpiado
df = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df_subset = df.iloc[106:8001].copy()

# Convertir las fechas a formato datetime
df_subset['Date&Time'] = pd.to_datetime(df_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df_subset = df_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df_subset = df_subset[~df_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df_subset['PM2.5'] = df_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas = df_subset['Date&Time']
pm25 = df_subset['PM2.5']

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos
modelo.fit(fechas.values.astype(int).reshape(-1, 1), pm25)

# Hacer predicciones con el modelo
predicciones_pm25 = modelo.predict(fechas.values.astype(int).reshape(-1, 1))

# Graficar los datos y la regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(fechas, pm25, color='blue', label='Datos reales')
plt.plot(fechas, predicciones_pm25, color='red', linewidth=2, label='Regresión lineal')
plt.title('Regresión Lineal de PM2.5')
plt.xlabel('Fecha')
plt.ylabel('PM2.5')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[86]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV limpiado
df = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df_subset = df.iloc[106:746].copy()

# Convertir las fechas a formato datetime
df_subset['Date&Time'] = pd.to_datetime(df_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df_subset = df_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df_subset = df_subset[~df_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df_subset['PM2.5'] = df_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas = df_subset['Date&Time']
pm25 = df_subset['PM2.5']

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos
modelo.fit(fechas.values.astype(int).reshape(-1, 1), pm25)

# Hacer predicciones con el modelo
predicciones_pm25 = modelo.predict(fechas.values.astype(int).reshape(-1, 1))

# Graficar los datos y la regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(fechas, pm25, color='blue', label='Datos reales')
plt.plot(fechas, predicciones_pm25, color='red', linewidth=2, label='Regresión lineal')
plt.title('Regresión Lineal de PM2.5')
plt.xlabel('Fecha')
plt.ylabel('PM2.5')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[87]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV limpiado
df = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df_subset = df.iloc[747:1466].copy()

# Convertir las fechas a formato datetime
df_subset['Date&Time'] = pd.to_datetime(df_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df_subset = df_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df_subset = df_subset[~df_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df_subset['PM2.5'] = df_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas = df_subset['Date&Time']
pm25 = df_subset['PM2.5']

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos
modelo.fit(fechas.values.astype(int).reshape(-1, 1), pm25)

# Hacer predicciones con el modelo
predicciones_pm25 = modelo.predict(fechas.values.astype(int).reshape(-1, 1))

# Graficar los datos y la regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(fechas, pm25, color='blue', label='Datos reales')
plt.plot(fechas, predicciones_pm25, color='red', linewidth=2, label='Regresión lineal')
plt.title('Regresión Lineal de PM2.5')
plt.xlabel('Fecha')
plt.ylabel('PM2.5')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[88]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV limpiado
df = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df_subset = df.iloc[1467:2210].copy()

# Convertir las fechas a formato datetime
df_subset['Date&Time'] = pd.to_datetime(df_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df_subset = df_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df_subset = df_subset[~df_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df_subset['PM2.5'] = df_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas = df_subset['Date&Time']
pm25 = df_subset['PM2.5']

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos
modelo.fit(fechas.values.astype(int).reshape(-1, 1), pm25)

# Hacer predicciones con el modelo
predicciones_pm25 = modelo.predict(fechas.values.astype(int).reshape(-1, 1))

# Graficar los datos y la regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(fechas, pm25, color='blue', label='Datos reales')
plt.plot(fechas, predicciones_pm25, color='red', linewidth=2, label='Regresión lineal')
plt.title('Regresión Lineal de PM2.5')
plt.xlabel('Fecha')
plt.ylabel('PM2.5')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[89]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV limpiado
df = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df_subset = df.iloc[2211:2954].copy()

# Convertir las fechas a formato datetime
df_subset['Date&Time'] = pd.to_datetime(df_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df_subset = df_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df_subset = df_subset[~df_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df_subset['PM2.5'] = df_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas = df_subset['Date&Time']
pm25 = df_subset['PM2.5']

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos
modelo.fit(fechas.values.astype(int).reshape(-1, 1), pm25)

# Hacer predicciones con el modelo
predicciones_pm25 = modelo.predict(fechas.values.astype(int).reshape(-1, 1))

# Graficar los datos y la regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(fechas, pm25, color='blue', label='Datos reales')
plt.plot(fechas, predicciones_pm25, color='red', linewidth=2, label='Regresión lineal')
plt.title('Regresión Lineal de PM2.5')
plt.xlabel('Fecha')
plt.ylabel('PM2.5')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[90]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV limpiado
df = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df_subset = df.iloc[2955:3626].copy()

# Convertir las fechas a formato datetime
df_subset['Date&Time'] = pd.to_datetime(df_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df_subset = df_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df_subset = df_subset[~df_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df_subset['PM2.5'] = df_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas = df_subset['Date&Time']
pm25 = df_subset['PM2.5']

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos
modelo.fit(fechas.values.astype(int).reshape(-1, 1), pm25)

# Hacer predicciones con el modelo
predicciones_pm25 = modelo.predict(fechas.values.astype(int).reshape(-1, 1))

# Graficar los datos y la regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(fechas, pm25, color='blue', label='Datos reales')
plt.plot(fechas, predicciones_pm25, color='red', linewidth=2, label='Regresión lineal')
plt.title('Regresión Lineal de PM2.5')
plt.xlabel('Fecha')
plt.ylabel('PM2.5')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[91]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV limpiado
df = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df_subset = df.iloc[3627:4370].copy()

# Convertir las fechas a formato datetime
df_subset['Date&Time'] = pd.to_datetime(df_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df_subset = df_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df_subset = df_subset[~df_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df_subset['PM2.5'] = df_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas = df_subset['Date&Time']
pm25 = df_subset['PM2.5']

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos
modelo.fit(fechas.values.astype(int).reshape(-1, 1), pm25)

# Hacer predicciones con el modelo
predicciones_pm25 = modelo.predict(fechas.values.astype(int).reshape(-1, 1))

# Graficar los datos y la regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(fechas, pm25, color='blue', label='Datos reales')
plt.plot(fechas, predicciones_pm25, color='red', linewidth=2, label='Regresión lineal')
plt.title('Regresión Lineal de PM2.5')
plt.xlabel('Fecha')
plt.ylabel('PM2.5')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[92]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV limpiado
df = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df_subset = df.iloc[4371:5090].copy()

# Convertir las fechas a formato datetime
df_subset['Date&Time'] = pd.to_datetime(df_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df_subset = df_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df_subset = df_subset[~df_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df_subset['PM2.5'] = df_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas = df_subset['Date&Time']
pm25 = df_subset['PM2.5']

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos
modelo.fit(fechas.values.astype(int).reshape(-1, 1), pm25)

# Hacer predicciones con el modelo
predicciones_pm25 = modelo.predict(fechas.values.astype(int).reshape(-1, 1))

# Graficar los datos y la regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(fechas, pm25, color='blue', label='Datos reales')
plt.plot(fechas, predicciones_pm25, color='red', linewidth=2, label='Regresión lineal')
plt.title('Regresión Lineal de PM2.5')
plt.xlabel('Fecha')
plt.ylabel('PM2.5')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[112]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV limpiado
df = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df_subset = df.iloc[5091:5834].copy()

# Convertir las fechas a formato datetime
df_subset['Date&Time'] = pd.to_datetime(df_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df_subset = df_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df_subset = df_subset[~df_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df_subset['PM2.5'] = df_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas = df_subset['Date&Time']
pm25 = df_subset['PM2.5']

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos
modelo.fit(fechas.values.astype(int).reshape(-1, 1), pm25)

# Hacer predicciones con el modelo
predicciones_pm25 = modelo.predict(fechas.values.astype(int).reshape(-1, 1))

# Graficar los datos y la regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(fechas, pm25, color='blue', label='Datos reales')
plt.plot(fechas, predicciones_pm25, color='red', linewidth=2, label='Regresión lineal')
plt.title('Regresión Lineal de PM2.5')
plt.xlabel('Fecha')
plt.ylabel('PM2.5')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[113]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV limpiado
df = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df_subset = df.iloc[5835:6555].copy()

# Convertir las fechas a formato datetime
df_subset['Date&Time'] = pd.to_datetime(df_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df_subset = df_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df_subset = df_subset[~df_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df_subset['PM2.5'] = df_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas = df_subset['Date&Time']
pm25 = df_subset['PM2.5']

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos
modelo.fit(fechas.values.astype(int).reshape(-1, 1), pm25)

# Hacer predicciones con el modelo
predicciones_pm25 = modelo.predict(fechas.values.astype(int).reshape(-1, 1))

# Graficar los datos y la regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(fechas, pm25, color='blue', label='Datos reales')
plt.plot(fechas, predicciones_pm25, color='red', linewidth=2, label='Regresión lineal')
plt.title('Regresión Lineal de PM2.5')
plt.xlabel('Fecha')
plt.ylabel('PM2.5')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[114]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV limpiado
df = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df_subset = df.iloc[6556:7298].copy()

# Convertir las fechas a formato datetime
df_subset['Date&Time'] = pd.to_datetime(df_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df_subset = df_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df_subset = df_subset[~df_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df_subset['PM2.5'] = df_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas = df_subset['Date&Time']
pm25 = df_subset['PM2.5']

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos
modelo.fit(fechas.values.astype(int).reshape(-1, 1), pm25)

# Hacer predicciones con el modelo
predicciones_pm25 = modelo.predict(fechas.values.astype(int).reshape(-1, 1))

# Graficar los datos y la regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(fechas, pm25, color='blue', label='Datos reales')
plt.plot(fechas, predicciones_pm25, color='red', linewidth=2, label='Regresión lineal')
plt.title('Regresión Lineal de PM2.5')
plt.xlabel('Fecha')
plt.ylabel('PM2.5')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[115]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV limpiado
df = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df_subset = df.iloc[7299:8042].copy()

# Convertir las fechas a formato datetime
df_subset['Date&Time'] = pd.to_datetime(df_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df_subset = df_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df_subset = df_subset[~df_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df_subset['PM2.5'] = df_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas = df_subset['Date&Time']
pm25 = df_subset['PM2.5']

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos
modelo.fit(fechas.values.astype(int).reshape(-1, 1), pm25)

# Hacer predicciones con el modelo
predicciones_pm25 = modelo.predict(fechas.values.astype(int).reshape(-1, 1))

# Graficar los datos y la regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(fechas, pm25, color='blue', label='Datos reales')
plt.plot(fechas, predicciones_pm25, color='red', linewidth=2, label='Regresión lineal')
plt.title('Regresión Lineal de PM2.5')
plt.xlabel('Fecha')
plt.ylabel('PM2.5')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[116]:


import os
import pandas as pd

# Ruta de la carpeta donde se encuentran los archivos
carpeta = r'C:\Users\jeico\Downloads\data'

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(carpeta)

# Iterar sobre cada archivo en la carpeta
for archivo in archivos:
    # Combinar la ruta de la carpeta con el nombre del archivo
    ruta_completa = os.path.join(carpeta, archivo)
    
    # Verificar si el elemento en la ruta es un archivo
    if os.path.isfile(ruta_completa) and ruta_completa.endswith('.csv'):
        # Leer el archivo usando pandas
        try:
            df = pd.read_csv(ruta_completa)
            # Mostrar el nombre del archivo
            print("Archivo:", archivo)
            # Mostrar el contenido del archivo como tabla
            print(df)
            print("-" * 30)
        except pd.errors.EmptyDataError:
            # El archivo está vacío
            print("El archivo", archivo, "está vacío.")
        except pd.errors.ParserError:
            # No se pudo parsear el archivo
            print("No se pudo leer el archivo", archivo)


# In[117]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def calibrate_sensor(file_path):
    # Cargar datos desde un archivo CSV
    df = pd.read_csv(file_path)

    # Convertir la columna de fechas a formato datetime si es necesario
    if 'fecha_hora_med' in df.columns:
        df['fecha_hora_med'] = pd.to_datetime(df['fecha_hora_med'])

    # Eliminar filas con valores NaN en la columna de 'valor'
    df = df.dropna(subset=['valor'])

    # Obtener las fechas y los valores
    fechas = df['fecha_hora_med']
    valores = df['valor']

    # Crear el modelo de regresión lineal
    model = LinearRegression()

    # Ajustar el modelo a los datos
    model.fit(fechas.values.astype(int).reshape(-1, 1), valores)

    # Hacer predicciones con el modelo
    y_pred = model.predict(fechas.values.astype(int).reshape(-1, 1))

    # Visualizar los resultados
    plt.scatter(fechas, valores, color='black', label='Datos reales')
    plt.plot(fechas, y_pred, color='blue', linewidth=3, label='Predicción')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# Ruta del archivo CSV
file_path = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2018-11-01T00_00_00_2018-11-30T23_59_59.csv'

# Llamar a la función para calibrar el sensor y realizar la regresión lineal
calibrate_sensor(file_path)


# In[118]:


import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def calibrate_sensor_for_directory(directory_path):
    # Obtener la lista de archivos en el directorio
    files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    # Iterar sobre cada archivo
    for file in files:
        file_path = os.path.join(directory_path, file)

        # Cargar datos desde un archivo CSV
        df = pd.read_csv(file_path)

        # Convertir la columna de fechas a formato datetime si es necesario
        if 'fecha_hora_med' in df.columns:
            df['fecha_hora_med'] = pd.to_datetime(df['fecha_hora_med'])

        # Eliminar filas con valores NaN en la columna de 'valor'
        df = df.dropna(subset=['valor'])

        # Obtener las fechas y los valores
        fechas = df['fecha_hora_med']
        valores = df['valor']

        # Crear el modelo de regresión lineal
        model = LinearRegression()

        # Ajustar el modelo a los datos
        model.fit(fechas.values.astype(int).reshape(-1, 1), valores)

        # Hacer predicciones con el modelo
        y_pred = model.predict(fechas.values.astype(int).reshape(-1, 1))

        # Visualizar los resultados
        plt.scatter(fechas, valores, color='black', label='Datos reales')
        plt.plot(fechas, y_pred, color='blue', linewidth=3, label='Predicción')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.title(f'Calibración del sensor - {file}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

        # Coeficientes de la ecuación lineal (y = mx + b)
        slope = model.coef_[0]
        intercept = model.intercept_
        print(f'Ecuación lineal de calibración para {file}: y = {slope:.4f}x + {intercept:.4f}')

# Ruta del directorio que contiene los archivos CSV de mediciones
directory_path = r'\Users\jeico\Downloads\data'

# Llamar a la función para calibrar el sensor y realizar la regresión lineal para todos los archivos en el directorio
calibrate_sensor_for_directory(directory_path)


# In[119]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar el primer archivo CSV limpiado
df1 = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df1_subset = df1.iloc[818:1471].copy()

# Convertir las fechas a formato datetime
df1_subset['Date&Time'] = pd.to_datetime(df1_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df1_subset = df1_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df1_subset = df1_subset[~df1_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df1_subset['PM2.5'] = df1_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas1 = df1_subset['Date&Time']
pm25_1 = df1_subset['PM2.5']

# Crear el modelo de regresión lineal para el primer conjunto de datos
modelo1 = LinearRegression()

# Ajustar el modelo a los datos
modelo1.fit(fechas1.values.astype(int).reshape(-1, 1), pm25_1)

# Hacer predicciones con el modelo
predicciones_pm25_1 = modelo1.predict(fechas1.values.astype(int).reshape(-1, 1))

# Cargar el segundo archivo CSV
file_path = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2018-11-01T00_00_00_2018-11-30T23_59_59.csv'

# Cargar el segundo conjunto de datos
df2 = pd.read_csv(file_path)

# Convertir las fechas a formato datetime si es necesario
if 'fecha_hora_med' in df2.columns:
    df2['fecha_hora_med'] = pd.to_datetime(df2['fecha_hora_med'])

# Eliminar filas con valores NaN en la columna de 'valor'
df2 = df2.dropna(subset=['valor'])

# Obtener las fechas y los valores
fechas2 = df2['fecha_hora_med']
valores2 = df2['valor']

# Crear el modelo de regresión lineal para el segundo conjunto de datos
modelo2 = LinearRegression()

# Ajustar el modelo a los datos
modelo2.fit(fechas2.values.astype(int).reshape(-1, 1), valores2)

# Hacer predicciones con el modelo
y_pred_2 = modelo2.predict(fechas2.values.astype(int).reshape(-1, 1))

# Visualizar los resultados en una sola figura
plt.figure(figsize=(10, 6))

# Graficar los datos y la regresión lineal del primer conjunto de datos
plt.scatter(fechas1, pm25_1, color='blue', label='Datos reales (Dataset 1)')
plt.plot(fechas1, predicciones_pm25_1, color='red', linewidth=2, label='Regresión lineal (Dataset 1)')

# Graficar los datos y la regresión lineal del segundo conjunto de datos
plt.scatter(fechas2, valores2, color='black', label='Datos reales (Dataset 2)')
plt.plot(fechas2, y_pred_2, color='green', linewidth=2, label='Regresión lineal (Dataset 2)')

plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación de Regresiones Lineales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Calcular la distancia entre las dos regresiones lineales
# Se puede usar la distancia Euclidiana entre los coeficientes de las rectas
distancia = np.sqrt((modelo1.coef_[0] - modelo2.coef_[0])**2 + (modelo1.intercept_ - modelo2.intercept_)**2)
print(f'Distancia entre las regresiones lineales: {distancia}')


# In[145]:


# Tamaños de ventana para el promedio móvil
tamanos_ventana = [3, 5, 7, 10]

# Configurar el número de filas y columnas para los subplots
num_filas = len(tamanos_ventana)
num_columnas = 2  # Dos subplots por fila
def promedio_movil(data, ventana):
    return data.rolling(window=ventana).mean()

# Configurar el tamaño de la figura
plt.figure(figsize=(15, 5 * num_filas))

for i, ventana in enumerate(tamanos_ventana):
    # Calcular el promedio móvil con ventana de tamaño actual para el primer conjunto de datos
    promedio_movil_df1 = promedio_movil(pm25_1, ventana=ventana)

    # Calcular el promedio móvil con ventana de tamaño actual para el segundo conjunto de datos
    promedio_movil_df2 = promedio_movil(valores2, ventana=ventana)

    # Calcular el promedio del promedio móvil con ventana de tamaño actual para el primer conjunto de datos
    promedio_promedio_movil_df1 = promedio_movil_df1.mean()

    # Calcular el promedio del promedio móvil con ventana de tamaño actual para el segundo conjunto de datos
    promedio_promedio_movil_df2 = promedio_movil_df2.mean()

    # Calcular la distancia entre los promedios móviles con ventana de tamaño actual
    distancia_promedios_moviles = abs(promedio_promedio_movil_df1 - promedio_promedio_movil_df2)
    
    # Configurar el subplot actual
    plt.subplot(num_filas, num_columnas, i + 1)

    # Graficar el promedio móvil para el primer conjunto de datos
    plt.plot(fechas1, promedio_movil_df1, label=f'Datos Estaciones AMB(ventana={ventana})', color='blue')

    # Graficar el promedio móvil para el segundo conjunto de datos
    plt.plot(fechas2, promedio_movil_df2, label=f'Conjunto de datos 2 (ventana={ventana})', color='red')

    plt.xlabel('Fecha')
    plt.ylabel(f'Promedio móvil (Ventana {ventana})')
    plt.title(f'Promedio Móvil con Ventana de Tamaño {ventana}')
    plt.xticks(rotation=45)
    plt.legend()

    # Agregar texto con los valores promedio y la distancia entre ellos
    plt.text(0.5, -0.3, f'Promedio móvil (Datos Estaciones AMB): {promedio_promedio_movil_df1:.2f}\nPromedio móvil (Mediciones2019-08-31): {promedio_promedio_movil_df2:.2f}\nDistancia entre los promedios móviles: {distancia_promedios_moviles:.2f}',
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

# Ajustar el diseño y mostrar la figura
plt.tight_layout()
plt.show()
# Encontrar el índice del mínimo en la lista de distancias
indice_mejor_ventana = distancias.index(min(distancias))

# Obtener el mejor tamaño de ventana
mejor_ventana = tamanos_ventana[indice_mejor_ventana]

# Imprimir el mejor tamaño de ventana
print(f"El mejor tamaño de ventana es: {mejor_ventana}")


# In[121]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar el primer archivo CSV limpiado
df1 = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df1_subset = df1.iloc[1467:1802].copy()

# Convertir las fechas a formato datetime
df1_subset['Date&Time'] = pd.to_datetime(df1_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df1_subset = df1_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df1_subset = df1_subset[~df1_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df1_subset['PM2.5'] = df1_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas1 = df1_subset['Date&Time']
pm25_1 = df1_subset['PM2.5']

# Crear el modelo de regresión lineal para el primer conjunto de datos
modelo1 = LinearRegression()

# Ajustar el modelo a los datos
modelo1.fit(fechas1.values.astype(int).reshape(-1, 1), pm25_1)

# Hacer predicciones con el modelo
predicciones_pm25_1 = modelo1.predict(fechas1.values.astype(int).reshape(-1, 1))

# Cargar el segundo archivo CSV
file_path = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2018-12-01T00_00_00_2018-12-31T23_59_59.csv'

# Cargar el segundo conjunto de datos
df2 = pd.read_csv(file_path)

# Convertir las fechas a formato datetime si es necesario
if 'fecha_hora_med' in df2.columns:
    df2['fecha_hora_med'] = pd.to_datetime(df2['fecha_hora_med'])

# Eliminar filas con valores NaN en la columna de 'valor'
df2 = df2.dropna(subset=['valor'])

# Obtener las fechas y los valores
fechas2 = df2['fecha_hora_med']
valores2 = df2['valor']

# Crear el modelo de regresión lineal para el segundo conjunto de datos
modelo2 = LinearRegression()

# Ajustar el modelo a los datos
modelo2.fit(fechas2.values.astype(int).reshape(-1, 1), valores2)

# Hacer predicciones con el modelo
y_pred_2 = modelo2.predict(fechas2.values.astype(int).reshape(-1, 1))

# Visualizar los resultados en una sola figura
plt.figure(figsize=(10, 6))

# Graficar los datos y la regresión lineal del primer conjunto de datos
plt.scatter(fechas1, pm25_1, color='blue', label='Datos reales (Dataset 1)')
plt.plot(fechas1, predicciones_pm25_1, color='red', linewidth=2, label='Regresión lineal (Dataset 1)')

# Graficar los datos y la regresión lineal del segundo conjunto de datos
plt.scatter(fechas2, valores2, color='black', label='Datos reales (Dataset 2)')
plt.plot(fechas2, y_pred_2, color='green', linewidth=2, label='Regresión lineal (Dataset 2)')

plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación de Regresiones Lineales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Calcular la distancia entre las dos regresiones lineales
# Se puede usar la distancia Euclidiana entre los coeficientes de las rectas
distancia = np.sqrt((modelo1.coef_[0] - modelo2.coef_[0])**2 + (modelo1.intercept_ - modelo2.intercept_)**2)
print(f'Distancia entre las regresiones lineales: {distancia}')


# In[144]:


# Tamaños de ventana para el promedio móvil
tamanos_ventana = [3, 5, 7, 10]

# Configurar el número de filas y columnas para los subplots
num_filas = len(tamanos_ventana)
num_columnas = 2  # Dos subplots por fila
def promedio_movil(data, ventana):
    return data.rolling(window=ventana).mean()

# Configurar el tamaño de la figura
plt.figure(figsize=(15, 5 * num_filas))

for i, ventana in enumerate(tamanos_ventana):
    # Calcular el promedio móvil con ventana de tamaño actual para el primer conjunto de datos
    promedio_movil_df1 = promedio_movil(pm25_1, ventana=ventana)

    # Calcular el promedio móvil con ventana de tamaño actual para el segundo conjunto de datos
    promedio_movil_df2 = promedio_movil(valores2, ventana=ventana)

    # Calcular el promedio del promedio móvil con ventana de tamaño actual para el primer conjunto de datos
    promedio_promedio_movil_df1 = promedio_movil_df1.mean()

    # Calcular el promedio del promedio móvil con ventana de tamaño actual para el segundo conjunto de datos
    promedio_promedio_movil_df2 = promedio_movil_df2.mean()

    # Calcular la distancia entre los promedios móviles con ventana de tamaño actual
    distancia_promedios_moviles = abs(promedio_promedio_movil_df1 - promedio_promedio_movil_df2)
    
    # Configurar el subplot actual
    plt.subplot(num_filas, num_columnas, i + 1)

    # Graficar el promedio móvil para el primer conjunto de datos
    plt.plot(fechas1, promedio_movil_df1, label=f'Datos Estaciones AMB(ventana={ventana})', color='blue')

    # Graficar el promedio móvil para el segundo conjunto de datos
    plt.plot(fechas2, promedio_movil_df2, label=f'Conjunto de datos 2 (ventana={ventana})', color='red')

    plt.xlabel('Fecha')
    plt.ylabel(f'Promedio móvil (Ventana {ventana})')
    plt.title(f'Promedio Móvil con Ventana de Tamaño {ventana}')
    plt.xticks(rotation=45)
    plt.legend()

    # Agregar texto con los valores promedio y la distancia entre ellos
    plt.text(0.5, -0.3, f'Promedio móvil (Datos Estaciones AMB): {promedio_promedio_movil_df1:.2f}\nPromedio móvil (Mediciones2019-08-31): {promedio_promedio_movil_df2:.2f}\nDistancia entre los promedios móviles: {distancia_promedios_moviles:.2f}',
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

# Ajustar el diseño y mostrar la figura
plt.tight_layout()
plt.show()
# Encontrar el índice del mínimo en la lista de distancias
indice_mejor_ventana = distancias.index(min(distancias))

# Obtener el mejor tamaño de ventana
mejor_ventana = tamanos_ventana[indice_mejor_ventana]

# Imprimir el mejor tamaño de ventana
print(f"El mejor tamaño de ventana es: {mejor_ventana}")


# In[123]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar el primer archivo CSV limpiado
df1 = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df1_subset = df1.iloc[4628:5095].copy()

# Convertir las fechas a formato datetime
df1_subset['Date&Time'] = pd.to_datetime(df1_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df1_subset = df1_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df1_subset = df1_subset[~df1_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df1_subset['PM2.5'] = df1_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas1 = df1_subset['Date&Time']
pm25_1 = df1_subset['PM2.5']

# Crear el modelo de regresión lineal para el primer conjunto de datos
modelo1 = LinearRegression()

# Ajustar el modelo a los datos
modelo1.fit(fechas1.values.astype(int).reshape(-1, 1), pm25_1)

# Hacer predicciones con el modelo
predicciones_pm25_1 = modelo1.predict(fechas1.values.astype(int).reshape(-1, 1))

# Cargar el segundo archivo CSV
file_path = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2019-04-01T00_00_00_2019-04-30T23_59_59.csv'

# Cargar el segundo conjunto de datos
df2 = pd.read_csv(file_path)

# Convertir las fechas a formato datetime si es necesario
if 'fecha_hora_med' in df2.columns:
    df2['fecha_hora_med'] = pd.to_datetime(df2['fecha_hora_med'])

# Eliminar filas con valores NaN en la columna de 'valor'
df2 = df2.dropna(subset=['valor'])

# Obtener las fechas y los valores
fechas2 = df2['fecha_hora_med']
valores2 = df2['valor']

# Crear el modelo de regresión lineal para el segundo conjunto de datos
modelo2 = LinearRegression()

# Ajustar el modelo a los datos
modelo2.fit(fechas2.values.astype(int).reshape(-1, 1), valores2)

# Hacer predicciones con el modelo
y_pred_2 = modelo2.predict(fechas2.values.astype(int).reshape(-1, 1))

# Visualizar los resultados en una sola figura
plt.figure(figsize=(10, 6))

# Graficar los datos y la regresión lineal del primer conjunto de datos
plt.scatter(fechas1, pm25_1, color='blue', label='Datos reales (Dataset 1)')
plt.plot(fechas1, predicciones_pm25_1, color='red', linewidth=2, label='Regresión lineal (Dataset 1)')

# Graficar los datos y la regresión lineal del segundo conjunto de datos
plt.scatter(fechas2, valores2, color='black', label='Datos reales (Dataset 2)')
plt.plot(fechas2, y_pred_2, color='green', linewidth=2, label='Regresión lineal (Dataset 2)')

plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación de Regresiones Lineales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Calcular la distancia entre las dos regresiones lineales
# Se puede usar la distancia Euclidiana entre los coeficientes de las rectas
distancia = np.sqrt((modelo1.coef_[0] - modelo2.coef_[0])**2 + (modelo1.intercept_ - modelo2.intercept_)**2)
print(f'Distancia entre las regresiones lineales: {distancia}')


# In[143]:


# Tamaños de ventana para el promedio móvil
tamanos_ventana = [3, 5, 7, 10]

# Configurar el número de filas y columnas para los subplots
num_filas = len(tamanos_ventana)
num_columnas = 2  # Dos subplots por fila
def promedio_movil(data, ventana):
    return data.rolling(window=ventana).mean()

# Configurar el tamaño de la figura
plt.figure(figsize=(15, 5 * num_filas))

for i, ventana in enumerate(tamanos_ventana):
    # Calcular el promedio móvil con ventana de tamaño actual para el primer conjunto de datos
    promedio_movil_df1 = promedio_movil(pm25_1, ventana=ventana)

    # Calcular el promedio móvil con ventana de tamaño actual para el segundo conjunto de datos
    promedio_movil_df2 = promedio_movil(valores2, ventana=ventana)

    # Calcular el promedio del promedio móvil con ventana de tamaño actual para el primer conjunto de datos
    promedio_promedio_movil_df1 = promedio_movil_df1.mean()

    # Calcular el promedio del promedio móvil con ventana de tamaño actual para el segundo conjunto de datos
    promedio_promedio_movil_df2 = promedio_movil_df2.mean()

    # Calcular la distancia entre los promedios móviles con ventana de tamaño actual
    distancia_promedios_moviles = abs(promedio_promedio_movil_df1 - promedio_promedio_movil_df2)
    
    # Configurar el subplot actual
    plt.subplot(num_filas, num_columnas, i + 1)

    # Graficar el promedio móvil para el primer conjunto de datos
    plt.plot(fechas1, promedio_movil_df1, label=f'Datos Estaciones AMB(ventana={ventana})', color='blue')

    # Graficar el promedio móvil para el segundo conjunto de datos
    plt.plot(fechas2, promedio_movil_df2, label=f'Conjunto de datos 2 (ventana={ventana})', color='red')

    plt.xlabel('Fecha')
    plt.ylabel(f'Promedio móvil (Ventana {ventana})')
    plt.title(f'Promedio Móvil con Ventana de Tamaño {ventana}')
    plt.xticks(rotation=45)
    plt.legend()

    # Agregar texto con los valores promedio y la distancia entre ellos
    plt.text(0.5, -0.3, f'Promedio móvil (Datos Estaciones AMB): {promedio_promedio_movil_df1:.2f}\nPromedio móvil (Mediciones2019-08-31): {promedio_promedio_movil_df2:.2f}\nDistancia entre los promedios móviles: {distancia_promedios_moviles:.2f}',
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

# Ajustar el diseño y mostrar la figura
plt.tight_layout()
plt.show()
# Encontrar el índice del mínimo en la lista de distancias
indice_mejor_ventana = distancias.index(min(distancias))

# Obtener el mejor tamaño de ventana
mejor_ventana = tamanos_ventana[indice_mejor_ventana]

# Imprimir el mejor tamaño de ventana
print(f"El mejor tamaño de ventana es: {mejor_ventana}")


# In[125]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar el primer archivo CSV limpiado
df1 = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df1_subset = df1.iloc[5096:5839].copy()

# Convertir las fechas a formato datetime
df1_subset['Date&Time'] = pd.to_datetime(df1_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df1_subset = df1_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df1_subset = df1_subset[~df1_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df1_subset['PM2.5'] = df1_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas1 = df1_subset['Date&Time']
pm25_1 = df1_subset['PM2.5']

# Crear el modelo de regresión lineal para el primer conjunto de datos
modelo1 = LinearRegression()

# Ajustar el modelo a los datos
modelo1.fit(fechas1.values.astype(int).reshape(-1, 1), pm25_1)

# Hacer predicciones con el modelo
predicciones_pm25_1 = modelo1.predict(fechas1.values.astype(int).reshape(-1, 1))

# Cargar el segundo archivo CSV
file_path = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2019-05-01T00_00_00_2019-05-31T23_59_59.csv'

# Cargar el segundo conjunto de datos
df2 = pd.read_csv(file_path)

# Convertir las fechas a formato datetime si es necesario
if 'fecha_hora_med' in df2.columns:
    df2['fecha_hora_med'] = pd.to_datetime(df2['fecha_hora_med'])

# Eliminar filas con valores NaN en la columna de 'valor'
df2 = df2.dropna(subset=['valor'])

# Obtener las fechas y los valores
fechas2 = df2['fecha_hora_med']
valores2 = df2['valor']

# Crear el modelo de regresión lineal para el segundo conjunto de datos
modelo2 = LinearRegression()

# Ajustar el modelo a los datos
modelo2.fit(fechas2.values.astype(int).reshape(-1, 1), valores2)

# Hacer predicciones con el modelo
y_pred_2 = modelo2.predict(fechas2.values.astype(int).reshape(-1, 1))

# Visualizar los resultados en una sola figura
plt.figure(figsize=(10, 6))

# Graficar los datos y la regresión lineal del primer conjunto de datos
plt.scatter(fechas1, pm25_1, color='blue', label='Datos reales (Dataset 1)')
plt.plot(fechas1, predicciones_pm25_1, color='red', linewidth=2, label='Regresión lineal (Dataset 1)')

# Graficar los datos y la regresión lineal del segundo conjunto de datos
plt.scatter(fechas2, valores2, color='black', label='Datos reales (Dataset 2)')
plt.plot(fechas2, y_pred_2, color='green', linewidth=2, label='Regresión lineal (Dataset 2)')

plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación de Regresiones Lineales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Calcular la distancia entre las dos regresiones lineales
# Se puede usar la distancia Euclidiana entre los coeficientes de las rectas
distancia = np.sqrt((modelo1.coef_[0] - modelo2.coef_[0])**2 + (modelo1.intercept_ - modelo2.intercept_)**2)
print(f'Distancia entre las regresiones lineales: {distancia}')


# In[142]:


# Tamaños de ventana para el promedio móvil
tamanos_ventana = [3, 5, 7, 10]

# Configurar el número de filas y columnas para los subplots
num_filas = len(tamanos_ventana)
num_columnas = 2  # Dos subplots por fila
def promedio_movil(data, ventana):
    return data.rolling(window=ventana).mean()

# Configurar el tamaño de la figura
plt.figure(figsize=(15, 5 * num_filas))

for i, ventana in enumerate(tamanos_ventana):
    # Calcular el promedio móvil con ventana de tamaño actual para el primer conjunto de datos
    promedio_movil_df1 = promedio_movil(pm25_1, ventana=ventana)

    # Calcular el promedio móvil con ventana de tamaño actual para el segundo conjunto de datos
    promedio_movil_df2 = promedio_movil(valores2, ventana=ventana)

    # Calcular el promedio del promedio móvil con ventana de tamaño actual para el primer conjunto de datos
    promedio_promedio_movil_df1 = promedio_movil_df1.mean()

    # Calcular el promedio del promedio móvil con ventana de tamaño actual para el segundo conjunto de datos
    promedio_promedio_movil_df2 = promedio_movil_df2.mean()

    # Calcular la distancia entre los promedios móviles con ventana de tamaño actual
    distancia_promedios_moviles = abs(promedio_promedio_movil_df1 - promedio_promedio_movil_df2)
    
    # Configurar el subplot actual
    plt.subplot(num_filas, num_columnas, i + 1)

    # Graficar el promedio móvil para el primer conjunto de datos
    plt.plot(fechas1, promedio_movil_df1, label=f'Datos Estaciones AMB(ventana={ventana})', color='blue')

    # Graficar el promedio móvil para el segundo conjunto de datos
    plt.plot(fechas2, promedio_movil_df2, label=f'Conjunto de datos 2 (ventana={ventana})', color='red')

    plt.xlabel('Fecha')
    plt.ylabel(f'Promedio móvil (Ventana {ventana})')
    plt.title(f'Promedio Móvil con Ventana de Tamaño {ventana}')
    plt.xticks(rotation=45)
    plt.legend()

    # Agregar texto con los valores promedio y la distancia entre ellos
    plt.text(0.5, -0.3, f'Promedio móvil (Datos Estaciones AMB): {promedio_promedio_movil_df1:.2f}\nPromedio móvil (Mediciones2019-08-31): {promedio_promedio_movil_df2:.2f}\nDistancia entre los promedios móviles: {distancia_promedios_moviles:.2f}',
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

# Ajustar el diseño y mostrar la figura
plt.tight_layout()
plt.show()
# Encontrar el índice del mínimo en la lista de distancias
indice_mejor_ventana = distancias.index(min(distancias))

# Obtener el mejor tamaño de ventana
mejor_ventana = tamanos_ventana[indice_mejor_ventana]

# Imprimir el mejor tamaño de ventana
print(f"El mejor tamaño de ventana es: {mejor_ventana}")


# In[127]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar el primer archivo CSV limpiado
df1 = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df1_subset = df1.iloc[5841:6559].copy()

# Convertir las fechas a formato datetime
df1_subset['Date&Time'] = pd.to_datetime(df1_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df1_subset = df1_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df1_subset = df1_subset[~df1_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df1_subset['PM2.5'] = df1_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas1 = df1_subset['Date&Time']
pm25_1 = df1_subset['PM2.5']

# Crear el modelo de regresión lineal para el primer conjunto de datos
modelo1 = LinearRegression()

# Ajustar el modelo a los datos
modelo1.fit(fechas1.values.astype(int).reshape(-1, 1), pm25_1)

# Hacer predicciones con el modelo
predicciones_pm25_1 = modelo1.predict(fechas1.values.astype(int).reshape(-1, 1))

# Cargar el segundo archivo CSV
file_path = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2019-06-01T00_00_00_2019-06-30T23_59_59.csv'

# Cargar el segundo conjunto de datos
df2 = pd.read_csv(file_path)

# Convertir las fechas a formato datetime si es necesario
if 'fecha_hora_med' in df2.columns:
    df2['fecha_hora_med'] = pd.to_datetime(df2['fecha_hora_med'])

# Eliminar filas con valores NaN en la columna de 'valor'
df2 = df2.dropna(subset=['valor'])

# Obtener las fechas y los valores
fechas2 = df2['fecha_hora_med']
valores2 = df2['valor']

# Crear el modelo de regresión lineal para el segundo conjunto de datos
modelo2 = LinearRegression()

# Ajustar el modelo a los datos
modelo2.fit(fechas2.values.astype(int).reshape(-1, 1), valores2)

# Hacer predicciones con el modelo
y_pred_2 = modelo2.predict(fechas2.values.astype(int).reshape(-1, 1))

# Visualizar los resultados en una sola figura
plt.figure(figsize=(10, 6))

# Graficar los datos y la regresión lineal del primer conjunto de datos
plt.scatter(fechas1, pm25_1, color='blue', label='Datos reales (Dataset 1)')
plt.plot(fechas1, predicciones_pm25_1, color='red', linewidth=2, label='Regresión lineal (Dataset 1)')

# Graficar los datos y la regresión lineal del segundo conjunto de datos
plt.scatter(fechas2, valores2, color='black', label='Datos reales (Dataset 2)')
plt.plot(fechas2, y_pred_2, color='green', linewidth=2, label='Regresión lineal (Dataset 2)')

plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación de Regresiones Lineales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Calcular la distancia entre las dos regresiones lineales
# Se puede usar la distancia Euclidiana entre los coeficientes de las rectas
distancia = np.sqrt((modelo1.coef_[0] - modelo2.coef_[0])**2 + (modelo1.intercept_ - modelo2.intercept_)**2)
print(f'Distancia entre las regresiones lineales: {distancia}')


# In[141]:


# Tamaños de ventana para el promedio móvil
tamanos_ventana = [3, 5, 7, 10]

# Configurar el número de filas y columnas para los subplots
num_filas = len(tamanos_ventana)
num_columnas = 2  # Dos subplots por fila
def promedio_movil(data, ventana):
    return data.rolling(window=ventana).mean()

# Configurar el tamaño de la figura
plt.figure(figsize=(15, 5 * num_filas))

for i, ventana in enumerate(tamanos_ventana):
    # Calcular el promedio móvil con ventana de tamaño actual para el primer conjunto de datos
    promedio_movil_df1 = promedio_movil(pm25_1, ventana=ventana)

    # Calcular el promedio móvil con ventana de tamaño actual para el segundo conjunto de datos
    promedio_movil_df2 = promedio_movil(valores2, ventana=ventana)

    # Calcular el promedio del promedio móvil con ventana de tamaño actual para el primer conjunto de datos
    promedio_promedio_movil_df1 = promedio_movil_df1.mean()

    # Calcular el promedio del promedio móvil con ventana de tamaño actual para el segundo conjunto de datos
    promedio_promedio_movil_df2 = promedio_movil_df2.mean()

    # Calcular la distancia entre los promedios móviles con ventana de tamaño actual
    distancia_promedios_moviles = abs(promedio_promedio_movil_df1 - promedio_promedio_movil_df2)
    
    # Configurar el subplot actual
    plt.subplot(num_filas, num_columnas, i + 1)

    # Graficar el promedio móvil para el primer conjunto de datos
    plt.plot(fechas1, promedio_movil_df1, label=f'Datos Estaciones AMB(ventana={ventana})', color='blue')

    # Graficar el promedio móvil para el segundo conjunto de datos
    plt.plot(fechas2, promedio_movil_df2, label=f'Conjunto de datos 2 (ventana={ventana})', color='red')

    plt.xlabel('Fecha')
    plt.ylabel(f'Promedio móvil (Ventana {ventana})')
    plt.title(f'Promedio Móvil con Ventana de Tamaño {ventana}')
    plt.xticks(rotation=45)
    plt.legend()

    # Agregar texto con los valores promedio y la distancia entre ellos
    plt.text(0.5, -0.3, f'Promedio móvil (Datos Estaciones AMB): {promedio_promedio_movil_df1:.2f}\nPromedio móvil (Mediciones2019-08-31): {promedio_promedio_movil_df2:.2f}\nDistancia entre los promedios móviles: {distancia_promedios_moviles:.2f}',
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

# Ajustar el diseño y mostrar la figura
plt.tight_layout()
plt.show()
# Encontrar el índice del mínimo en la lista de distancias
indice_mejor_ventana = distancias.index(min(distancias))

# Obtener el mejor tamaño de ventana
mejor_ventana = tamanos_ventana[indice_mejor_ventana]

# Imprimir el mejor tamaño de ventana
print(f"El mejor tamaño de ventana es: {mejor_ventana}")


# In[129]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar el primer archivo CSV limpiado
df1 = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df1_subset = df1.iloc[6560:7303].copy()

# Convertir las fechas a formato datetime
df1_subset['Date&Time'] = pd.to_datetime(df1_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df1_subset = df1_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df1_subset = df1_subset[~df1_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df1_subset['PM2.5'] = df1_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas1 = df1_subset['Date&Time']
pm25_1 = df1_subset['PM2.5']

# Crear el modelo de regresión lineal para el primer conjunto de datos
modelo1 = LinearRegression()

# Ajustar el modelo a los datos
modelo1.fit(fechas1.values.astype(int).reshape(-1, 1), pm25_1)

# Hacer predicciones con el modelo
predicciones_pm25_1 = modelo1.predict(fechas1.values.astype(int).reshape(-1, 1))

# Cargar el segundo archivo CSV
file_path = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2019-07-01T00_00_00_2019-07-31T23_59_59.csv'

# Cargar el segundo conjunto de datos
df2 = pd.read_csv(file_path)

# Convertir las fechas a formato datetime si es necesario
if 'fecha_hora_med' in df2.columns:
    df2['fecha_hora_med'] = pd.to_datetime(df2['fecha_hora_med'])

# Eliminar filas con valores NaN en la columna de 'valor'
df2 = df2.dropna(subset=['valor'])

# Obtener las fechas y los valores
fechas2 = df2['fecha_hora_med']
valores2 = df2['valor']

# Crear el modelo de regresión lineal para el segundo conjunto de datos
modelo2 = LinearRegression()

# Ajustar el modelo a los datos
modelo2.fit(fechas2.values.astype(int).reshape(-1, 1), valores2)

# Hacer predicciones con el modelo
y_pred_2 = modelo2.predict(fechas2.values.astype(int).reshape(-1, 1))

# Visualizar los resultados en una sola figura
plt.figure(figsize=(10, 6))

# Graficar los datos y la regresión lineal del primer conjunto de datos
plt.scatter(fechas1, pm25_1, color='blue', label='Datos reales (Dataset 1)')
plt.plot(fechas1, predicciones_pm25_1, color='red', linewidth=2, label='Regresión lineal (Dataset 1)')

# Graficar los datos y la regresión lineal del segundo conjunto de datos
plt.scatter(fechas2, valores2, color='black', label='Datos reales (Dataset 2)')
plt.plot(fechas2, y_pred_2, color='green', linewidth=2, label='Regresión lineal (Dataset 2)')

plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación de Regresiones Lineales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Calcular la distancia entre las dos regresiones lineales
# Se puede usar la distancia Euclidiana entre los coeficientes de las rectas
distancia = np.sqrt((modelo1.coef_[0] - modelo2.coef_[0])**2 + (modelo1.intercept_ - modelo2.intercept_)**2)
print(f'Distancia entre las regresiones lineales: {distancia}')


# In[140]:


# Tamaños de ventana para el promedio móvil
tamanos_ventana = [3, 5, 7, 10]

# Configurar el número de filas y columnas para los subplots
num_filas = len(tamanos_ventana)
num_columnas = 2  # Dos subplots por fila
def promedio_movil(data, ventana):
    return data.rolling(window=ventana).mean()

# Configurar el tamaño de la figura
plt.figure(figsize=(15, 5 * num_filas))

for i, ventana in enumerate(tamanos_ventana):
    # Calcular el promedio móvil con ventana de tamaño actual para el primer conjunto de datos
    promedio_movil_df1 = promedio_movil(pm25_1, ventana=ventana)

    # Calcular el promedio móvil con ventana de tamaño actual para el segundo conjunto de datos
    promedio_movil_df2 = promedio_movil(valores2, ventana=ventana)

    # Calcular el promedio del promedio móvil con ventana de tamaño actual para el primer conjunto de datos
    promedio_promedio_movil_df1 = promedio_movil_df1.mean()

    # Calcular el promedio del promedio móvil con ventana de tamaño actual para el segundo conjunto de datos
    promedio_promedio_movil_df2 = promedio_movil_df2.mean()

    # Calcular la distancia entre los promedios móviles con ventana de tamaño actual
    distancia_promedios_moviles = abs(promedio_promedio_movil_df1 - promedio_promedio_movil_df2)
    
    # Configurar el subplot actual
    plt.subplot(num_filas, num_columnas, i + 1)

    # Graficar el promedio móvil para el primer conjunto de datos
    plt.plot(fechas1, promedio_movil_df1, label=f'Datos Estaciones AMB(ventana={ventana})', color='blue')

    # Graficar el promedio móvil para el segundo conjunto de datos
    plt.plot(fechas2, promedio_movil_df2, label=f'Conjunto de datos 2 (ventana={ventana})', color='red')

    plt.xlabel('Fecha')
    plt.ylabel(f'Promedio móvil (Ventana {ventana})')
    plt.title(f'Promedio Móvil con Ventana de Tamaño {ventana}')
    plt.xticks(rotation=45)
    plt.legend()

    # Agregar texto con los valores promedio y la distancia entre ellos
    plt.text(0.5, -0.3, f'Promedio móvil (Datos Estaciones AMB): {promedio_promedio_movil_df1:.2f}\nPromedio móvil (Mediciones2019-08-31): {promedio_promedio_movil_df2:.2f}\nDistancia entre los promedios móviles: {distancia_promedios_moviles:.2f}',
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

# Ajustar el diseño y mostrar la figura
plt.tight_layout()
plt.show()
# Encontrar el índice del mínimo en la lista de distancias
indice_mejor_ventana = distancias.index(min(distancias))

# Obtener el mejor tamaño de ventana
mejor_ventana = tamanos_ventana[indice_mejor_ventana]

# Imprimir el mejor tamaño de ventana
print(f"El mejor tamaño de ventana es: {mejor_ventana}")


# In[137]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar el primer archivo CSV limpiado
df1 = pd.read_csv('datos_limpios.csv')

# Seleccionar un rango de filas del DataFrame
df1_subset = df1.iloc[7303:8042].copy()

# Convertir las fechas a formato datetime
df1_subset['Date&Time'] = pd.to_datetime(df1_subset['Date&Time'])

# Eliminar filas con valores NaN en la columna de PM2.5
df1_subset = df1_subset.dropna(subset=['PM2.5'])

# Filtrar las filas que contienen '<Samp' en la columna 'PM2.5'
df1_subset = df1_subset[~df1_subset['PM2.5'].str.contains('<Samp')]

# Reemplazar las comas por puntos en la columna de PM2.5 y convertirla a flotantes
df1_subset['PM2.5'] = df1_subset['PM2.5'].str.replace(',', '.').astype(float)

# Obtener las fechas y los valores de PM2.5
fechas1 = df1_subset['Date&Time']
pm25_1 = df1_subset['PM2.5']

# Crear el modelo de regresión lineal para el primer conjunto de datos
modelo1 = LinearRegression()

# Ajustar el modelo a los datos
modelo1.fit(fechas1.values.astype(int).reshape(-1, 1), pm25_1)

# Hacer predicciones con el modelo
predicciones_pm25_1 = modelo1.predict(fechas1.values.astype(int).reshape(-1, 1))

# Cargar el segundo archivo CSV
file_path = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2019-08-01T00_00_00_2019-08-31T23_59_59.csv'

# Cargar el segundo conjunto de datos
df2 = pd.read_csv(file_path)

# Convertir las fechas a formato datetime si es necesario
if 'fecha_hora_med' in df2.columns:
    df2['fecha_hora_med'] = pd.to_datetime(df2['fecha_hora_med'])

# Eliminar filas con valores NaN en la columna de 'valor'
df2 = df2.dropna(subset=['valor'])

# Obtener las fechas y los valores
fechas2 = df2['fecha_hora_med']
valores2 = df2['valor']

# Crear el modelo de regresión lineal para el segundo conjunto de datos
modelo2 = LinearRegression()

# Ajustar el modelo a los datos
modelo2.fit(fechas2.values.astype(int).reshape(-1, 1), valores2)

# Hacer predicciones con el modelo
y_pred_2 = modelo2.predict(fechas2.values.astype(int).reshape(-1, 1))

# Visualizar los resultados en una sola figura
plt.figure(figsize=(10, 6))

# Graficar los datos y la regresión lineal del primer conjunto de datos
plt.scatter(fechas1, pm25_1, color='blue', label='Datos reales (Dataset 1)')
plt.plot(fechas1, predicciones_pm25_1, color='red', linewidth=2, label='Regresión lineal (Dataset 1)')

# Graficar los datos y la regresión lineal del segundo conjunto de datos
plt.scatter(fechas2, valores2, color='black', label='Datos reales (Dataset 2)')
plt.plot(fechas2, y_pred_2, color='green', linewidth=2, label='Regresión lineal (Dataset 2)')

plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación de Regresiones Lineales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Calcular la distancia entre las dos regresiones lineales
# Se puede usar la distancia Euclidiana entre los coeficientes de las rectas
distancia = np.sqrt((modelo1.coef_[0] - modelo2.coef_[0])**2 + (modelo1.intercept_ - modelo2.intercept_)**2)
print(f'Distancia entre las regresiones lineales: {distancia}')


# In[139]:


# Tamaños de ventana para el promedio móvil
tamanos_ventana = [3, 5, 7, 10]

# Configurar el número de filas y columnas para los subplots
num_filas = len(tamanos_ventana)
num_columnas = 2  # Dos subplots por fila
def promedio_movil(data, ventana):
    return data.rolling(window=ventana).mean()

# Configurar el tamaño de la figura
plt.figure(figsize=(15, 5 * num_filas))

for i, ventana in enumerate(tamanos_ventana):
    # Calcular el promedio móvil con ventana de tamaño actual para el primer conjunto de datos
    promedio_movil_df1 = promedio_movil(pm25_1, ventana=ventana)

    # Calcular el promedio móvil con ventana de tamaño actual para el segundo conjunto de datos
    promedio_movil_df2 = promedio_movil(valores2, ventana=ventana)

    # Calcular el promedio del promedio móvil con ventana de tamaño actual para el primer conjunto de datos
    promedio_promedio_movil_df1 = promedio_movil_df1.mean()

    # Calcular el promedio del promedio móvil con ventana de tamaño actual para el segundo conjunto de datos
    promedio_promedio_movil_df2 = promedio_movil_df2.mean()

    # Calcular la distancia entre los promedios móviles con ventana de tamaño actual
    distancia_promedios_moviles = abs(promedio_promedio_movil_df1 - promedio_promedio_movil_df2)
    
    # Configurar el subplot actual
    plt.subplot(num_filas, num_columnas, i + 1)

    # Graficar el promedio móvil para el primer conjunto de datos
    plt.plot(fechas1, promedio_movil_df1, label=f'Datos Estaciones AMB(ventana={ventana})', color='blue')

    # Graficar el promedio móvil para el segundo conjunto de datos
    plt.plot(fechas2, promedio_movil_df2, label=f'Conjunto de datos 2 (ventana={ventana})', color='red')

    plt.xlabel('Fecha')
    plt.ylabel(f'Promedio móvil (Ventana {ventana})')
    plt.title(f'Promedio Móvil con Ventana de Tamaño {ventana}')
    plt.xticks(rotation=45)
    plt.legend()

    # Agregar texto con los valores promedio y la distancia entre ellos
    plt.text(0.5, -0.3, f'Promedio móvil (Datos Estaciones AMB): {promedio_promedio_movil_df1:.2f}\nPromedio móvil (Mediciones2019-08-31): {promedio_promedio_movil_df2:.2f}\nDistancia entre los promedios móviles: {distancia_promedios_moviles:.2f}',
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

# Ajustar el diseño y mostrar la figura
plt.tight_layout()
plt.show()
# Encontrar el índice del mínimo en la lista de distancias
indice_mejor_ventana = distancias.index(min(distancias))

# Obtener el mejor tamaño de ventana
mejor_ventana = tamanos_ventana[indice_mejor_ventana]

# Imprimir el mejor tamaño de ventana
print(f"El mejor tamaño de ventana es: {mejor_ventana}")


# In[18]:





# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def cargar_datos(rango_df1, archivo_df2):
    df1 = pd.read_csv('datos_limpios.csv')
    df1_subset = df1.iloc[rango_df1].copy()
    df1_subset['Date&Time'] = pd.to_datetime(df1_subset['Date&Time'])
    df1_subset = df1_subset.dropna(subset=['PM2.5'])
    df1_subset = df1_subset[~df1_subset['PM2.5'].str.contains('<Samp')]
    df1_subset['PM2.5'] = df1_subset['PM2.5'].str.replace(',', '.').astype(float)
    df1_subset.loc[df1_subset['PM2.5'].isnull(), 'PM2.5'] = np.random.uniform(8, 14, df1_subset['PM2.5'].isnull().sum())
    fechas1 = df1_subset['Date&Time']
    pm25_1 = df1_subset['PM2.5']

    df2 = pd.read_csv(archivo_df2)
    if 'fecha_hora_med' in df2.columns:
        df2['fecha_hora_med'] = pd.to_datetime(df2['fecha_hora_med'])
    df2 = df2.dropna(subset=['valor'])
    df2_subset = df2.sample(n=len(pm25_1), replace=True, random_state=0)
    valores2 = df2_subset['valor']

    return pm25_1, valores2

def entrenar_modelos(X, Y):
    X = X.values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    rf_model = RandomForestRegressor(n_estimators=1000, random_state=0, max_features="sqrt", criterion="squared_error").fit(X_train, y_train)
    lr_model = LinearRegression().fit(X_train, y_train)
    return X_test, y_test, rf_model, lr_model

def evaluar_modelos(X_test, y_test, rf_model, lr_model):
    def evaluar(modelo, predicciones):
        mae = mean_absolute_error(y_test, predicciones)
        rmse = np.sqrt(mean_squared_error(y_test, predicciones))
        accuracy = 100 * (1 - (mae / np.mean(y_test)))
        return mae, rmse, accuracy

    mae_rf, rmse_rf, accuracy_rf = evaluar("Random Forest", rf_model.predict(X_test))
    mae_lr, rmse_lr, accuracy_lr = evaluar("Linear Regression", lr_model.predict(X_test))
    return mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr

def visualizar_resultados(X_test, y_test, predictions_rf, predictions_lr, mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr):
    def visualizar(titulo, predicciones, color):
        plt.scatter(X_test, y_test, color='blue', label='Valores reales')
        plt.scatter(X_test, predicciones, color=color, label=f'Predicciones ({titulo})')
        plt.xlabel('Valores Medidos')
        plt.ylabel('Valores Predichos')
        plt.legend()
        plt.title(f'Predicciones vs. Valores Reales ({titulo})')
        plt.text(0.5, -2, f'MAE: {mae_rf:.2f}\nRMSE: {rmse_rf:.2f}\nAccuracy: {accuracy_rf:.2f}%')
        plt.show()

    visualizar("Random Forest", predictions_rf, 'red')
    visualizar("Linear Regression", predictions_lr, 'green')

def main():
    rango_df1 = slice(747, 1466)
    archivo_df2 = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2018-11-01T00_00_00_2018-11-30T23_59_59.csv'

    pm25_1, valores2 = cargar_datos(rango_df1, archivo_df2)
    X_test, y_test, rf_model, lr_model = entrenar_modelos(valores2, pm25_1)
    mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr = evaluar_modelos(X_test, y_test, rf_model, lr_model)
    visualizar_resultados(X_test, y_test, rf_model.predict(X_test), lr_model.predict(X_test), mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr)

if __name__ == "__main__":
    main()


# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def cargar_datos(rango_df1, archivo_df2):
    df1 = pd.read_csv('datos_limpios.csv')
    df1_subset = df1.iloc[rango_df1].copy()
    df1_subset['Date&Time'] = pd.to_datetime(df1_subset['Date&Time'])
    df1_subset = df1_subset.dropna(subset=['PM2.5'])
    df1_subset = df1_subset[~df1_subset['PM2.5'].str.contains('<Samp')]
    df1_subset['PM2.5'] = df1_subset['PM2.5'].str.replace(',', '.').astype(float)
    df1_subset.loc[df1_subset['PM2.5'].isnull(), 'PM2.5'] = np.random.uniform(8, 14, df1_subset['PM2.5'].isnull().sum())
    fechas1 = df1_subset['Date&Time']
    pm25_1 = df1_subset['PM2.5']

    df2 = pd.read_csv(archivo_df2)
    if 'fecha_hora_med' in df2.columns:
        df2['fecha_hora_med'] = pd.to_datetime(df2['fecha_hora_med'])
    df2 = df2.dropna(subset=['valor'])
    df2_subset = df2.sample(n=len(pm25_1), replace=True, random_state=0)
    valores2 = df2_subset['valor']

    return pm25_1, valores2

def entrenar_modelos(X, Y):
    X = X.values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    rf_model = RandomForestRegressor(n_estimators=1000, random_state=0, max_features="sqrt", criterion="squared_error").fit(X_train, y_train)
    lr_model = LinearRegression().fit(X_train, y_train)
    return X_test, y_test, rf_model, lr_model

def evaluar_modelos(X_test, y_test, rf_model, lr_model):
    def evaluar(modelo, predicciones):
        mae = mean_absolute_error(y_test, predicciones)
        rmse = np.sqrt(mean_squared_error(y_test, predicciones))
        accuracy = 100 * (1 - (mae / np.mean(y_test)))
        return mae, rmse, accuracy

    mae_rf, rmse_rf, accuracy_rf = evaluar("Random Forest", rf_model.predict(X_test))
    mae_lr, rmse_lr, accuracy_lr = evaluar("Linear Regression", lr_model.predict(X_test))
    return mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr

def visualizar_resultados(X_test, y_test, predictions_rf, predictions_lr, mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr):
    def visualizar(titulo, predicciones, color):
        plt.scatter(X_test, y_test, color='blue', label='Valores reales')
        plt.scatter(X_test, predicciones, color=color, label=f'Predicciones ({titulo})')
        plt.xlabel('Valores Medidos')
        plt.ylabel('Valores Predichos')
        plt.legend()
        plt.title(f'Predicciones vs. Valores Reales ({titulo})')
        plt.text(0.5, -2, f'MAE: {mae_rf:.2f}\nRMSE: {rmse_rf:.2f}\nAccuracy: {accuracy_rf:.2f}%')
        plt.show()

    visualizar("Random Forest", predictions_rf, 'red')
    visualizar("Linear Regression", predictions_lr, 'green')

def main():
    rango_df1 = slice(1467, 1802)
    archivo_df2 = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2018-12-01T00_00_00_2018-12-31T23_59_59.csv'

    pm25_1, valores2 = cargar_datos(rango_df1, archivo_df2)
    X_test, y_test, rf_model, lr_model = entrenar_modelos(valores2, pm25_1)
    mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr = evaluar_modelos(X_test, y_test, rf_model, lr_model)
    visualizar_resultados(X_test, y_test, rf_model.predict(X_test), lr_model.predict(X_test), mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr)

if __name__ == "__main__":
    main()


# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def cargar_datos(rango_df1, archivo_df2):
    df1 = pd.read_csv('datos_limpios.csv')
    df1_subset = df1.iloc[rango_df1].copy()
    df1_subset['Date&Time'] = pd.to_datetime(df1_subset['Date&Time'])
    df1_subset = df1_subset.dropna(subset=['PM2.5'])
    df1_subset = df1_subset[~df1_subset['PM2.5'].str.contains('<Samp')]
    df1_subset['PM2.5'] = df1_subset['PM2.5'].str.replace(',', '.').astype(float)
    df1_subset.loc[df1_subset['PM2.5'].isnull(), 'PM2.5'] = np.random.uniform(8, 14, df1_subset['PM2.5'].isnull().sum())
    fechas1 = df1_subset['Date&Time']
    pm25_1 = df1_subset['PM2.5']

    df2 = pd.read_csv(archivo_df2)
    if 'fecha_hora_med' in df2.columns:
        df2['fecha_hora_med'] = pd.to_datetime(df2['fecha_hora_med'])
    df2 = df2.dropna(subset=['valor'])
    df2_subset = df2.sample(n=len(pm25_1), replace=True, random_state=0)
    valores2 = df2_subset['valor']

    return pm25_1, valores2

def entrenar_modelos(X, Y):
    X = X.values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    rf_model = RandomForestRegressor(n_estimators=1000, random_state=0, max_features="sqrt", criterion="squared_error").fit(X_train, y_train)
    lr_model = LinearRegression().fit(X_train, y_train)
    return X_test, y_test, rf_model, lr_model

def evaluar_modelos(X_test, y_test, rf_model, lr_model):
    def evaluar(modelo, predicciones):
        mae = mean_absolute_error(y_test, predicciones)
        rmse = np.sqrt(mean_squared_error(y_test, predicciones))
        accuracy = 100 * (1 - (mae / np.mean(y_test)))
        return mae, rmse, accuracy

    mae_rf, rmse_rf, accuracy_rf = evaluar("Random Forest", rf_model.predict(X_test))
    mae_lr, rmse_lr, accuracy_lr = evaluar("Linear Regression", lr_model.predict(X_test))
    return mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr

def visualizar_resultados(X_test, y_test, predictions_rf, predictions_lr, mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr):
    def visualizar(titulo, predicciones, color):
        plt.scatter(X_test, y_test, color='blue', label='Valores reales')
        plt.scatter(X_test, predicciones, color=color, label=f'Predicciones ({titulo})')
        plt.xlabel('Valores Medidos')
        plt.ylabel('Valores Predichos')
        plt.legend()
        plt.title(f'Predicciones vs. Valores Reales ({titulo})')
        plt.text(0.5, -2, f'MAE: {mae_rf:.2f}\nRMSE: {rmse_rf:.2f}\nAccuracy: {accuracy_rf:.2f}%')
        plt.show()

    visualizar("Random Forest", predictions_rf, 'red')
    visualizar("Linear Regression", predictions_lr, 'green')

def main():
    rango_df1 = slice(4628, 5095)
    archivo_df2 = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2019-04-01T00_00_00_2019-04-30T23_59_59.csv'

    pm25_1, valores2 = cargar_datos(rango_df1, archivo_df2)
    X_test, y_test, rf_model, lr_model = entrenar_modelos(valores2, pm25_1)
    mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr = evaluar_modelos(X_test, y_test, rf_model, lr_model)
    visualizar_resultados(X_test, y_test, rf_model.predict(X_test), lr_model.predict(X_test), mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr)

if __name__ == "__main__":
    main()


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def cargar_datos(rango_df1, archivo_df2):
    df1 = pd.read_csv('datos_limpios.csv')
    df1_subset = df1.iloc[rango_df1].copy()
    df1_subset['Date&Time'] = pd.to_datetime(df1_subset['Date&Time'])
    df1_subset = df1_subset.dropna(subset=['PM2.5'])
    df1_subset = df1_subset[~df1_subset['PM2.5'].str.contains('<Samp')]
    df1_subset['PM2.5'] = df1_subset['PM2.5'].str.replace(',', '.').astype(float)
    df1_subset.loc[df1_subset['PM2.5'].isnull(), 'PM2.5'] = np.random.uniform(8, 14, df1_subset['PM2.5'].isnull().sum())
    fechas1 = df1_subset['Date&Time']
    pm25_1 = df1_subset['PM2.5']

    df2 = pd.read_csv(archivo_df2)
    if 'fecha_hora_med' in df2.columns:
        df2['fecha_hora_med'] = pd.to_datetime(df2['fecha_hora_med'])
    df2 = df2.dropna(subset=['valor'])
    df2_subset = df2.sample(n=len(pm25_1), replace=True, random_state=0)
    valores2 = df2_subset['valor']

    return pm25_1, valores2

def entrenar_modelos(X, Y):
    X = X.values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    rf_model = RandomForestRegressor(n_estimators=1000, random_state=0, max_features="sqrt", criterion="squared_error").fit(X_train, y_train)
    lr_model = LinearRegression().fit(X_train, y_train)
    return X_test, y_test, rf_model, lr_model

def evaluar_modelos(X_test, y_test, rf_model, lr_model):
    def evaluar(modelo, predicciones):
        mae = mean_absolute_error(y_test, predicciones)
        rmse = np.sqrt(mean_squared_error(y_test, predicciones))
        accuracy = 100 * (1 - (mae / np.mean(y_test)))
        return mae, rmse, accuracy

    mae_rf, rmse_rf, accuracy_rf = evaluar("Random Forest", rf_model.predict(X_test))
    mae_lr, rmse_lr, accuracy_lr = evaluar("Linear Regression", lr_model.predict(X_test))
    return mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr

def visualizar_resultados(X_test, y_test, predictions_rf, predictions_lr, mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr):
    def visualizar(titulo, predicciones, color):
        plt.scatter(X_test, y_test, color='blue', label='Valores reales')
        plt.scatter(X_test, predicciones, color=color, label=f'Predicciones ({titulo})')
        plt.xlabel('Valores Medidos')
        plt.ylabel('Valores Predichos')
        plt.legend()
        plt.title(f'Predicciones vs. Valores Reales ({titulo})')
        plt.text(0.5, -2, f'MAE: {mae_rf:.2f}\nRMSE: {rmse_rf:.2f}\nAccuracy: {accuracy_rf:.2f}%')
        plt.show()

    visualizar("Random Forest", predictions_rf, 'red')
    visualizar("Linear Regression", predictions_lr, 'green')

def main():
    rango_df1 = slice(5096, 5839)
    archivo_df2 = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2019-05-01T00_00_00_2019-05-31T23_59_59.csv'

    pm25_1, valores2 = cargar_datos(rango_df1, archivo_df2)
    X_test, y_test, rf_model, lr_model = entrenar_modelos(valores2, pm25_1)
    mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr = evaluar_modelos(X_test, y_test, rf_model, lr_model)
    visualizar_resultados(X_test, y_test, rf_model.predict(X_test), lr_model.predict(X_test), mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr)

if __name__ == "__main__":
    main()


# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def cargar_datos(rango_df1, archivo_df2):
    df1 = pd.read_csv('datos_limpios.csv')
    df1_subset = df1.iloc[rango_df1].copy()
    df1_subset['Date&Time'] = pd.to_datetime(df1_subset['Date&Time'])
    df1_subset = df1_subset.dropna(subset=['PM2.5'])
    df1_subset = df1_subset[~df1_subset['PM2.5'].str.contains('<Samp')]
    df1_subset['PM2.5'] = df1_subset['PM2.5'].str.replace(',', '.').astype(float)
    df1_subset.loc[df1_subset['PM2.5'].isnull(), 'PM2.5'] = np.random.uniform(8, 14, df1_subset['PM2.5'].isnull().sum())
    fechas1 = df1_subset['Date&Time']
    pm25_1 = df1_subset['PM2.5']

    df2 = pd.read_csv(archivo_df2)
    if 'fecha_hora_med' in df2.columns:
        df2['fecha_hora_med'] = pd.to_datetime(df2['fecha_hora_med'])
    df2 = df2.dropna(subset=['valor'])
    df2_subset = df2.sample(n=len(pm25_1), replace=True, random_state=0)
    valores2 = df2_subset['valor']

    return pm25_1, valores2

def entrenar_modelos(X, Y):
    X = X.values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    rf_model = RandomForestRegressor(n_estimators=1000, random_state=0, max_features="sqrt", criterion="squared_error").fit(X_train, y_train)
    lr_model = LinearRegression().fit(X_train, y_train)
    return X_test, y_test, rf_model, lr_model

def evaluar_modelos(X_test, y_test, rf_model, lr_model):
    def evaluar(modelo, predicciones):
        mae = mean_absolute_error(y_test, predicciones)
        rmse = np.sqrt(mean_squared_error(y_test, predicciones))
        accuracy = 100 * (1 - (mae / np.mean(y_test)))
        return mae, rmse, accuracy

    mae_rf, rmse_rf, accuracy_rf = evaluar("Random Forest", rf_model.predict(X_test))
    mae_lr, rmse_lr, accuracy_lr = evaluar("Linear Regression", lr_model.predict(X_test))
    return mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr

def visualizar_resultados(X_test, y_test, predictions_rf, predictions_lr, mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr):
    def visualizar(titulo, predicciones, color):
        plt.scatter(X_test, y_test, color='blue', label='Valores reales')
        plt.scatter(X_test, predicciones, color=color, label=f'Predicciones ({titulo})')
        plt.xlabel('Valores Medidos')
        plt.ylabel('Valores Predichos')
        plt.legend()
        plt.title(f'Predicciones vs. Valores Reales ({titulo})')
        plt.text(0.5, -2, f'MAE: {mae_rf:.2f}\nRMSE: {rmse_rf:.2f}\nAccuracy: {accuracy_rf:.2f}%')
        plt.show()

    visualizar("Random Forest", predictions_rf, 'red')
    visualizar("Linear Regression", predictions_lr, 'green')

def main():
    rango_df1 = slice(5841, 6559)
    archivo_df2 = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2019-06-01T00_00_00_2019-06-30T23_59_59.csv'

    pm25_1, valores2 = cargar_datos(rango_df1, archivo_df2)
    X_test, y_test, rf_model, lr_model = entrenar_modelos(valores2, pm25_1)
    mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr = evaluar_modelos(X_test, y_test, rf_model, lr_model)
    visualizar_resultados(X_test, y_test, rf_model.predict(X_test), lr_model.predict(X_test), mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr)

if __name__ == "__main__":
    main()


# In[26]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def cargar_datos(rango_df1, archivo_df2):
    df1 = pd.read_csv('datos_limpios.csv')
    df1_subset = df1.iloc[rango_df1].copy()
    df1_subset['Date&Time'] = pd.to_datetime(df1_subset['Date&Time'])
    df1_subset = df1_subset.dropna(subset=['PM2.5'])
    df1_subset = df1_subset[~df1_subset['PM2.5'].str.contains('<Samp')]
    df1_subset['PM2.5'] = df1_subset['PM2.5'].str.replace(',', '.').astype(float)
    df1_subset.loc[df1_subset['PM2.5'].isnull(), 'PM2.5'] = np.random.uniform(8, 14, df1_subset['PM2.5'].isnull().sum())
    fechas1 = df1_subset['Date&Time']
    pm25_1 = df1_subset['PM2.5']

    df2 = pd.read_csv(archivo_df2)
    if 'fecha_hora_med' in df2.columns:
        df2['fecha_hora_med'] = pd.to_datetime(df2['fecha_hora_med'])
    df2 = df2.dropna(subset=['valor'])
    df2_subset = df2.sample(n=len(pm25_1), replace=True, random_state=0)
    valores2 = df2_subset['valor']

    return pm25_1, valores2

def entrenar_modelos(X, Y):
    X = X.values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    rf_model = RandomForestRegressor(n_estimators=1000, random_state=0, max_features="sqrt", criterion="squared_error").fit(X_train, y_train)
    lr_model = LinearRegression().fit(X_train, y_train)
    return X_test, y_test, rf_model, lr_model

def evaluar_modelos(X_test, y_test, rf_model, lr_model):
    def evaluar(modelo, predicciones):
        mae = mean_absolute_error(y_test, predicciones)
        rmse = np.sqrt(mean_squared_error(y_test, predicciones))
        accuracy = 100 * (1 - (mae / np.mean(y_test)))
        return mae, rmse, accuracy

    mae_rf, rmse_rf, accuracy_rf = evaluar("Random Forest", rf_model.predict(X_test))
    mae_lr, rmse_lr, accuracy_lr = evaluar("Linear Regression", lr_model.predict(X_test))
    return mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr

def visualizar_resultados(X_test, y_test, predictions_rf, predictions_lr, mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr):
    def visualizar(titulo, predicciones, color):
        plt.scatter(X_test, y_test, color='blue', label='Valores reales')
        plt.scatter(X_test, predicciones, color=color, label=f'Predicciones ({titulo})')
        plt.xlabel('Valores Medidos')
        plt.ylabel('Valores Predichos')
        plt.legend()
        plt.title(f'Predicciones vs. Valores Reales ({titulo})')
        plt.text(0.5, -2, f'MAE: {mae_rf:.2f}\nRMSE: {rmse_rf:.2f}\nAccuracy: {accuracy_rf:.2f}%')
        plt.show()

    visualizar("Random Forest", predictions_rf, 'red')
    visualizar("Linear Regression", predictions_lr, 'green')

def main():
    rango_df1 = slice(6560, 7303)
    archivo_df2 = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2019-07-01T00_00_00_2019-07-31T23_59_59.csv'

    pm25_1, valores2 = cargar_datos(rango_df1, archivo_df2)
    X_test, y_test, rf_model, lr_model = entrenar_modelos(valores2, pm25_1)
    mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr = evaluar_modelos(X_test, y_test, rf_model, lr_model)
    visualizar_resultados(X_test, y_test, rf_model.predict(X_test), lr_model.predict(X_test), mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr)

if __name__ == "__main__":
    main()


# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def cargar_datos(rango_df1, archivo_df2):
    df1 = pd.read_csv('datos_limpios.csv')
    df1_subset = df1.iloc[rango_df1].copy()
    df1_subset['Date&Time'] = pd.to_datetime(df1_subset['Date&Time'])
    df1_subset = df1_subset.dropna(subset=['PM2.5'])
    df1_subset = df1_subset[~df1_subset['PM2.5'].str.contains('<Samp')]
    df1_subset['PM2.5'] = df1_subset['PM2.5'].str.replace(',', '.').astype(float)
    df1_subset.loc[df1_subset['PM2.5'].isnull(), 'PM2.5'] = np.random.uniform(8, 14, df1_subset['PM2.5'].isnull().sum())
    fechas1 = df1_subset['Date&Time']
    pm25_1 = df1_subset['PM2.5']

    df2 = pd.read_csv(archivo_df2)
    if 'fecha_hora_med' in df2.columns:
        df2['fecha_hora_med'] = pd.to_datetime(df2['fecha_hora_med'])
    df2 = df2.dropna(subset=['valor'])
    df2_subset = df2.sample(n=len(pm25_1), replace=True, random_state=0)
    valores2 = df2_subset['valor']

    return pm25_1, valores2

def entrenar_modelos(X, Y):
    X = X.values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    rf_model = RandomForestRegressor(n_estimators=1000, random_state=0, max_features="sqrt", criterion="squared_error").fit(X_train, y_train)
    lr_model = LinearRegression().fit(X_train, y_train)
    return X_test, y_test, rf_model, lr_model

def evaluar_modelos(X_test, y_test, rf_model, lr_model):
    def evaluar(modelo, predicciones):
        mae = mean_absolute_error(y_test, predicciones)
        rmse = np.sqrt(mean_squared_error(y_test, predicciones))
        accuracy = 100 * (1 - (mae / np.mean(y_test)))
        return mae, rmse, accuracy

    mae_rf, rmse_rf, accuracy_rf = evaluar("Random Forest", rf_model.predict(X_test))
    mae_lr, rmse_lr, accuracy_lr = evaluar("Linear Regression", lr_model.predict(X_test))
    return mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr

def visualizar_resultados(X_test, y_test, predictions_rf, predictions_lr, mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr):
    def visualizar(titulo, predicciones, color):
        plt.scatter(X_test, y_test, color='blue', label='Valores reales')
        plt.scatter(X_test, predicciones, color=color, label=f'Predicciones ({titulo})')
        plt.xlabel('Valores Medidos')
        plt.ylabel('Valores Predichos')
        plt.legend()
        plt.title(f'Predicciones vs. Valores Reales ({titulo})')
        plt.text(0.5, -2, f'MAE: {mae_rf:.2f}\nRMSE: {rmse_rf:.2f}\nAccuracy: {accuracy_rf:.2f}%')
        plt.show()

    visualizar("Random Forest", predictions_rf, 'red')
    visualizar("Linear Regression", predictions_lr, 'green')

def main():
    rango_df1 = slice(7303, 8042)
    archivo_df2 = r'\Users\jeico\Downloads\data\mediciones_clg_normalsup_pm25_a_2019-08-01T00_00_00_2019-08-31T23_59_59.csv'

    pm25_1, valores2 = cargar_datos(rango_df1, archivo_df2)
    X_test, y_test, rf_model, lr_model = entrenar_modelos(valores2, pm25_1)
    mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr = evaluar_modelos(X_test, y_test, rf_model, lr_model)
    visualizar_resultados(X_test, y_test, rf_model.predict(X_test), lr_model.predict(X_test), mae_rf, rmse_rf, accuracy_rf, mae_lr, rmse_lr, accuracy_lr)

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




