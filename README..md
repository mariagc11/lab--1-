# lab--1-

## INTRODUCCIÓN:
Para el desarrollo de este informe de laboratorio, se realizó un análisis estadistico de señanes biomédicas extraidas de la Pagina Physionet
y seleccionamos una señal EMG; esta herramienta es fundamental ya que por medio de ella se extrae información relevante con datos para permitir llegar al desarrollo de esta practica.
Para la programación, se utilizo la herramienta python que permite implementar algoritmos para calcular y analizar los datos proporcionados por la señal, graficarlos, contaminar la señal, y otras opciones más que se observarán en éste informe.
## Descripción de los datos:
Los datos se recopilaron con un sistema de monitorización EMG Medelec Synergy N2 (Oxford Instruments Medical, Old Woking, Reino Unido). Se colocó un electrodo de aguja concéntrico de 25 mm en el músculo tibial anterior de cada sujeto. A continuación, se le pidió al paciente que flexionara el pie suavemente contra resistencia. El electrodo de aguja se reposicionó hasta que se identificaron potenciales de unidad motora con un tiempo de ascenso rápido. A continuación, se recopilaron datos durante varios segundos, momento en el que se le pidió al paciente que se relajara y se retiró la aguja.
## Paso a paso:
- Seleccionar la señal EMG por medio de Physionet [link Physionet](https://physionet.org/)
- Guardar los archivos .hea, .data en una misma carpeta junto con la señal
- Abrir Python, nombrar el archivo y guardarlo en la misma carpeta donde se encuentran los archivos .hea y .data
- Abrir de nuevo python y iniciar con la programación que explicaremos a continuación:
  
## Programación y Datos estadísticos:
Inicialmente agregamos las librerias:
```  import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm.
```


- **wfdb** Es una libreria que permite trabajar con bases de datos, en este caso señales fisiologicas de PrysiNet.
Esta libreria permite leer, escribir y analizar señales fisiologicas
- **Numpy** Esta librería es fundamental para en la programación poder utilizar sintaxis numericas en python, permitiendo arreglos multidimencionalesy operaciones matematicas
- **Matplotlib.pyplot(plt)** Se usa para graficar los datos
- **Scipy.stats.norm** Se usa para trabajar con distribuciones normales y ajustar curvas.

Éstas librerias son fundamentales porque sin ellas tendriamos muchos errores al momento de usar operciones matematicas y correr el codigo.  

#### Cargar la señal EMG desde el archivo
```
record = wfdb.rdrecord('emg_healthy')
señal = record.p_signal[:, 0] * 0.5  
fs = record.fs * 0.5  
time = np.arange(len(señal)) / fs  

```

- **record:** Esta funcion especifica la ruta del archivo que contiene la señal EMG de la libreria wfdb lo que permite leer el archivo con los datos llamada 'emg_neuropathy.

- **señal:** Se utiliza esta función matriz donde cada columna representa un canal de la señal registrada. la expresion [:,0] extrae la primera columna, los valores iniciales y se multiplica por 0.5 reduciendo la amplitud a la mitad.

- **fs:** Es un diccionario que contiene metadatos asociados con la señal, como la frecuencia de muestreo ( fs), nombres de los canales, unidades, entre otros.

#### Ajuste intervalo 
```
tiempo_limite = 8  # en segundos
indice_limite = int(tiempo_limite * fs)
señal = señal[:indice_limite]
time = time[:indice_limite]
```
- **tiempo_limite:** esta funcion permite tomar in intervalo de n segundos para establecer la duracion maxima de la señal que se tomará
- **indice_limite:** la funcion fs representa la frecuencia de muestreo en hz (muestras por segundo multiplicando el tiempo limite por fs y obteniendo la cantidad total de muestras.
  
#### Graficar la señal original

```
plt.figure(figsize=(12, 6))
plt.plot(time, señal, label="Señal EMG Neuropathy", color='blue')
plt.title("Señal EMG Neuropathy")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (mV)")
plt.grid()
plt.legend()
plt.show()

```
- **plt.figure( figsize=(12,6))** Se crea una figura grande para visualizar la señal claramente.
- **plt.plot(time, señal)**: Se grafica la señal en función del tiempo.
- **plt.grid()**: Se agrega una cuadrícula para mejorar la visualización.
- **plt.legend()**: Se muestra la etiqueta de la señal.

<img width="1404" height="702" alt="image" src="https://github.com/user-attachments/assets/33eec203-f5f6-4ccb-afeb-1c86cc0f32e5" />

#### Cálculo: 
```
suma = 0
for x in señal:
    suma += x
media = suma / len(señal)
```
Aquí se calcula manualmente la media (promedio) de la señal.

#### Cálculo de desviación estándar 
```
suma = 0
for x in señal:
    suma += (x - media) ** 2
desviacion = (suma / len(señal))
```
Se calcula manualmente la desviación estándar

#### Cálculo del coeficiente de variación
```
coef = (desviacion / media) * 100
```
Se calcula el coeficiente de variación , que mide la dispersión en porcentaje.

#### Usando funciones predefinidas
```
media = np.mean(señal)
desviacion = np.std(señal)
coef = (desviacion / media) * 100

```
Aquí se usan las funciones de numpypara hacer lo mismo de forma más eficiente.

#### Mostrar resultados
```
print("\nMDC")
print("Calculos:")
print(f"Media: {media:.4f}")
print(f"Desviación estándar: {desviacion:.4f}")
print(f"Coeficiente de variación: {coef:.2f} %")

print("\nCalculos con funciones:")
print(f"Media: {media:.4f}")
print(f"Desviación estándar: {desviacion:.4f}")
print(f"Coeficiente de variación: {coef:.2f} %")

```
Se imprimen, organiza y presenta cálculos de forma legible, permitiendo verificar que los métodos utilizados para calcular la media, la desviación estándar y el coeficiente de variación sean correctos.

<img width="426" height="205" alt="image" src="https://github.com/user-attachments/assets/4a331170-a3e7-4299-a670-a859e7f27dc6" />


#### Histograma con PDF ajustada
```

plt.figure(figsize=(12, 6))
plt.hist(señal, bins=50, density=True, alpha=0.75, color='blue', label="Histograma")
```

Un histograma muestra la distribución de los valores de la señal. En este caso, se usa para visualizar cómo están distribuidos los valores de la señal EMG y compararlos con una curva de distribución normal.

#### Ajuste de una distribución normal (PDF)
```
mu, sigma = norm.fit(señal)
pdf_x = np.linspace(min(señal), max(señal), 1000)
pdf_y = norm.pdf(pdf_x, mu, sigma)
factor_escala = 8.2 / max(pdf_y)
pdf_y *= factor_escala
plt.plot(pdf_x, pdf_y, 'k-', label=f"Curva G")

plt.title("Histograma de la Señal EMG")
plt.xlabel("Amplitud")
plt.ylabel("Frecuencia Normalizada")
plt.legend()
plt.grid()
plt.show()

```
Se ajusta una distribución normal a los datos de una señal EMG y la grafica sobre un histograma. Primero, estima la media y la desviación estándar de la señal, luego genera y escala una curva normal, y finalmente la grafica junto con las barras del histograma para analizar cómo se distribuyen los valores de la señal.

!<img width="1438" height="717" alt="image" src="https://github.com/user-attachments/assets/e2a76361-d3b0-421c-a98f-df016f6f90dc" />


#### Función de probabilidad acumulativa (CDF)
```
sorted_data = np.sort(señal)
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

plt.figure(figsize=(10, 6))
plt.plot(sorted_data, cdf, label="CDF (Empírica)")
plt.title("Función de Probabilidad Acumulativa (CDF)")
plt.xlabel("Amplitud")
plt.ylabel("Probabilidad acumulada")
plt.grid()
plt.legend()
plt.show()

```

Este código calcula y grafica la Función de Probabilidad Acumulativa (CDF) de una señal. Primero, ordena los datos con np.sort(señal), luego calcula la CDF dividiendo los índices acumulados por la cantidad total de datos, asegurando valores entre 0 y 1.

Se configura la figura con plt.figure(figsize=(10,6)) y se grafica la CDF con plt.plot(sorted_data, cdf, label="CDF (Empírica)"), donde el eje X representa la amplitud de la señal y el Y la probabilidad acumulada. Finalmente, se agregan título, etiquetas, cuadrícula y leyenda para mejorar la visualización, mostrando cómo se distribuyen los valores de la señal.

<img width="1191" height="708" alt="image" src="https://github.com/user-attachments/assets/3d8074dd-73ee-4887-8ff0-9904cc4968d7" />

### Que es el SNR?

El SNR (Signal-to-Noise Ratio) es una medida que nos dice qué tan fuerte es una señal en comparación con el ruido que la acompaña. En términos simples, es como tratar de escuchar a alguien hablar en una fiesta ruidosa: si la voz es clara y fuerte en comparación con el ruido de fondo, el SNR es alto; si apenas se distingue lo que dice entre todo el ruido, el SNR es bajo.

### ¿Cómo se calcula?
Se obtiene con la siguiente fórmula:

![Imagen de WhatsApp 2025-02-06 a las 22 56 18_28fd9c72](https://github.com/user-attachments/assets/2738b8e3-aae9-4fc0-9b87-0fdeb704fe08)


Esto significa que tomamos la potencia promedio de la señal (su energía) y la comparamos con la potencia promedio del ruido. Luego, aplicamos un logaritmo para expresarlo en decibeles (dB), que es una escala más útil para interpretar los valores.

### Función para calcular SNR

```
def calcular_snr(señal , ruido):
    potencia_senal = np.mean(señal ** 2)
    potencia_ruido = np.mean(ruido ** 2)
    snr = 10 * np.log10(potencia_senal / potencia_ruido)
    return snr

  ```

- Calcula la potencia media de la señal (potencia_senal).
- Calcula la potencia media del ruido (potencia_ruido).
- Calcula el SNR en decibeles (dB) con la fórmula


#### Función para agregar ruido 

```
def agregar_ruido(señal, tipo="gaussiano", intensidad=0.05, frecuencia=50, porcentaje=0.05, time=None):
    if tipo == "gaussiano":
        ruido = np.random.normal(0, intensidad, len(señal))
    elif tipo == "impulso":
        ruido = np.zeros(len(señal))
        num_impulsos = int(porcentaje * len(señal))
        indices = np.random.randint(0, len(señal), num_impulsos)
        ruido[indices] = np.random.choice([-1, 1], size=num_impulsos) * np.max(señal) * 0.5
    elif tipo == "artefacto" and time is not None:
        ruido = intensidad * np.sin(2 * np.pi * frecuencia * time)
    else:
        raise ValueError("Tipo de ruido no válido")
    return señal + ruido, calcular_snr(señal, ruido)

  ```
Parámetros:

- tipo: Tipo de ruido a añadir ("gaussiano", "impulso", "artefacto").
- intensidad: Nivel de ruido en los casos de gaussiano y artefacto.
- frecuencia: Frecuencia del ruido de artefacto (50 Hz).
- porcentaje: Proporción de muestras con ruido de impulso.
- time: Vector de tiempo (necesario para el ruido de artefacto).

Ruido Gaussiano:
  
Se genera ruido blanco gaussiano con np.random.normal(), de media 0 y desviación intensidad.
Este tipo de ruido es similar al ruido térmico en señales fisiológicas.

Ruido de impulsos:

Este ruido simula picos aleatorios en la señal:

- Se crea un array de ceros del mismo tamaño que la señal.
- Se calcula el número de impulsos a introducir (num_impulsos).
- Se generan posiciones aleatorias (indices).
- Se asignan valores positivos o negativos (0.5 * max(señal)) en esas posiciones.

Ruido de artefacto (interferencia senoidal):
  
 - Se crea un ruido senoidal con frecuencia frecuencia (50 Hz).

  
Esta función agrega ruido a una señal y calcula su relación señal-ruido (SNR). Dependiendo del tipo de ruido seleccionado, puede ser: gaussiano, generado con una distribución normal; impulsivo, donde se insertan valores aleatorios en ciertos puntos; o artefacto, una señal sinusoidal de una frecuencia específica. Si el tipo no es válido, se genera un error. Finalmente, la señal modificada y su SNR se retornan, permitiendo evaluar el impacto del ruido en la señal original.

#### Contaminación de la señal
```
senal_gaussiana, snr_gaussiano = agregar_ruido(señal, tipo="gaussiano")
senal_impulso, snr_impulso = agregar_ruido(señal, tipo="impulso")
senal_artefacto, snr_artefacto = agregar_ruido(señal, tipo="artefacto", time=time)

```
#### Mostrar valores de SNR
```
print("\nRelación Señal-Ruido (SNR)")
for tipo, snr in zip("Gaussiano", "Impulso", "Artefacto"], [snr_gaussiano, snr_impulso, snr_artefacto):
    print(f"SNR con ruido {tipo.lower()}: {snr:.2f} dB")
```
#### Graficar señales con ruido en gráficos separados junto con la señal original
```
ruidos = ("Ruido Gaussiano", senal_gaussiana, snr_gaussiano, "red"),
          ("Ruido Impulso", senal_impulso, snr_impulso, "green"),
          ("Ruido Artefacto", senal_artefacto, snr_artefacto, "yellow")
```

Aquí se introduce tres tipos de ruido en una señal original (gaussiano, impulsivo y artefacto) mediante la función agregar_ruido(), obteniendo tanto la señal contaminada como su relación señal-ruido (SNR) en cada caso.

Luego, imprime los valores de SNR en decibeles (dB), permitiendo comparar cuánto afecta cada tipo de ruido a la señal. Finalmente, organiza las señales modificadas en una lista con su nombre, SNR y color asignado, lo que facilita su posterior graficación para visualizar el impacto de cada ruido en la señal original.

<img width="1432" height="712" alt="image" src="https://github.com/user-attachments/assets/c0afebf4-90cf-4bc1-977f-8de9f3aab9b3" />


<img width="1410" height="703" alt="image" src="https://github.com/user-attachments/assets/fc14a55c-01d3-4dff-85eb-fd42476d65a7" />


<img width="1408" height="702" alt="image" src="https://github.com/user-attachments/assets/c33e6ed7-ff46-4203-bff4-a8efb25a1983" />


## Curtosis

### Cálculo de curtosis
```
curtosis_scipy = kurtosis(señal, fisher=False)  # Curtosis normal (no excesiva)
curtosis_excesiva = curtosis_scipy - 3

```
La gráfica muestra la distribución de amplitudes de una señal EMG (electromiografía) junto con su curtosis. En el histograma (barras azules) se observa cómo se distribuyen los valores de amplitud de la señal, mientras que la línea punteada negra representa una distribución normal ajustada para comparar.

La curtosis es una medida estadística que describe la forma de la distribución, específicamente cuánto se concentran los datos en torno a la media y qué tan “pesadas” son las colas (valores extremos).

- Una curtosis alta (como en este caso, 17.90) indica que hay más valores extremos y un pico más pronunciado que en una distribución normal.

- La curtosis excesiva (14.90) es la curtosis ajustada restando 3 (que es el valor de la distribución normal). Un valor tan alto sugiere que la señal EMG presenta picos muy marcados y eventos inusuales con más frecuencia que lo que sería “normal”.
  
<img width="1389" height="720" alt="image" src="https://github.com/user-attachments/assets/c2e2c9b5-4bc7-4347-bcd7-7e499a6c0d83" />

## Adquisición de la señal fisiologíca 
## Captura de la señal 
En primera instancia se tomara de referencia el musculo sóleo y musculo tibial anterior, estos pertenecientes a la parte anterior-inferior de la pierna, con el fin de capturar la señal EMG proveniente de dicho musculo, teniendo en cuenta que se procesara mediante un DAQ para lograr observar la señal a una frecuencia adecuada para observar correctamente la representación de la actividad muscular, la señal adquirida fue exportada en formato .txt para proceder a analizar. 

Para visualizar la señal adquirida se desarrolla una interfaz en Python para la lectura, vicualización y análisis de dicha señal, como se presenta a continuación: 

Se debe realizar la respectiva configuración de la DAQ, que es la que procesara la señal

```
CHANNEL = "Dev5/ai0"           # Cambiar según el canal EMG conectado
SAMPLE_RATE = 1000             # Frecuencia de muestreo en Hz
SAMPLES_PER_READ = 100         # Cuántas muestras por lectura
DURATION = 10                  # Duración máxima en segundos

```
A partir de ello se implementa una interfaz que permitira iniciar y obtener la adquisión de la señal EMG, con el fin de visualizarla en tiempo real, así mismo se guardaran los datos en un archivo CSV para lograr realizarle un analisis estadistico. 

## Clase para adquirir la señal EMG

```
class EMGTask(Task):
    def _init_(self):
        Task._init_(self)
        self.CreateAIVoltageChan(CHANNEL, "", daqc.DAQmx_Val_Cfg_Default,
                                 -5.0, 5.0, daqc.DAQmx_Val_Volts, None)
        self.CfgSampClkTiming("", SAMPLE_RATE, daqc.DAQmx_Val_Rising,
                              daqc.DAQmx_Val_ContSamps, SAMPLES_PER_READ)

    def read_data(self):
        data = np.zeros((SAMPLES_PER_READ,), dtype=np.float64)
        read = daqf.int32()
        self.ReadAnalogF64(SAMPLES_PER_READ, 10.0, daqc.DAQmx_Val_GroupByChannel,
                           data, SAMPLES_PER_READ, daqf.byref(read), None)
        return data


```
## Función para iniciar la captura 

```
def start_acquisition():
    global running, data_storage, task
    data_storage = []
    running = True
    task = EMGTask()
    task.StartTask()
    threading.Thread(target=acquire_data).start()



```
## Adquirir datos en segundo plano 
```
def acquire_data():
    global running, data_storage
    start_time = time.time()
    while running and (time.time() - start_time) < DURATION:
        data = task.read_data()
        timestamp = time.time()
        for value in data:
            data_storage.append((timestamp, value))
        update_plot(data)
    task.StopTask()
    save_data()

```
## Detener la adquisición 
```
def acquire_data():
    global running, data_storage
    start_time = time.time()
    while running and (time.time() - start_time) < DURATION:
        data = task.read_data()
        timestamp = time.time()
        for value in data:
            data_storage.append((timestamp, value))
        update_plot(data)
    task.StopTask()
    save_data()

```
## Visualización de imagen capturada de la señal EMG 
Para poder visualizar la captura de la señal EMG se debe adicionar una parte al codigo, teniendo en cuenta que al procesar la señal es necesario importar los datos almacenados en un archivo txt. con el fin de que se analizara a información de manera adecuada y garantizar que el formato del archivo no generara errores. 

```
import numpy as np
import csv

ruta = r"C:\Users\majo1\OneDrive\Escritorio\señales\lab señales\lab 1\datos_ecg.txt"

# Detectar delimitador automáticamente
with open(ruta, 'r') as f:
    # Lee la primera línea (encabezado)
    linea = f.readline()
    dialect = csv.Sniffer().sniff(linea)
    delimitador = dialect.delimiter

print(f"Delimitador detectado: '{delimitador}'")

# Cargar datos ignorando el encabezado (primera fila)
data = np.genfromtxt(ruta, delimiter=delimitador, skip_header=1)

# Separar columnas
tiempo = data[:, 0]   # primera columna
señal = data[:, 1]    # segunda columna

print("Ejemplo de datos cargados:")
print("Tiempo:", tiempo[:10])
print("Señal:", señal[:10])

```
Con este codigo se imprime los primeros diez valores de cada pico con el fin de verificar que los datos hallan sido cargados correctamente, para el procesamiento y el analisis de la señal.

## Grafica de la señal muscular
<img width="1280" height="632" alt="image" src="https://github.com/user-attachments/assets/736fc70c-f113-4cac-a705-d4fb92f52e4b" />

A partir de la señal capturada del musculo, se realiza un analisis probabilistico de la siguiente manera:

## La media (promedio de la señal)
Esta nos muestra el valor promedio de las muestras en el tiempo.
Indica el nivel de referencia en el cual oscila la señal.
En el caso de las señales EMG la media puede encontrarse desplazada debido a offsets eléctricos del sistema de adquisición o a la actividad basal del musculo.

## La desviación estandar
Esta nos indica que tanta variación tienen los valores respecto a la media.
En este caso la desviación estandar arrojo que la señal se encuentra muy cerca de la media y ahi muy poca variación.
Nos muestra a intensidad de contracción muscular.

## El coeficiente de variación 
Nos indica la medida relativa de dispersión y la variabilidad de la señal con su media.

Se imprimen, organizan y presentan los datos legibles, permitiendo analizar y verificar los métodos utilizados para hallar las medidas estadisticas 

<img width="412" height="213" alt="image" src="https://github.com/user-attachments/assets/cce92fbe-a591-4f11-9b49-d93477523aa2" />

## HISTOGRAMA SEÑAL EMG MUSCULAR 

La imagen muestra un **histograma de una señal EMG**. Las barras azules representan la distribución de amplitudes, concentradas principalmente alrededor de **1.2**, mientras que la línea negra es una **curva normal** usada como referencia. Se observa que la mayoría de los valores se agrupan en un rango estrecho, con algunos pocos datos atípicos hacia los extremos. Esto indica que la señal tiene una distribución bastante concentrada y cercana a la normalidad, aunque con un pico marcado en el valor central.

<img width="1280" height="617" alt="image" src="https://github.com/user-attachments/assets/5b1bb158-ddd6-44a7-af81-21c132e47734" />

## Ajuste de una distribución normal (PDF)
La gráfica muestra la **Función de Probabilidad Acumulada (CDF)** de la señal EMG. Indica que la mayoría de los valores de amplitud se concentran entre **1.2 y 1.3**, donde la probabilidad sube bruscamente de casi 0 a 1. Antes de ese rango casi no hay datos y después prácticamente ya están todos acumulados. En resumen, la señal está fuertemente concentrada en ese intervalo de amplitudes.

<img width="1214" height="723" alt="image" src="https://github.com/user-attachments/assets/8529d19e-d2ff-48e1-9292-a1d02cc4f7a0" />

## Graficas señales con ruido en gráficos separados junto con la señal original

### Señal con ruido gaussiano

La gráfica muestra una **señal EMG comparada con la misma señal al agregarle ruido gaussiano**. La línea negra corresponde a la señal original y la roja, con un área sombreada, a la señal ruidosa con una relación señal/ruido (SNR) de 28 dB. Se observan picos de actividad muscular al inicio (0–3 s), mientras que después de los 5 s la señal se estabiliza alrededor de **1.3 mV**, aunque con pequeñas fluctuaciones generadas por el ruido, lo que ilustra cómo éste afecta la amplitud real sin ocultar la forma general de la señal.

<img width="1280" height="636" alt="image" src="https://github.com/user-attachments/assets/0cebe6b8-9ba0-4ff8-a8cd-aa293c0cbb90" />

### Señal con ruido impulso

La gráfica muestra una **señal EMG con ruido de tipo impulso**. La línea negra corresponde a la señal original, mientras que la verde representa la señal afectada por el ruido, con una relación señal/ruido (SNR) de **14.88 dB**. Se observan los picos iniciales de la señal muscular, pero luego aparecen múltiples impulsos que generan oscilaciones rápidas y bruscas alrededor de la señal real. En resumen, aunque la forma general de la EMG aún se distingue, el ruido impulso introduce interferencias notorias que distorsionan la lectura a lo largo del tiempo.

<img width="1280" height="635" alt="image" src="https://github.com/user-attachments/assets/ce4d675d-1508-4b8d-be54-7604106c10ec" />

### señal con ruido artefacto

La gráfica muestra una **señal EMG afectada por ruido de artefacto**. La línea negra corresponde a la señal original, mientras que la amarilla representa la señal con el ruido añadido, con una relación señal/ruido (SNR) de **31.20 dB**, lo que indica que la calidad de la señal sigue siendo bastante buena. Se observan los picos iniciales de actividad muscular y, aunque el ruido introduce pequeñas variaciones en torno a los 1.3 mV, la forma general de la señal se conserva casi intacta. En resumen, el ruido por artefactos está presente pero tiene un impacto bajo en la distorsión de la EMG.


<img width="1280" height="640" alt="image" src="https://github.com/user-attachments/assets/fbba72e9-5e58-4b08-9c97-3936ae22445e" />

## SNR 

La imagen muestra los resultados de la **Relación Señal-Ruido (SNR)** para distintos tipos de ruido aplicados a la señal EMG:

* **Ruido gaussiano:** SNR de **28.14 dB**, lo que indica que la señal mantiene buena calidad a pesar del ruido.
* **Ruido impulso:** SNR de **14.88 dB**, representando la condición más crítica, ya que este tipo de ruido degrada notablemente la señal.
* **Ruido artefacto:** SNR de **31.20 dB**, el más alto de los tres, lo que significa que el ruido apenas afecta la forma original de la señal.
  
La señal EMG es más resistente al ruido de artefacto y al gaussiano, pero se ve considerablemente afectada por el ruido de impulso.

<img width="456" height="92" alt="image" src="https://github.com/user-attachments/assets/00553270-d790-4d2c-a7eb-1d16c35b38ef" />

## curtosis

La gráfica muestra la **distribución de amplitudes de una señal EMG con análisis de curtosis**. El histograma (barras azules) concentra la mayoría de los valores alrededor de **1.3 mV**, mientras que la línea negra representa una distribución normal ajustada como referencia. El valor de **curtosis excesiva (20.21)** indica que la señal tiene una distribución muy **picuda y con colas pesadas**, es decir, los datos se concentran fuertemente en torno a un valor central pero con la presencia de valores extremos. En resumen, la señal EMG presenta una alta curtosis, lo que refleja gran concentración en un rango estrecho de amplitudes y la existencia de picos atípicos.

<img width="1280" height="641" alt="image" src="https://github.com/user-attachments/assets/3ed3ef1d-a97d-4522-95a5-61bf6b31caff" />

## Análisis de resultados.

### Señales originales 

<img width="1404" height="702" alt="image" src="https://github.com/user-attachments/assets/8e03eaa5-aafc-47b4-aeea-cb064d12e0fb" />
<img width="1280" height="632" alt="image" src="https://github.com/user-attachments/assets/51698c38-a229-47c6-8b2c-dae6b11eea9f" />

* La primera señal refleja un registro con baja intensidad y ruido, posiblemente sin contracción muscular voluntaria significativa.

* La segunda señal muestra una contracción muscular clara, con picos iniciales fuertes y luego un mantenimiento estable, lo cual indica que el músculo fue activado de manera sostenida.

### Calculos

<img width="426" height="205" alt="image" src="https://github.com/user-attachments/assets/4161fdb3-05e8-457a-a886-5fed7487505f" />
<img width="412" height="213" alt="image" src="https://github.com/user-attachments/assets/dc0092d2-8f53-4b23-847d-077f96ba5124" />

* La primera señal muscular se encuentra en un rango de los 8 s, observando que tiene variaciones con altas frecuencias con amplitud pequeña (-0.2 y 0.5 mV).
* Tiene un aspecto de señal en crudo (sin procesar demasiado), con bastante ruido y oscilaciones rapidas.
* Despues de los 6s se observan picos con alta actividad, lo que puede indicar alta concentración muscular. 
* La primera señal representa más ruido que actividad muscular real, con un nivel de variabilidad tan alto que los datos no son confiables para análisis clínico o biomecánico.

*La segunda señal muscular se encuentra en un rango de los 30s, teniendo en cuenta que al incio se evidencian unos picos altos en 2 mV referenres a las contracciones fuertes.
*Entre 1.2 - 1.3 mV se comienza a estabilizar la señal y a volver más continua con pocas variaciones bruscas.
* La segunda señal refleja una contracción muscular fisiológica clara y estable, con un nivel de variación aceptable que sí permite hacer estudios de fuerza, fatiga o control de exoesqueletos.

Respecto a la amplitud la segunda señal contiene una anplitud muy baja <0.5mv, esto se debe alguna atenuación, retificación o contracción debil y la primera señal tiene una ampliación mucho mayor >2mV en los picos, esto indica contracciones mas fuertes. 

* En términos prácticos, los cálculos estadísticos muestran por qué la segunda señal es mucho más útil: tiene una media significativa y un coeficiente de variación bajo, lo que respalda su estabilidad.

### Histograma.

<img width="1438" height="717" alt="image" src="https://github.com/user-attachments/assets/ac2a858e-54d6-4c04-8e6e-2c8fff266e60" />
<img width="1280" height="617" alt="image" src="https://github.com/user-attachments/assets/8e842cd3-b16e-4734-a2e0-29c4fd2ceb8d" />

* El primer histograma representa una señal con comportamiento caótico y amplitud muy baja, sin valor fisiológico significativo.

* El segundo histograma refleja un patrón real de actividad muscular sostenida, donde la mayoría de los datos se agrupan alrededor de un valor medio estable.

* En términos de análisis EMG, el histograma 2 permite estudiar fuerza, fatiga o control muscular, mientras que el histograma 1 es más representativo de ruido y microfluctuaciones.

### Función de probabilidad acumulativa.

<img width="1191" height="708" alt="image" src="https://github.com/user-attachments/assets/7164c331-27ff-4460-b565-0eb5488dc531" />
<img width="1214" height="723" alt="image" src="https://github.com/user-attachments/assets/2a0ba9b7-919a-4e2e-927e-09f8a5d4d13e" />

* La primera CDF muestra que casi toda la probabilidad acumulada está en torno a valores muy pequeños → ruido sin relevancia fisiológica.

* La segunda CDF concentra su probabilidad acumulada en valores mucho más altos (≈1.2 mV), lo que corresponde a una contracción muscular clara y estable.

* En conjunto con los histogramas y los estadísticos, queda confirmado que:

* Señal 1 = actividad basal/ruido

* Señal 2 = contracción real, estable y útil para análisis clínico o biomecánico.

  
### Señal-ruido (SNR)

🔴 1. Ruido Gaussiano

* Primera señal (débil, SNR = -1.02 dB):

El ruido gaussiano es tan fuerte que casi se superpone con la señal.

La señal original (picos pequeños, <0.5 mV) se ve prácticamente oculta.

El SNR negativo significa que el ruido domina sobre la señal.

* Segunda señal (fuerte, SNR = 28.14 dB):

Aunque hay ruido, la señal muscular (picos de 2 mV y contracción sostenida en 1.2 mV) se mantiene clara.

El SNR alto permite diferenciar muy bien la señal de fondo del ruido.

✅ Comparación:
El ruido gaussiano afecta de manera crítica a señales de baja amplitud, donde se confunde con la actividad muscular. En cambio, en señales de alta amplitud es mucho menos invasivo y se puede filtrar, ya que la contracción sigue siendo visible.
Se procesó una señal EMG , obteniendo sus estadísticos descriptivos como la media, desviación estándar y coeficiente de variación, los cuales permitieron describir su comportamiento. El histograma reveló que la señal sigue una distribución normal, mientras que la función de probabilidad mostró cómo se distribuyen los valores de la señal y permitió analizar la probabilidad de encontrar valores en rangos específicos.
Posteriormente, se añadió ruido gaussiano, de impulso y tipo artefacto, evaluando su alteración mediante el cálculo del SNR.



🟢 2. Ruido de Impulso

* Primera señal (débil, SNR = -2.78 dB):

Los picos de ruido de impulso superan la señal original.

La actividad muscular queda totalmente oculta entre los impulsos aleatorios.

El SNR negativo confirma que no es posible diferenciar señal de ruido.

* Segunda señal (fuerte, SNR = 14.88 dB):

Se distingue la contracción sostenida alrededor de 1.2 mV, pero con picos abruptos falsos (saltos >2 mV).

Aunque el SNR es positivo, el ruido de impulso sigue siendo molesto porque se confunde con picos musculares reales.

✅ Comparación:
El ruido de impulso es el más invasivo de todos: en señales débiles destruye la información, y en señales fuertes genera falsos positivos que complican el análisis.



🟡 3. Ruido de Artefacto

* Primera señal (débil, SNR = 1.96 dB):

Los artefactos producen una oscilación adicional sobre la señal pequeña.

Aunque hay distorsión, la forma de la señal original se intuye en segundo plano.

* Segunda señal (fuerte, SNR = 31.20 dB):

La señal muscular permanece clara y estable.

Los artefactos añaden pequeñas variaciones, pero no destruyen la información útil.

Es el caso con el SNR más alto de todos, lo que lo hace más manejable.

✅ Comparación:
El ruido de artefacto es el menos perjudicial, sobre todo en señales fuertes. Incluso con señales débiles, no destruye totalmente la forma de la señal, lo que lo hace más fácil de corregir con filtros.


### curtosis 


🔹 1. Señal EMG débil (primera imagen)

Curtosis: 17.90

Curtosis excesiva: 14.90

Interpretación:

La distribución de amplitudes está fuertemente concentrada alrededor de 0 mV, con picos muy marcados.

Una curtosis tan alta indica que hay una gran aglomeración de valores cerca de la media, pero con colas largas (valores extremos ocasionales).

Esto concuerda con una señal dominada por ruido, donde la mayor parte del tiempo no hay actividad muscular significativa, pero aparecen picos pequeños que generan colas largas en la distribución.


🔹 2. Señal EMG fuerte (segunda imagen)

Curtosis: 23.21

Curtosis excesiva: 20.21

Interpretación:

La distribución está centrada alrededor de 1.2–1.3 mV, con valores muy concentrados en torno a esa amplitud.

La curtosis es aún más alta que en la señal débil, lo que significa que existe una gran concentración de valores en torno al nivel de contracción sostenida, con algunos valores atípicos (picos iniciales de activación).

Aquí, la alta curtosis refleja una señal fisiológica muy estable pero con fuerte concentración en un rango estrecho de amplitudes.

## Conclusión. 

Al analizar las diferentes señales EMG, se ve claramente la diferencia entre una señal dominada por ruido y una que realmente refleja la contracción del músculo. La primera, de baja amplitud, mostró valores cercanos a cero, mucha variabilidad y poca estabilidad, lo que indica que no aporta información fisiológica relevante. En cambio, la señal de mayor amplitud permitió identificar contracciones sostenidas y organizadas, con un valor medio estable y una variabilidad mucho más controlada, lo que la hace útil para estudiar la actividad muscular de manera confiable.

Además, al incorporar distintos tipos de ruido, se pudo observar cómo cada uno afecta la interpretación de la señal: el ruido gaussiano tiende a difuminarla, el de impulso resulta el más agresivo al generar falsos picos, y el de artefacto es el menos perjudicial, sobre todo cuando la señal es fuerte. En conjunto, todo este análisis demuestra que la calidad de la señal y la relación señal/ruido son claves para poder trabajar con registros de EMG, ya sea con fines clínicos, de investigación o en el desarrollo de tecnologías como los exoesqueletos.

## Referencias

- Procesando señales electrofisiológicas usando Python - YouTube. (2020, 17 de diciembre) — PyDay Chile 2020 Procesando señales electrofisiológicas usando Python. Faviconyoutube.com
- What is Signal to Noise Ratio and How to calculate it? (2024, 17 julio). Advanced PCB Design Blog | Cadence. https://resources-pcb-cadence-com.translate.goog/blog/2020-what-is-signal-to-noise-ratio-and-how-to-calculate-it?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc
- Goyal B, Dogra A, Agrawal S, Sohi BS Problemas de ruido que prevalecen en varios tipos de imágenes médicas. Biomed Pharmacol J 2018;11(3). Disponible en: http://biomedpharmajournal.org/?p=22526
- Medidas de tendencia central y dispersión. (s. f.). Medwave. https://www.medwave.cl/series/MBE04/4934.html
