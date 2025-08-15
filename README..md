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
En primera instancia se tomara de referencia el musculo sóleo y musculo tibial anterior, estos pertenecientes a la parte anterior-inferior de la pierna, con el fin de capturar la señal EMG proveniente de dicho musculo, donde se utilizara una DAQ que es el sistema de adquisición de datos permitiendo convertir la señal analoga generada por el musculo en una señal digital que se procesara en una interfaz en python. 

Para la realizacióin de la interfaz, se debe tener la configuración de la DAQ
...
class EMGTask(Task):
    def _init_(self):
        Task._init_(self)
        self.CreateAIVoltageChan(CHANNEL, "", daqc.DAQmx_Val_Cfg_Default,
                                 -5.0, 5.0, daqc.DAQmx_Val_Volts, None)
        self.CfgSampClkTiming("", SAMPLE_RATE, daqc.DAQmx_Val_Rising,
                              daqc.DAQmx_Val_ContSamps, SAMPLES_PER_READ)


...


## Análisis de resultados.

Se procesó una señal EMG , obteniendo sus estadísticos descriptivos como la media, desviación estándar y coeficiente de variación, los cuales permitieron describir su comportamiento. El histograma reveló que la señal sigue una distribución normal, mientras que la función de probabilidad mostró cómo se distribuyen los valores de la señal y permitió analizar la probabilidad de encontrar valores en rangos específicos.
Posteriormente, se añadió ruido gaussiano, de impulso y tipo artefacto, evaluando su alteración mediante el cálculo del SNR.


## Conclusión. 
El análisis estadístico permitió caracterizar la señal EMG, evidenciando su variabilidad y comportamiento mediante la media, desviación estándar y coeficiente de variación. La distribución aproximadamente normal.
La introducción de distintos tipos de ruido permitió evaluar su impacto en la calidad de la señal. Esto destaca la importancia del cálculo del SNR como herramienta para medir la degradación de la señal y la necesidad de aplicar filtros adecuados según el tipo de ruido presente

## Referencias:
- Procesando señales electrofisiológicas usando Python - YouTube. (2020, 17 de diciembre) — PyDay Chile 2020 Procesando señales electrofisiológicas usando Python. Faviconyoutube.com
- What is Signal to Noise Ratio and How to calculate it? (2024, 17 julio). Advanced PCB Design Blog | Cadence. https://resources-pcb-cadence-com.translate.goog/blog/2020-what-is-signal-to-noise-ratio-and-how-to-calculate-it?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc
- Goyal B, Dogra A, Agrawal S, Sohi BS Problemas de ruido que prevalecen en varios tipos de imágenes médicas. Biomed Pharmacol J 2018;11(3). Disponible en: http://biomedpharmajournal.org/?p=22526
- Medidas de tendencia central y dispersión. (s. f.). Medwave. https://www.medwave.cl/series/MBE04/4934.html
