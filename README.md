# LABORATORIO 2

# Introducción 
El procesamiento digital de señales contiene ciertas operaciones entre las mas relevantes se encuentran la convoluación y la correlación, las cuales proporcionan una base matematica y practica para la comprensión del comportamiento de los sistemas y la relación entre señales. La convolución es una operación que describe como un sistema lineal e invariante en el tiempo responde a una señal de entrada, convirtiendose así en un metodo esencial para el modelamiento y analisis de sistemas digitales. Por su parte la correlación, mide el grado de similitud entre dos señales en función del desfase temporal, siendo amplimente utilizada en aplicaciones de detección de patrones, reconocimiento de señales y analisis estadisticos. 
Así mismo, la transformada de Fourier se emplea como una herramienta complementaria que permite trasladar el analisis al dominio de la frecuencia, facilitando la caracterización espectral de las señales y la comprensión de sus componentes armonicas. De esta forma, el presente laboratorio desarrolla afianzar los conceptos matematicos involucrados en estas operaciones, sino también fomentar la capacidad de implementar y validar los procedimientos de manera manual y computacional, utilizando el software de Python.

El presente laboratorio tiene como propósito analizar y aplicar los conceptos de convolución, correlación y transformada de Fourier en el procesamiento digital de señales, con el fin de comprender su utilidad en la caracterización de sistemas lineales e invariantes en el tiempo, así como en la interpretación de señales biomédicas. En este sentido, se busca reconocer la convolución como herramienta fundamental para determinar la respuesta de un sistema ante diferentes entradas y aplicar la correlación como método para medir el grado de similitud entre dos señales discretas. Asimismo, se pretende implementar la transformada de Fourier para estudiar las señales en el dominio de la frecuencia, identificando sus componentes espectrales y evaluando parámetros estadisticos que permitan describir su comportamiento.

# MARCO TEORICO 
# Convolución de señales discretas 

La convolución es una operación matemática fundamental en el análisis de sistemas lineales e invariantes en el tiempo (LTI). Permite calcular la salida de un sistema cuando se conoce su respuesta el impulso h[n] y la señal de entrada x[n].

La definición discreta de la convolución 

<img width="424" height="93" alt="image" src="https://github.com/user-attachments/assets/223c6ebd-c2e6-4cb5-ba6a-5cee8ee7e51f" />
Cada muestra de la entrada multiplica la respuesta al impulso y la desplaza en el tiempo.




