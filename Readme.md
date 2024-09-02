# Detección de Objetos en Imágenes Aéreas Utilizando Métodos Clásicos

Este proyecto tiene como objetivo explorar alternativas a las redes neuronales para la detección de objetos en imágenes aéreas, específicamente aviones. Utilizando técnicas clásicas de procesamiento de imágenes, como la detección de bordes y el análisis de cambios de color, combinadas con optimización de parámetros mediante Optuna, se busca demostrar que es posible lograr resultados efectivos sin recurrir a modelos de deep learning.
## Descripción del Proyecto

La detección de objetos en imágenes aéreas presenta desafíos únicos debido a las variaciones en el ángulo de captura, condiciones de iluminación y la naturaleza diversa de los fondos. Este proyecto adopta un enfoque basado en la combinación de filtros clásicos y métodos de procesamiento de imágenes, evitando la complejidad de las redes neuronales profundas.

### Metodología:

1. **Preprocesamiento de la imagen:**
   - Conversión de las imágenes a escala de grises para simplificar el procesamiento.
   - Aplicación de un filtro de desenfoque gaussiano para reducir el ruido y mejorar la detección de bordes.

2. **Detección de bordes:**
   - Uso del algoritmo de Canny para resaltar los contornos de los objetos en la imagen, un método robusto y probado en diversas aplicaciones.

3. **Detección de cambios de color:**
   - Segmentación de la imagen en sus canales de color (rojo, verde, azul) para detectar cambios bruscos de color a lo largo de los bordes identificados. Esta técnica permite resaltar áreas donde es más probable que se encuentren los aviones, aprovechando las diferencias cromáticas típicas.

4. **Combinación lineal de filtros:**
   - Integración de las máscaras obtenidas de la detección de bordes y los cambios de color mediante una combinación lineal ponderada, resaltando las regiones de interés.

5. **Optimización con Optuna:**
   - Ajuste de los parámetros del modelo utilizando Optuna, un framework para la optimización de hiperparámetros, para mejorar la precisión del modelo.


## Estructura del Repositorio

- **Aerial Airport.v1-v1.yolov9/**: Carpeta que contiene las imágenes y etiquetas en formato YOLOv9 utilizadas en el proyecto.
- **entrenofinal.py**: Código que implementa el proceso de optimización de hiperparámetros utilizando Optuna, junto con la detección y evaluación en un conjunto de imágenes.
- **pruebasaviones.py**: Código que realiza la detección de aviones en una sola imagen, utilizando la metodología explicada.
- **requirements.txt**: Archivo que contiene las dependencias necesarias para ejecutar los scripts en este repositorio.

## Requisitos

Para ejecutar el código en este repositorio, es necesario instalar las siguientes dependencias. Puedes hacerlo ejecutando:

```bash
pip install -r requirements.txt
```

## Uso

### Detección en Conjunto de Imágenes

 El script `entrenofinal.py` se utiliza para realizar la detección de aviones en un conjunto de imágenes y optimizar los parámetros del modelo utilizando Optuna.

 ```bash
 python entrenofinal.py
 ```

 ### Detección en una Imagen

 El script `pruebasaviones.py` permite realizar la detección de aviones en una sola imagen, mostrando el proceso paso a paso.

 ```bash
 python pruebasaviones.py
 ```

 ### Estructura de las Imágenes y Etiquetas

 Las imágenes y etiquetas utilizadas en este proyecto se encuentran en la carpeta `Aerial Airport.v1-v1.yolov9`, 
 organizadas en subcarpetas `train`, `valid`, y `test`. Asegúrate de que las rutas en los scripts apunten 
 correctamente a estas carpetas para que los scripts funcionen sin problemas.

 ## Resultados

 Este proyecto demuestra que, utilizando una combinación de técnicas clásicas de procesamiento de imágenes, es 
 posible detectar objetos en imágenes aéreas de manera efectiva. A través de la optimización de parámetros y la 
 combinación de filtros, logramos resultados comparables a los obtenidos con redes neuronales, pero con una fracción 
 del costo computacional.

 ## Contribuciones

 Cualquier contribución para mejorar este proyecto es bienvenida. Si encuentras un error o tienes sugerencias, no 
 dudes en crear un issue o enviar un pull request.

 ## Licencia

 Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
