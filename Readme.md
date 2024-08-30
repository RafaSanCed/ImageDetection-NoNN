# Detección de Objetos en Imágenes Aéreas Utilizando Métodos Clásicos

Este repositorio contiene el código y los recursos necesarios para detectar aviones en imágenes aéreas utilizando 
métodos clásicos de procesamiento de imágenes. El objetivo de este proyecto es explorar alternativas a las redes 
neuronales para la detección de objetos, demostrando que técnicas más simples pueden ofrecer resultados efectivos.

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