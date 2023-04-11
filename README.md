# whisper-sentiment-classifier-cpp
Proyecto desarrollado para el curso de Introducción a la Computación Científica de Alto Rendimiento 2023-1.

Grupo 7 conformado por:
- Camilo Esteban Zambrano
<!--- - Mario Oswaldo Peña --->
- Juan Felipe Saavedra
- Camilo Andres Valencia

## Funcionamiento

Ubicar el audio en inglés que se desea analizar en la carpeta Entrada, con el nombre Entrada.wav. Importante que esté en el formato .Wav de 16 bits, para lo cual se puede hacer uso de la herramienta `ffmpeg`, la cual es instalada durante la compilación de Whisper o si se desea se puede navegar al directorio `Whisper.cpp` y ejecutar el siguiente comando para descargar varias muestras de audio e instalar la herramienta mencionada sin compilar el modelo.
``` bash
make samples
```
También puede realizar la conversión manualmente tras instalar ffmpeg:

```bash
sudo apt install ffmpeg
ffmpeg -loglevel -0 -y -i {path}/{name.cualquierExtension} -ar 16000 -ac 1 -c:a pcm_s16le {path}/{name.wav}
```
Con el audio ubicado, navegar a la carpeta raíz del repositorio y simplemente correr el comando 
```bash
sudo make
```
Con este se ejecuta el archivo MakeFile el cual se encarga de inciar todo el proceso de intereferencia y predicción. Se ejecuta con permisos elevados dado que se realiza la instalación de cualquier programa requerido que no se encuentre ya presente en cada etapa del proceso.
