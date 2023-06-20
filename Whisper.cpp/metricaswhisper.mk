.PHONY: do 
do:
#Debido a la sencillez de la ejecución, la relativa baja cantidad de ejeuciones deseadas y la cantidad de texto e información que arroja cada ejecución, se
#decidió optar por una recolección de datos manual, pues tomaba más tiempo realizar un  script de limpieza de datos tras recolectarlos todos que realizar la 
#transcripción manual para este caso específico

#Se inicia compilando el modelo y ejecutable principal de whisper si no existen
	@echo "Iniciando Whisper..."
	cp ../Entrada/Entrada.wav samples/Input
	@if ! [ -e main ]; then \
	echo "Compilando Whisper..."; \
	make -f makeWhisper.mk; \
	fi
#Una vez compilado y con el audio de prueba en su lugar, realiza la inferencia de texto aumentando el número de hilos asignados en cada iteración
	for i in 1 2 3 4 5 6 7 8 9 10 11 12; do ./main -f samples/Input/Entrada.wav -m models/ggml-medium.bin -l en -t $$i; done

#Debido a que el equipo donde se ejecutó solo posee 6 núcelos  12 hilos, al ejecutar la inferencia con 13 hilos el proceso avanzaba, culminaba pero al finalizar 
#en lugar de arrojar los datos de la ejecución, se reiniciaba la misma. Para ello se realizó una aproximación midiendo el tiempo total de todas las ejecuciones 
#mediante la herramienta de Htop hasta el momento en que se reiniciaban, tomando ese tiempo total y dividiendolo entre el número de hilos de esa ejecución 
#obtenemos un aproximado del tiempo real de ejecución, el cual indistintamente del método de captura es un tiempo cada vez mayor al de 12 hilos.
	./main -f samples/Input/Entrada.wav -m models/ggml-medium.bin -l en -t 13
	./main -f samples/Input/Entrada.wav -m models/ggml-medium.bin -l en -t 14
	./main -f samples/Input/Entrada.wav -m models/ggml-medium.bin -l en -t 15
	./main -f samples/Input/Entrada.wav -m models/ggml-medium.bin -l en -t 16