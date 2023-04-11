.PHONY: do 
do:
#Primero le pregunta al usuario si ya tiene ubicado el audio en la carpeta indicada con el nombre correcto.
#En caso de ser así, continua el programa copiando el audio a la ubicación necesaria y compilando Whisper 
#solo si es necesario. En caso contrario termina el programa y pide reintentar.
	@while [ -z "$$ENTRADA" ]; do \
	read -r -p "El audio deseado está en la carpeta 'Entrada' con el nombre 'Entrada.wav'? [y/n]: " ENTRADA; \
	done ; \
	[ $$ENTRADA = "Y" ] || [ $$ENTRADA = "y" ] || (echo "Ubica el audio e intentalo de nuevo"; exit 1;)
	@echo "Iniciando Whisper..."
	cp Entrada/Entrada.wav Whisper.cpp/samples/Input
	@if ! [ -e Whisper.cpp/main ]; then \
	echo "Compilando Whisper..."; \
	cd Whisper.cpp; \
	make -f makeWhisper.mk; \
	fi
#Una vez compilado y con el audio en su lugar, realiza la inferencia de texto y lleva la salida a la carpeta de la NN.
	@cd Whisper.cpp; \
	./main -f samples/Input/Entrada.wav -m models/ggml-medium.bin -l en -t 12 -otxt
	@cd ..
	@cp Whisper.cpp/samples/Input/Entrada.wav.txt NN/whisperOutput/output.txt
	@rm Whisper.cpp/samples/Input/Entrada.wav.txt
#Ejecuta la NN para la clasificación de sentimientos
	@cd NN; \
	chmod +x install.sh; \
	./install.sh; \
	chmod +x predict.sh; \
	./predict.sh


	

