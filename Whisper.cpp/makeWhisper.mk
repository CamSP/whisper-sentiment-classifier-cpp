# Makefile start

# Defining system parmateres such as OS and architecture in order to set the correct compilation flags
ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

CCV := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)

# Mac OS + Arm can report x86_64
ifeq ($(UNAME_S),Darwin)
	ifneq ($(UNAME_P),arm)
		SYSCTL_M := $(shell sysctl -n hw.optional.arm64)
		ifeq ($(SYSCTL_M),1)
			# UNAME_P := arm
			# UNAME_M := arm64
			warn := $(warning Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lead to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-1282546789)
		endif
	endif
endif

#
# Compile flags
# C : Binary compiler for ggml files
# CXX: g++ compiler for Cpp files

CFLAGS   = -I.              -O3 -DNDEBUG -std=c11   -fPIC
CXXFLAGS = -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -fsanitize=address -fsanitize=leak -fsanitize=undefined
LDFLAGS  =

ifneq ($(wildcard /usr/include/musl/*),)
	CFLAGS   += -D_POSIX_SOURCE -D_GNU_SOURCE
	CXXFLAGS += -D_POSIX_SOURCE -D_GNU_SOURCE
endif

# OS specific
ifeq ($(UNAME_S),Linux)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Darwin)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),FreeBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Haiku)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif

# Architecture specific
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686))
	ifeq ($(UNAME_S),Darwin)
		CFLAGS += -mf16c
		AVX1_M := $(shell sysctl machdep.cpu.features)
		ifneq (,$(findstring FMA,$(AVX1_M)))
			CFLAGS += -mfma
		endif
		ifneq (,$(findstring AVX1.0,$(AVX1_M)))
			CFLAGS += -mavx
		endif
		AVX2_M := $(shell sysctl machdep.cpu.leaf7_features)
		ifneq (,$(findstring AVX2,$(AVX2_M)))
			CFLAGS += -mavx2
		endif
	else ifeq ($(UNAME_S),Linux)
		AVX1_M := $(shell grep "avx " /proc/cpuinfo)
		ifneq (,$(findstring avx,$(AVX1_M)))
			CFLAGS += -mavx
		endif
		AVX2_M := $(shell grep "avx2 " /proc/cpuinfo)
		ifneq (,$(findstring avx2,$(AVX2_M)))
			CFLAGS += -mavx2
		endif
		FMA_M := $(shell grep "fma " /proc/cpuinfo)
		ifneq (,$(findstring fma,$(FMA_M)))
			CFLAGS += -mfma
		endif
		F16C_M := $(shell grep "f16c " /proc/cpuinfo)
		ifneq (,$(findstring f16c,$(F16C_M)))
			CFLAGS += -mf16c
		endif
		SSE3_M := $(shell grep "sse3 " /proc/cpuinfo)
		ifneq (,$(findstring sse3,$(SSE3_M)))
			CFLAGS += -msse3
		endif
	else ifeq ($(UNAME_S),Haiku)
		AVX1_M := $(shell sysinfo -cpu | grep "AVX ")
		ifneq (,$(findstring avx,$(AVX1_M)))
			CFLAGS += -mavx
		endif
		AVX2_M := $(shell sysinfo -cpu | grep "AVX2 ")
		ifneq (,$(findstring avx2,$(AVX2_M)))
			CFLAGS += -mavx2
		endif
		FMA_M := $(shell sysinfo -cpu | grep "FMA ")
		ifneq (,$(findstring fma,$(FMA_M)))
			CFLAGS += -mfma
		endif
		F16C_M := $(shell sysinfo -cpu | grep "F16C ")
		ifneq (,$(findstring f16c,$(F16C_M)))
			CFLAGS += -mf16c
		endif
	else
		CFLAGS += -mfma -mf16c -mavx -mavx2
	endif
endif
ifeq ($(UNAME_M),amd64)
	CFLAGS += -mavx -mavx2 -mfma -mf16c
endif
ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
	ifneq (,$(findstring POWER9,$(POWER9_M)))
		CFLAGS += -mpower9-vector
	endif
	# Require c++23's std::byteswap for big-endian support.
	ifeq ($(UNAME_M),ppc64)
		CXXFLAGS += -std=c++23 -DGGML_BIG_ENDIAN
	endif
endif
ifndef WHISPER_NO_ACCELERATE
	# Mac M1 - include Accelerate framework
	ifeq ($(UNAME_S),Darwin)
		CFLAGS  += -DGGML_USE_ACCELERATE
		LDFLAGS += -framework Accelerate
	endif
endif
ifdef WHISPER_OPENBLAS
	CFLAGS  += -DGGML_USE_OPENBLAS -I/usr/local/include/openblas
	LDFLAGS += -lopenblas
endif
ifdef WHISPER_GPROF
	CFLAGS   += -pg
	CXXFLAGS += -pg
endif
ifneq ($(filter aarch64%,$(UNAME_M)),)
	CFLAGS += -mcpu=native
	CXXFLAGS += -mcpu=native
endif

#
# Print build information
#

$(info I whisper.cpp build info: )
$(info I UNAME_S:  $(UNAME_S))
$(info I UNAME_P:  $(UNAME_P))
$(info I UNAME_M:  $(UNAME_M))
$(info I CFLAGS:   $(CFLAGS))
$(info I CXXFLAGS: $(CXXFLAGS))
$(info I LDFLAGS:  $(LDFLAGS))
$(info I CC:       $(CCV))
$(info I CXX:      $(CXXV))
$(info )

#When running "make" this is the default section run, can be main, stream or talk in our case.
default: main

#
# Build library
#

ggml.o: ggml.c ggml.h
	$(CC)  $(CFLAGS)   -c ggml.c -o ggml.o

whisper.o: whisper.cpp whisper.h
	$(CXX) $(CXXFLAGS) -c whisper.cpp -o whisper.o


libwhisper.a: ggml.o whisper.o
	$(AR) rcs libwhisper.a ggml.o whisper.o

libwhisper.so: ggml.o whisper.o
	$(CXX) $(CXXFLAGS) -shared -o libwhisper.so ggml.o whisper.o $(LDFLAGS)

clean:
	rm -f *.o main stream command talk bench libwhisper.a libwhisper.so

#
# Examples
#

CC_SDL=`sdl2-config --cflags --libs`

SRC_COMMON = examples/common.cpp
SRC_COMMON_SDL = examples/common-sdl.cpp

main: examples/main/main.cpp $(SRC_COMMON) ggml.o whisper.o
	@if ! [ -e models/ggml-medium.bin ]; then \
	cd models; \
	bash ./download-ggml-model.sh medium; \
	cd ..;\
	fi
	$(CXX) $(CXXFLAGS) examples/main/main.cpp $(SRC_COMMON) ggml.o whisper.o -o main $(LDFLAGS)
	@sudo apt -y install ffmpeg
	./main -h

stream: examples/stream/stream.cpp $(SRC_COMMON) $(SRC_COMMON_SDL) ggml.o whisper.o
	$(CXX) $(CXXFLAGS) examples/stream/stream.cpp $(SRC_COMMON) $(SRC_COMMON_SDL) ggml.o whisper.o -o stream $(CC_SDL) $(LDFLAGS)

talk: examples/talk/talk.cpp examples/talk/gpt-2.cpp $(SRC_COMMON) $(SRC_COMMON_SDL) ggml.o whisper.o
	$(CXX) $(CXXFLAGS) examples/talk/talk.cpp examples/talk/gpt-2.cpp $(SRC_COMMON) $(SRC_COMMON_SDL) ggml.o whisper.o -o talk $(CC_SDL) $(LDFLAGS)

#
# Audio samples
#

# download a few audio samples into folder "./samples", to run use "make samples"
.PHONY: samples
samples:
	@echo "Downloading samples..."
	@mkdir -p samples
	@wget --quiet --show-progress -O samples/gb0.ogg https://upload.wikimedia.org/wikipedia/commons/2/22/George_W._Bush%27s_weekly_radio_address_%28November_1%2C_2008%29.oga
	@wget --quiet --show-progress -O samples/gb1.ogg https://upload.wikimedia.org/wikipedia/commons/1/1f/George_W_Bush_Columbia_FINAL.ogg
	@wget --quiet --show-progress -O samples/hp0.ogg https://upload.wikimedia.org/wikipedia/en/d/d4/En.henryfphillips.ogg
	@wget --quiet --show-progress -O samples/mm1.wav https://cdn.openai.com/whisper/draft-20220913a/micro-machines.wav
	@echo "Converting to 16-bit WAV ..."
	@sudo apt -y install ffmpeg
	@ffmpeg -loglevel -0 -y -i samples/gb0.ogg -ar 16000 -ac 1 -c:a pcm_s16le samples/gb0.wav
	@ffmpeg -loglevel -0 -y -i samples/gb1.ogg -ar 16000 -ac 1 -c:a pcm_s16le samples/gb1.wav
	@ffmpeg -loglevel -0 -y -i samples/hp0.ogg -ar 16000 -ac 1 -c:a pcm_s16le samples/hp0.wav
	@ffmpeg -loglevel -0 -y -i samples/mm1.wav -ar 16000 -ac 1 -c:a pcm_s16le samples/mm0.wav
	@rm samples/mm1.wav
	@rm samples/gb0.ogg
	@rm samples/gb1.ogg
	@rm samples/hp0.ogg