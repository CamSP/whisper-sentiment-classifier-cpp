# Whisper.cpp

## Modified whisper.cpp implementation (original from https://github.com/ggerganov/whisper.cpp) by group 7 for HPC 2023-1

High-performance inference of [OpenAI's Whisper](https://github.com/openai/whisper) automatic speech recognition (ASR) model:

- Plain C/C++ implementation 
- AVX intrinsics support for x86 architectures
- VSX intrinsics support for POWER architectures
- Mixed F16 / F32 precision
- Low memory usage (Flash Attention)
- Zero memory allocations at runtime
- Runs on the CPU
- [C-style API](https://github.com/ggerganov/whisper.cpp/blob/master/whisper.h)

Supported platforms of this variation:

- [x] Mac OS (Intel and Arm)
- [x] Linux

The entire implementation of the model is contained in 2 source files:

- Tensor operations: [ggml.h](ggml.h) / [ggml.c](ggml.c)
- Transformer inference: [whisper.h](whisper.h) / [whisper.cpp](whisper.cpp)

## Implementation details

- The core tensor operations are implemented in C ([ggml.h](ggml.h) / [ggml.c](ggml.c))
- The transformer model and the high-level C-style API are implemented in C++ ([whisper.h](whisper.h) / [whisper.cpp](whisper.cpp))
- Sample usage is demonstrated in [main.cpp](examples/main)
- Sample real-time audio transcription from the microphone is demonstrated in [stream.cpp](examples/stream)

The tensor operators are optimized heavily for Apple silicon CPUs. Depending on the computation size, Arm Neon SIMD
instrisics or CBLAS Accelerate framework routines are used. The latter are especially effective for bigger sizes since
the Accelerate framework utilizes the special-purpose AMX coprocessor available in modern Apple products.

## Modified Quick start 

First, download one of the Whisper models converted in [ggml format](models). In this case we will be using:

```bash
bash ./models/download-ggml-model.sh medium
```
**Note:** Now it is not necesary to download the model in advance, the Medium model gets downloaded automatically if non-existant through the Makefile during the first execution.


Now build the [main](examples/main) example which is the most typical use case for prerecorded audio transcprition. This is done using the Makefile inside the whisper.cpp main directory:

```bash
# build the main example
make -f makeWhisper.mk
```
This might take a bit, but once it is finished you can simply run the [main](examples/main) exectubale alongside the needed or desired options in order to obtain transcribe the audio
```bash
# transcribe an audio file
./main -f samples/jfk.wav
```
In our case once again, we'll use the following command instead:
```bash
# Modified execution
./main -f samples/jfk.wav -m models/ggml-medium.bin -l en -t 10 -otxt
```

This extra flags will make sure we are using our downloaded medium model, we specify the language of the audio in order to save execution time, we make sure to use more threads to accelerate the processing and we save the output to a .txt file in the same path as the audio.

This is all needed for different reasons. We specify the model as he default one after compilation is the Tiny english one, and that does not suit our application. We neet to specify the language since even though Whisper is able to identify the language and adjust itself accordingly, this takes between 30 to 40 additional seconds to process. We increase the used threads from the default 4 whenever possible as this greatly reduces the execution time, from over 70 seconds with 4 threads to under 30 with 10 threads. And the output is saved since it needs to get passed along to our Neural network for sentiment analysis.

## More audio samples

There are more audio samples to test that are easily available only running the command:

```bash
make samples
```

This will download audio files from Wikipedia and convert them to 16-bit WAV format via `ffmpeg`, which is needed as the model is made to receive files with those specifications.

```bash
ffmpeg -loglevel -0 -y -i samples/{name} -ar 16000 -ac 1 -c:a pcm_s16le samples/{name.wav}
```

## Memory usage

This table summarizes the Disk and RAM usage of all models
| Model  | Disk   | Mem     |
| ---    | ---    | ---     |
| tiny   |  75 MB | ~125 MB |
| base   | 142 MB | ~210 MB |
| small  | 466 MB | ~600 MB |
| medium | 1.5 GB | ~1.7 GB |
| large  | 2.9 GB | ~3.3 GB |

## Limitations

- Inference only, can not be re-trained or modified
- No GPU support, therefore slower execution times, but greater compatibility.



## Real-time audio input example

This is a naive example of performing real-time inference on audio from your microphone.
The [stream](examples/stream) tool samples the audio every half a second and runs the transcription continously.
More info is available in [issue #10](https://github.com/ggerganov/whisper.cpp/issues/10).

```bash
make stream
./stream -m ./models/ggml-medium.bin -t 10 --step 500 --length 5000
```

## Confidence color-coding

The `--print-colors` flag will print out a color coded text on the terminal showing the confidence level on each word:

<img width="965" alt="image" src="https://user-images.githubusercontent.com/1991296/197356445-311c8643-9397-4e5e-b46e-0b4b4daa2530.png">

## Controlling the length of the generated text segments (experimental)

It is possible to change the length of each string transcribed using the flag `-ml N`, where N is the desired length in characters, where if the string exceeds the N number, it will split on the last complete word that fits. for example `-ml 16`:

```java
./main -m ./models/ggml-medium.bin -f ./samples/jfk.wav -ml 16

whisper_model_load: loading model from './/models/ggml-medium.bin'
...
system_info: n_threads = 4 / 10 | AVX2 = 0 | AVX512 = 0 | NEON = 1 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 |

main: processing './samples/jfk.wav' (176000 samples, 11.0 sec), 4 threads, 1 processors, lang = en, task = transcribe, timestamps = 1 ...

[00:00:00.000 --> 00:00:00.850]   And so my
[00:00:00.850 --> 00:00:01.590]   fellow
[00:00:01.590 --> 00:00:04.140]   Americans, ask
[00:00:04.140 --> 00:00:05.660]   not what your
[00:00:05.660 --> 00:00:06.840]   country can do
[00:00:06.840 --> 00:00:08.430]   for you, ask
[00:00:08.430 --> 00:00:09.440]   what you can do
[00:00:09.440 --> 00:00:10.020]   for your
[00:00:10.020 --> 00:00:11.000]   country.
```

### Word-level timestamp

The `-ml N` flag can also be used to obtain each word in a separate string, simply by setting the N parametre to `1`: 

```java
./main -m ./models/ggml-medium.bin -f ./samples/jfk.wav -ml 1

whisper_model_load: loading model from './models/ggml-medium.bin'
...
system_info: n_threads = 4 / 10 | AVX2 = 0 | AVX512 = 0 | NEON = 1 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 |

main: processing './samples/jfk.wav' (176000 samples, 11.0 sec), 4 threads, 1 processors, lang = en, task = transcribe, timestamps = 1 ...

[00:00:00.000 --> 00:00:00.320]  
[00:00:00.320 --> 00:00:00.370]   And
[00:00:00.370 --> 00:00:00.690]   so
[00:00:00.690 --> 00:00:00.850]   my
[00:00:00.850 --> 00:00:01.590]   fellow
[00:00:01.590 --> 00:00:02.850]   Americans
[00:00:02.850 --> 00:00:03.300]  ,
[00:00:03.300 --> 00:00:04.140]   ask
[00:00:04.140 --> 00:00:04.990]   not
[00:00:04.990 --> 00:00:05.410]   what
[00:00:05.410 --> 00:00:05.660]   your
[00:00:05.660 --> 00:00:06.260]   country
[00:00:06.260 --> 00:00:06.600]   can
[00:00:06.600 --> 00:00:06.840]   do
[00:00:06.840 --> 00:00:07.010]   for
[00:00:07.010 --> 00:00:08.170]   you
[00:00:08.170 --> 00:00:08.190]  ,
[00:00:08.190 --> 00:00:08.430]   ask
[00:00:08.430 --> 00:00:08.910]   what
[00:00:08.910 --> 00:00:09.040]   you
[00:00:09.040 --> 00:00:09.320]   can
[00:00:09.320 --> 00:00:09.440]   do
[00:00:09.440 --> 00:00:09.760]   for
[00:00:09.760 --> 00:00:10.020]   your
[00:00:10.020 --> 00:00:10.510]   country
[00:00:10.510 --> 00:00:11.000]  .
```

## ggml format

The original models are converted to a custom binary format. This allows to pack everything needed into a single file:

- model parameters
- mel filters
- vocabulary
- weights

You can download the converted models using the [models/download-ggml-model.sh](models/download-ggml-model.sh) script.


## Examples

There are various examples of using the library for different projects in the [examples](examples) folder.
Some of the examples are even ported to run in the browser using WebAssembly. Check them out!

| Example | Description |
| ---     | ---         |
| [main](examples/main) | Translating and transcribing audio using Whisper Cpp model |
| [stream](examples/stream) | Real-time transcription of raw microphone capture |
| [talk](examples/talk) | Talk with a GPT-2 bot |
