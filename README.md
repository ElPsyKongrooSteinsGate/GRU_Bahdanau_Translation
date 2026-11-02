# GRU + Bahdanau Attention Translation (C++20 / LibTorch)

A CPU-first, train-from-scratch encoder-decoder translation starter:
- One-layer, unidirectional GRU encoder and decoder
- Bahdanau additive attention
- Teacher forcing, cross-entropy, Adam, gradient clipping
- SQLite parallel-corpus storage
- JSON vocabulary/configuration files
- Greedy decoding

## Requirements
- MSVC x64 developer environment
- CMake 3.20+
- LibTorch **CPU** for MSVC, e.g. `C:/Library/libtorch`
- vcpkg with `sqlite3` and `nlohmann-json`

Install dependencies:
```cmd
vcpkg install sqlite3:x64-windows
vcpkg install nlohmann-json:x64-windows
```

SentencePiece and ICU are not required by this starter; its baseline tokenizer splits on whitespace.

For step-by-step Windows build, training, and inference instructions, see the [CLI tutorial](doc/tutorial/README.md).

## Configure and build
From the project directory, in an MSVC x64 Developer Command Prompt:
```cmd
cmake -S . -B build -A x64 ^
  -DCMAKE_PREFIX_PATH="C:/Library/libtorch" ^
  -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"

cmake --build build --config Release
```

If CMake cannot locate vcpkg, adjust the toolchain path. If Windows cannot locate LibTorch DLLs at runtime, add `C:\Library\libtorch\lib` to PATH:
```cmd
set PATH=C:\Library\libtorch\lib;%PATH%
```

## Prepare training data
Create a UTF-8 TSV file named `parallel.tsv`, one pair per line:
```text
hello	bonjour
good morning	bonjour
thank you	merci
how are you	comment allez-vous
```
Each line must contain source text, a TAB character, then target text. Do not put a header row in the file.

Initialize and import:
```cmd
build\Release\translate.exe init-db data.db
build\Release\translate.exe import-tsv data.db parallel.tsv
```

Alternatively, add individual pairs:
```cmd
build\Release\translate.exe add data.db "hello" "bonjour"
```

## Train
```cmd
build\Release\translate.exe train data.db models\demo 30
```
The final argument is the number of epochs. Default is 20.

## Translate
```cmd
build\Release\translate.exe infer models\demo "good morning"
```

## Important limitations
1. This is a teaching baseline, not a production translation system. Translation quality requires a sufficiently large, clean, aligned parallel corpus.
2. The tokenizer is whitespace-based and case-sensitive. It does not implement subword segmentation, punctuation normalization, or language-specific tokenization. SentencePiece is a good next step.
3. The trainer reports training loss only; add a held-out validation set, BLEU/chrF evaluation, checkpoint selection, and learning-rate scheduling for serious experiments.
4. The decoder uses greedy search. Beam search is a useful next improvement.
5. The model is unidirectional and uses fixed-size hidden vectors. Long sequences can be difficult for this architecture.
6. For real datasets, split training/validation/test data before building vocabularies, and avoid evaluating on training examples.
7. Train and inference use CPU. The code is designed around the CPU LibTorch distribution.
