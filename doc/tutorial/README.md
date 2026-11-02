# Build, train, and run the translation CLI

This tutorial covers rebuilding and running the `translate.exe` command-line program on Windows with Visual Studio Build Tools and LibTorch.

## 1. Open the Visual Studio developer command prompt

In a Windows `cmd.exe` terminal, start the x64 developer environment with:

```cmd
%comspec% /k "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
```

If the prompt opens in a different folder, switch to the project root (adjust the path if you moved the repository):

```cmd
cd /d "C:\Users\Acer\Downloads\libtorch\GRU_Bahdanau_Translation"
```

## 2. Configure and build

Configure once, or repeat this if the LibTorch/vcpkg location or CMake configuration changes:

```cmd
cmake -S . -B build -A x64 -DCMAKE_PREFIX_PATH="C:/Library/libtorch" -DCMAKE_TOOLCHAIN_FILE="C:/Library/vcpkg-2026.10.16/scripts/buildsystems/vcpkg.cmake"
```

Build or rebuild the CLI whenever the source code changes:

```cmd
cmake --build build --config Release
```

The executable is created at `build\Release\translate.exe`.

## 3. Make the LibTorch DLLs available

In the same command prompt, add LibTorch's DLL folder to `PATH` before running the executable:

```cmd
set PATH=C:\Library\libtorch\lib;%PATH%
```

Repeat this `set PATH` command in each new command prompt.

## 4. Prepare the training data (first run)

Create a UTF-8 TSV file named `parallel.tsv` in the project root. Each non-empty row must contain source text, one TAB character, then target text. Do not add a header row. For example:

```text
hello<TAB>bonjour
good morning<TAB>bonjour
thank you<TAB>merci
```

Here `<TAB>` represents a literal tab character; do not type the angle-bracket text.

Create the database and import the TSV once:

```cmd
build\Release\translate.exe init-db data.db
build\Release\translate.exe import-tsv data.db parallel.tsv
```

`import-tsv` appends rows. Do not rerun it on the same database unless you intend to import duplicate training pairs. To start over with a fresh dataset, use a new database filename.

## 5. Train or retrain

Train for 30 epochs and save the model files under `models\demo`:

```cmd
build\Release\translate.exe train data.db models\demo 30
```

The epoch count is optional; when omitted, the program uses its default. Each training command trains a new model from scratch and writes the model and vocabulary files to the specified directory; it does not resume a previous training run.

## 6. Translate text

Use the model directory produced by training:

```cmd
build\Release\translate.exe infer models\demo "good morning"
```

Replace `good morning` with the source-language text to translate. The current tokenizer splits on whitespace and is case-sensitive.

## Quick rerun checklist

For a later session, open a fresh Visual Studio developer command prompt, then run these from the project root:

```cmd
set PATH=C:\Library\libtorch\lib;%PATH%
cmake --build build --config Release
build\Release\translate.exe infer models\demo "good morning"
```

If the database and model already exist, there is no need to run `init-db`, `import-tsv`, or `train` again unless you want to create/retrain them.
