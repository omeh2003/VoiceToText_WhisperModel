## README

Это руководство по использованию класса `SoundFileProcessor`. Он предназначен для обработки звуковых файлов, включая разделение звуковых файлов на части, изменение частоты дискретизации и транскрибацию с использованием моделей Whisper от OpenAI.

### Установка

Прежде всего, вам необходимо установить все зависимости. Вы можете сделать это двумя способами: с использованием `pip` (Python package manager) или `conda` (Anaconda package manager).

#### Установка зависимостей с использованием `pip`

1. Откройте терминал
2. Установите необходимые пакеты, выполнив следующую команду:

    ```bash
    pip install soundfile resampy pydub transformers
    ```

#### Установка зависимостей с использованием `conda`

1. Откройте терминал
2. Создайте новую среду conda, выполнив следующую команду:

    ```bash
    conda create -n myenv
    ```

3. Активируйте среду, используя следующую команду:

    ```bash
    conda activate myenv
    ```

4. Установите необходимые пакеты, выполнив следующую команду:

    ```bash
    conda install -c conda-forge soundfile resampy pydub transformers
    ```

### Установка PyTorch

Как правило, модели transformers требуют PyTorch. Вот как вы можете установить PyTorch.

#### Установка с помощью `pip`

Выполните следующую команду в терминале:

```bash
pip install torch
```

#### Установка с помощью `conda`

Выполните следующую команду в терминале:

```bash
conda install pytorch torchvision torchaudio -c pytorch
```

### Использование класса `SoundFileProcessor`

Вот базовый пример использования `SoundFileProcessor`:

```python
from sfProccessor import SoundFileProcessor
processor = SoundFileProcessor()
processor.process_file("path_to_your_audio_file.wav")
```

В этом примере "path_to_your_audio_file.wav" - это путь к аудиофайлу, который вы хотите обработать. В результате его выполнения в директории с исходным файлом будет создана поддиректория "wave", в которой будут храниться части исходного файла, а также файл "output.txt", содержащий транскрибацию аудиофайла.

## Известные проблемы

### Переполнение памяти при обработке аудиофайлов с высоким битрейтом

В некоторых случаях, если битрейт исходного аудиофайла слишком высок, может возникнуть проблема переполнения памяти. Это происходит из-за того, что аудиофайлы с высоким битрейтом могут занимать существенное количество памяти при загрузке и обработке.

#### Решение проблемы

Если вы столкнулись с этой проблемой, рекомендуется снизить битрейт исходного аудиофайла. Это можно сделать с помощью аудиоредактора, такого как Audacity или Adobe Audition. Процесс будет отличаться в зависимости от используемого программного обеспечения, но в общем случае он включает в себя следующие шаги:

1. Откройте исходный аудиофайл в вашем аудиоредакторе.
2. Найдите опцию для изменения битрейта аудио. В Audacity это можно сделать, выбрав "Tracks" > "Resample" в верхнем меню.
3. Введите новое значение битрейта (например, 16000 Гц - это стандартная частота дискретизации для многих задач обработки аудио).
4. Сохраните измененный аудиофайл и попробуйте снова выполнить обработку с помощью `SoundFileProcessor`.