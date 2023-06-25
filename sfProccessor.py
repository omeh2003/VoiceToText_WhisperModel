import os
import soundfile as sf
import resampy
import logging
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class SoundFileProcessor:
    """
    Класс SoundFileProcessor обрабатывает звуковые файлы.
    """

    def __init__(self, sampling_rate=16000, chunk_duration=10, overlap=2):
        """
        Конструктор класса SoundFileProcessor.

        Args:
            sampling_rate: Целевая частота дискретизации.
            chunk_duration: Продолжительность части файла.
            overlap: Перекрытие между сегментами в секундах.
        """
        self.target_sampling_rate = sampling_rate
        self.chunk_duration = chunk_duration
        self.overlap = overlap

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        self.logger.info("Загрузка модели...")
        self.processor, self.model, self.forced_decoder_ids = self._load_model()
        self.logger.info("Модель загружена.")
        if not os.path.exists("wave"):
            os.mkdir("wave")
        self.logger.info("Папка wave создана.")

    def _load_model(self):
        """
        Загружает модель и процессор Whisper.
        """
        model_dir = ".\\model\\model.ckpt"
        if not os.path.exists(model_dir):
            if not os.path.exists(".\\model"):
                os.mkdir(".\\model")
            self.logger.info("Загрузка модели...")
            # Это модель среднего размера, но вы можете использовать любую другую модель из списка:
            # мне хватает возможностей этой.
            # https://huggingface.co/models?filter=whisper
            # tiny - 60 мб, small - 200 мб, medium - 400 мб, large - 1.5 гб, large-v2 - непонятно сколько :)
            processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
            self.logger.info("Модель загружена. processor загружен.")
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
            self.logger.info("Модель загружена. model загружен.")
            processor.save_pretrained(model_dir)
            self.logger.info("Модель загружена. processor сохранен.")
            model.save_pretrained(model_dir)
            self.logger.info("Модель загружена. model сохранен.")
        else:
            processor = WhisperProcessor.from_pretrained(model_dir)
            model = WhisperForConditionalGeneration.from_pretrained(model_dir)
            self.logger.info("Модель загружена. processor и model загружены.")

        forced_decoder_ids = processor.get_decoder_prompt_ids(language="russian", task="transcribe")
        return processor, model, forced_decoder_ids

    def split_sound_file(self, file_sound):
        """
        Разделяет звуковой файл на части.
        """
        for sound in os.listdir("wave"):
            os.remove(os.path.join("wave", sound))

        # загрузка звукового файла
        sound = AudioSegment.from_file(file_sound, format=file_sound.split(".")[-1])
        self.logger.info(f"Длительность звукового файла: {len(sound) / 1000} секунд")
        # длительность частей в миллисекундах
        chunk_length_ms = 1000 * self.chunk_duration

        # разбиение звукового файла на части
        chunks = list(sound[::chunk_length_ms])
        self.logger.info(f"Количество частей: {len(chunks)}")
        # сохранение каждой части в отдельный файл
        for i, chunk in enumerate(chunks):
            chunk.export(os.path.join("wave", f"part{i}.wav"), format="wav")
        self.logger.info("Звуковой файл разбит на части.")
    def transcribe(self, segmets_file):
        """
        Распознает каждый сегмент и записывает результат в файл.
        """
        self.logger.info("Начало распознавания...")
        for segment in segmets_file:
            self.logger.info(f"Распознавание сегмента {segment}...")
            input_features = self.processor(segment, sampling_rate=self.target_sampling_rate,
                                            return_tensors="pt").input_features
            predicted_ids = self.model.generate(input_features, forced_decoder_ids=self.forced_decoder_ids)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            self.logger.info(f"Распознано: {transcription}")
            with open("output.txt", "a", encoding="utf-8") as f:
                f.write(transcription[0] + "\n")
        self.logger.info("Распознавание завершено.")

    def segmentation(self, audio_file):
        """
        Разделяет аудиофайл на сегменты.
        """
        self.logger.info("Начало разделения аудиофайла на сегменты...")
        input_audio, original_sampling_rate = sf.read(audio_file)

        # Преобразование частоты дискретизации аудиофайла к 16000 Гц
        resampled_audio = resampy.resample(x=input_audio, sr_orig=original_sampling_rate,
                                           sr_new=self.target_sampling_rate)

        segment_samples = int(self.chunk_duration * self.target_sampling_rate)
        overlap_samples = int(self.overlap * self.target_sampling_rate)

        segments = []
        start = 0
        while start < len(resampled_audio):
            end = min(start + segment_samples, len(resampled_audio))
            segment = resampled_audio[start:end]
            segments.append(segment)
            start += segment_samples - overlap_samples
        self.logger.info("Разделение аудиофайла на сегменты завершено.")

        return segments

    def process_file(self, file_path):
        """
        Обрабатывает заданный звуковой файл.
        """
        self.logger.info("Начало обработки файла...")
        self.split_sound_file(file_path)
        for file in os.listdir("wave"):
            segments_file = self.segmentation(os.path.join("wave", file))
            self.transcribe(segments_file)
        self.logger.info("Обработка файла завершена.")


if __name__ == '__main__':
    processor = SoundFileProcessor()
    processor.process_file("input.wav")
