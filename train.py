import glob
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from datasets import Audio
import soundfile as sf
from tqdm import tqdm

from multiprocessing import Pool, cpu_count

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 입력 오디오 특징 벡터 추출 후 padding
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 라벨 정보 padding
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # 레이블 시퀀스에 대해 최대 길이만큼 패딩 작업을 실시한다.
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # padding token을 -100으로 바꿔서 loss 계산에서 무시되도록 설정
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # BOS 토큰 제거 (Whisper는 BOS가 필수 아님)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch



def process_row(row):
    file_name = row['fileName']
    reading = row['ReadingLabelText']
    answer = row['AnswerLabelText']
    
    # 오디오 파일 찾기
    audio_files = glob.glob(f'./data/train/source/*/*/{file_name}')
    if not audio_files:
        return None  # 파일 없음

    audio_path = audio_files[0]
    
    # 오디오 파일이 정상인지 확인
    try:
        with sf.SoundFile(audio_path) as f:
            _ = f.frames
    except RuntimeError:
        return None  # 손상된 오디오

    # 텍스트는 ReadingLabelText가 우선, 없다면 AnswerLabelText
    transcript = answer if pd.isna(reading) else reading
    return (audio_path, transcript)

# tqdm이 Pool과 함께 작동하도록 helper
def process_all_rows(data):
    # 멀티프로세싱을 활용해 모든 데이터 처리
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_row, data.to_dict(orient="records")), total=len(data)))
    return results


def prepare_dataset(batch):
    # 오디오 파일을 16kHz로 로드
    audio = batch["audio"]

    # input audio array로부터 log-Mel spectrogram 변환
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # target text를 label ids로 변환
    batch["labels"] = tokenizer(batch["transcripts"]).input_ids
    return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # pad token이 -100으로 되어 있으므로 tokenizer의 pad_token_id로 되돌림
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # 디코딩 (특수 토큰 제거)
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # CER (Character Error Rate) 계산
    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


if __name__ == "__main__":
    filtered_data = pd.read_pickle('./data/train/labeling/filtered_balanced.pikl')
    filtered_data_valid = pd.read_pickle('./data/train/labeling/filtered_balanced_valid.pikl')
    filtered_data_test = pd.read_pickle('./data/train/labeling/filtered_balanced_test.pikl')

    print(f'filtered data: {len(filtered_data)}')

    # Dataset 생성
    ds = Dataset.from_dict({
        "audio": filtered_data.loc[:,'fileName'],
        "transcripts": filtered_data.loc[:,'text']
    }).cast_column("audio", Audio(sampling_rate=16000))
    ds_valid = Dataset.from_dict({
        "audio": filtered_data_valid.loc[:,'fileName'],
        "transcripts": filtered_data_valid.loc[:,'text']
    }).cast_column("audio", Audio(sampling_rate=16000))
    ds_test = Dataset.from_dict({
        "audio": filtered_data_test.loc[:,'fileName'],
        "transcripts": filtered_data_test.loc[:,'text']
    }).cast_column("audio", Audio(sampling_rate=16000))
    
    print(f'ds len: {len(ds)}')
    
    datasets = DatasetDict({
        "train": ds,
        "test": ds_test,
        "valid": ds_valid})
    
    
    # 모델 및 토크나이저 초기화
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="Korean", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="Korean", task="transcribe")
    
    # 데이터셋 전처리 적용
    low_call_voices = datasets.map(prepare_dataset, remove_columns=datasets.column_names["train"], num_proc=None)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    metric = evaluate.load('cer')
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    
    # Whisper-specific 설정
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # Trainer 정의
    training_args = Seq2SeqTrainingArguments(
        output_dir="test",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        push_to_hub=False,
    )
    
    # Trainer로 학습 및 평가
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=low_call_voices["train"],
        eval_dataset=low_call_voices["valid"],  # or "test"
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    
    trainer.train()
    
    # 평가 결과 출력
    eval_results = trainer.evaluate()
    print("Evaluation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")
    
    # 모델 저장
    trainer.save_model("./model/whisper-base-tuned.pt")
