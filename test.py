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

import pickle


def map_to_pred(batch):
    audio = batch["audio"]
    print(audio)
    
    # Whisper 입력 포맷으로 변환
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    # 정답 텍스트 정규화
    batch["reference"] = processor.tokenizer._normalize(batch['transcripts'])

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)
    # 예측 결과 정규화
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    
    print(batch['transcripts'])
    print(processor.decode(predicted_ids, skip_special_tokens=True))
    return batch


if __name__ == "__main__":
    # Whisper processor 및 사전학습된 파인튜닝 모델 로드
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="Korean", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("./model/whisper-base-tuned.pt").to("cuda")

    try:
        # 전처리된 평가 데이터를 pickle에서 불러오기
        with open('./data/validation/test.pickle', 'rb') as f:
            low_call_voices = pickle.load(f)
    except:
        # pickle이 없을 경우 raw 데이터 로딩 및 Dataset 생성
        filtered_data = pd.read_pickle('./data/train/labeling/filtered_balanced_val_test.pikl')

        print(f'filtered data: {len(filtered_data)}')

        # Dataset 생성
        ds = Dataset.from_dict({
            "audio": filtered_data.loc[:5,'fileName'],
            "transcripts": filtered_data.loc[:5,'text']
        }).cast_column("audio", Audio(sampling_rate=16000))
        
        print(f'ds len: {len(ds)}')
    
    result = ds.map(map_to_pred)
    
    # CER 계산
    metric = evaluate.load('cer')
    print(100 * metric.compute(references=result["reference"], predictions=result["prediction"]))
    