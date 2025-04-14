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


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 인풋 데이터와 라벨 데이터의 길이가 다르며, 따라서 서로 다른 패딩 방법이 적용되어야 한다. 그러므로 두 데이터를 분리해야 한다.
        # 먼저 오디오 인풋 데이터를 간단히 토치 텐서로 반환하는 작업을 수행한다.
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Tokenize된 레이블 시퀀스를 가져온다.
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # 레이블 시퀀스에 대해 최대 길이만큼 패딩 작업을 실시한다.
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 패딩 토큰을 -100으로 치환하여 loss 계산 과정에서 무시되도록 한다.
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 이전 토크나이즈 과정에서 bos 토큰이 추가되었다면 bos 토큰을 잘라낸다.
        # 해당 토큰은 이후 언제든 추가할 수 있다.
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

    # 텍스트 결정
    transcript = answer if pd.isna(reading) else reading
    return (audio_path, transcript)

# tqdm이 Pool과 함께 작동하도록 helper
def process_all_rows(data):
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
    
    batch["input_features"]  = processor(batch["input_features"], sampling_rate=16000, return_tensors="pt").input_features
    return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # pad_token을 -100으로 치환
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # metrics 계산 시 special token들을 빼고 계산하도록 설정
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


def map_to_pred(batch):
    audio = batch["audio"]
    print(audio)
    
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch["reference"] = processor.tokenizer._normalize(batch['transcripts'])

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    
    print(batch['transcripts'])
    print(processor.decode(predicted_ids, skip_special_tokens=True))
    return batch


if __name__ == "__main__":
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="Korean", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("./model/whisper-base-tuned.pt").to("cuda")

    try:
        with open('./data/validation/test.pickle', 'rb') as f:
            low_call_voices = pickle.load(f)
    except:
        filtered_data = pd.read_pickle('./data/train/labeling/filtered_balanced_val_test.pikl')

        print(f'filtered data: {len(filtered_data)}')

        # Dataset 생성
        ds = Dataset.from_dict({
            "audio": filtered_data.loc[:5,'fileName'],
            "transcripts": filtered_data.loc[:5,'text']
        }).cast_column("audio", Audio(sampling_rate=16000))
        
        print(f'ds len: {len(ds)}')
        
        # datasets = DatasetDict({
        #     "train": ds})
        
        # low_call_voices = datasets.map(prepare_dataset, remove_columns=datasets.column_names["train"], num_proc=None)
        # with open('./data/validation/test.pickle', 'wb') as f:
        #     pickle.dump(low_call_voices, f, pickle.HIGHEST_PROTOCOL)
    # print(low_call_voices)
    
    result = ds.map(map_to_pred)
    
    metric = evaluate.load('cer')
    print(100 * metric.compute(references=result["reference"], predictions=result["prediction"]))
    
    # data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # model.config.forced_decoder_ids = None
    # model.config.suppress_tokens = []
    
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir="test",  # 원하는 리포지토리 이름을 임력한다.
    #     per_device_train_batch_size=16,
    #     gradient_accumulation_steps=1,  # 배치 크기가 2배 감소할 때마다 2배씩 증가
    #     learning_rate=1e-5,
    #     warmup_steps=500,
    #     max_steps=4000,  # epoch 대신 설정
    #     gradient_checkpointing=True,
    #     fp16=True,
    #     evaluation_strategy="steps",
    #     per_device_eval_batch_size=8,
    #     predict_with_generate=True,
    #     generation_max_length=225,
    #     save_steps=1000,
    #     eval_steps=1000,
    #     logging_steps=25,
    #     report_to=["tensorboard"],
    #     load_best_model_at_end=True,
    #     metric_for_best_model="cer",  # 한국어의 경우 'wer'보다는 'cer'이 더 적합할 것
    #     greater_is_better=False,
    #     push_to_hub=False,
    # )
    
    # trainer = Seq2SeqTrainer(
    #     args=training_args,
    #     model=model,
    #     eval_dataset=low_call_voices["train"],  # or "test"
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    #     tokenizer=processor.feature_extractor,
    # )

    
    # eval_results = trainer.evaluate()
    # print("Evaluation results:")
    # for key, value in eval_results.items():
    #     print(f"  {key}: {value}")
        
    # trainer.save_model("./model/whisper-base-tuned.pt")
    
    # predictions = []
    # references = []
    
    # model.eval()

    # for sample in tqdm(low_call_voices['train']):
    #     # sample["audio"]는 이미 Audio feature로 캐스팅되어 있으므로, processor에 그대로 전달합니다.
    #     # processor는 WhisperProcessor로, sampling_rate=16000, return_tensors="pt" 옵션 사용
    #     # input_features = processor(sample["input_features"], sampling_rate=16000, return_tensors="pt").input_features.to(model.device)
    #     input_features = sample["input_features"].to(model.device)
        
    #     # 모델에 입력하여 예측 (max_length 등은 필요에 따라 조절)
    #     generated_ids = model.generate(input_features, max_length=225)
        
    #     # 생성된 토큰을 텍스트로 디코딩 (skip_special_tokens 옵션으로 특수 토큰 제거)
    #     pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
    #     predictions.append(pred_text)
    #     references.append(sample["labels"])

    # # CER 계산 (predictions와 references 리스트 사용)
    # results = metric.compute(predictions=predictions, references=references)
    # print("Direct test CER:", results["cer"])