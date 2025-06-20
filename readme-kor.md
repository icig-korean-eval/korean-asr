# Foreign Korean Speech Recognition with Whisper

이 레포지토리는 **외국인의 한국어 발화를 인식하여 한국어로 변환**하기 위해 OpenAI의 Whisper 모델을 파인튜닝한 프로젝트입니다.  
모델은 `whisper-base`를 기반으로 하며, AI Hub에서 제공하는 공공 데이터셋을 활용하였습니다.



## 프로젝트 개요

- **모델**: [OpenAI Whisper-Base](https://huggingface.co/openai/whisper-base)
- **데이터**: 
  [AI Hub - 인공지능 학습용 외국인 한국어 발화 음성 데이터  ](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=505)

- **기여도**: 김준철 - 100%



## 프로젝트 구조

```text
├── data.ipynb
├── data_analyze.ipynb
├── data_test_prepare.ipynb
├── train.py
└── test.py
```


### 상세 설명

- `data.ipynb`:  
  전체 데이터를 불러와 Whisper 파인튜닝 프로세스를 간단히 테스트한 노트북입니다.

- `data_analyze.ipynb`:  
  학습용(train) 데이터셋의 언어 및 라벨별 분포를 분석하고, 균형 잡힌 분포를 갖도록 재조정한 후 train/validation으로 분할하여 저장합니다.

- `data_test_prepare.ipynb`:  
  테스트용(test) 데이터를 위와 같은 방식으로 분할 및 저장하는 노트북입니다.

- `train.py`:  
  `transformers.Seq2SeqTrainer`를 이용하여 Whisper-Base 모델을 파인튜닝합니다.

- `test.py`:  
  파인튜닝된 모델을 test 데이터셋에 대해 평가하고 결과를 출력합니다.



## 결과

### 성능

- **Character Error Rate (CER)**: **2.1%**

### 모델

- [huggingface 이동](https://huggingface.co/icig/non-native-korean-speech-asr)


## License

This project is licensed under the [Apache License 2.0](LICENSE).

We use the pretrained [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) model hosted on HuggingFace, which is also licensed under Apache 2.0.

For details, see [`NOTICE`](NOTICE).
