# Foreign Korean Speech Recognition with Whisper

- 외국인의 한국어 발화를 인식하여 한국어로 변환하기 위해 OpenAI의 Whisper 모델을 파인튜닝
- 모델: [whisper-base](https://huggingface.co/openai/whisper-base) 파인튜닝
- AI Hub에서 제공하는 '[인공지능 학습용 외국인 한국어 발화 음성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=505)'로 학습



## 프로젝트 개요

OpenAI의 Whisper는 한국어를 포함한 다국어 음성 인식을 지원하지만, **한국어 네이티브가 아닌 외국인의 발화에 대해서는 인식 정확도가 매우 낮습니다**.  
이에 따라, 사전 학습된 [`whisper-base`](https://huggingface.co/openai/whisper-base) 모델을
**AI Hub의 '인공지능 학습용 외국인 한국어 발화 음성 데이터'**로 파인튜닝하여 외국인 화자의 한국어 발화를 더 정확하게 전사할 수 있도록 개선하였습니다.


## 데이터셋

- 사용한 데이터: 
  [AI Hub – 인공지능 학습용 외국인 한국어 발화 음성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=505)

### 데이터 분석

- 전체 분량: 약 **4300시간**, 총 **약 120만 개**의 음성-문장 페어
- 전체 용량: 약 **340GB**
- 제한된 GPU 자원으로 인해 전체 데이터셋을 학습에 사용할 수는 없었으며, 아래 기준에 따라 필터링 및 샘플링을 수행하였습니다:
  - **모국어가 영어, 중국어, 베트남어, 태국어, 일본어**인 화자만 선택
  - **8초 이상 14초 미만** 길이의 음성만 사용
  - 각 언어별로 균형 잡힌 분포를 위해 **동일한 수의 샘플을 무작위 추출**하여 학습에 활용
- 전체 데이터셋 분포  

  | languageClass | culture1 | culture2 | general | life1 | life2 |
  |---------------|----------|----------|---------|--------|--------|
  | 베트남어        | 18913    | 14345    | 21434   | 19242  | 17255  |
  | 영어           | 1489     | 943      | 8512    | 1797   | 1905   |
  | 일본어         | 27720    | 25192    | 25539   | 24629  | 24275  |
  | 중국어         | 31587    | 0        | 38643   | 33311  | 29151  |

- 최종 데이터셋 분포
  - trian

    | languageClass | culture1 | general | life1 | life2 |
    |---------------|----------|---------|--------|--------|
    | 베트남어        | 1489     | 1489    | 1489   | 1489   |
    | 영어           | 1489     | 1489    | 1489   | 1489   |
    | 일본어         | 1489     | 1489    | 1489   | 1489   |
    | 중국어         | 1489     | 1489    | 1489   | 1489   |
  - validation

    | languageClass | culture1 | general | life1 | life2 |
    |---------------|----------|---------|--------|--------|
    | 베트남어        | 308      | 308     | 308    | 308    |
    | 영어           | 0        | 308     | 308    | 308    |
    | 일본어         | 308      | 308     | 308    | 308    |
    | 중국어         | 308      | 308     | 308    | 308    |
  - test

    | languageClass | culture1 | general | life1 | life2 |
    |---------------|----------|---------|--------|--------|
    | 베트남어        | 108      | 108     | 108    | 108    |
    | 영어           | 0        | 108     | 0      | 108    |
    | 일본어         | 108      | 108     | 108    | 108    |
    | 중국어         | 108      | 108     | 108    | 108    |


## 학습 방법

- 파인튜닝 방식: HuggingFace의 `transformers` 라이브러리에서 제공하는 **`Seq2SeqTrainer`** 를 이용해 학습 진행


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
  학습용(train) 데이터셋의 언어 및 라벨별 분포를 분석하고, 균형 잡힌 분포를 갖도록 재조정한 후 train/validation/test으로 분할하여 저장합니다.

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


## 기여

- 김준철 - 100%
  - 모든 작업 진행


## License

This project is licensed under the [Apache License 2.0](LICENSE).

We use the pretrained [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) model hosted on HuggingFace, which is also licensed under Apache 2.0.

For details, see [`NOTICE`](NOTICE).
