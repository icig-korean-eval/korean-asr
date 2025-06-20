# Foreign Korean Speech Recognition with Whisper

- Fine-tuned OpenAI’s Whisper model to transcribe Korean speech spoken by non-native speakers
- Model: Fine-tuned version of [whisper-base](https://huggingface.co/openai/whisper-base)
- Training data: [Korean Speech by Foreigners for AI Learning](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=505) provided by AI Hub



## Project Overview

While OpenAI’s Whisper model supports multilingual speech recognition including Korean, 
its performance is significantly limited when it comes to recognizing **Korean spoken by non-native speakers**.

To address this, we fine-tuned the pretrained [`whisper-base`](https://huggingface.co/openai/whisper-base) model 
using the **[Korean Speech by Foreigners for AI Learning](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=505)** dataset from AI Hub 
to improve recognition accuracy for non-native Korean speech.


## Dataset

- Dataset used: [AI Hub – Korean Speech by Foreigners for AI Learning](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=505)

### Dataset Analysis

- Total duration: approx. **4,300 hours**
- Total samples: approx. **1.2 million** utterance-text pairs
- Total size: approx. **340 GB**

Due to limited GPU resources, we were not able to train on the entire dataset. Instead, we applied the following filtering and sampling strategy:

- Selected only speakers whose **native language** is English, Chinese, Vietnamese, Thai, or Japanese
- Included only audio samples with **duration between 8 and 14 seconds**
- To ensure balance, we **uniformly sampled** an equal number of utterances for each language group by category

- See below for full dataset distribution

  | languageClass | culture1 | culture2 | general | life1 | life2 |
  |---------------|----------|----------|---------|--------|--------|
  | Vietnamese        | 18913    | 14345    | 21434   | 19242  | 17255  |
  | English           | 1489     | 943      | 8512    | 1797   | 1905   |
  | Japanese         | 27720    | 25192    | 25539   | 24629  | 24275  |
  | Chinese         | 31587    | 0        | 38643   | 33311  | 29151  |

- Final dataset distribution
  - trian

    | languageClass | culture1 | general | life1 | life2 |
    |---------------|----------|---------|--------|--------|
    | Vietnamese        | 1489     | 1489    | 1489   | 1489   |
    | English           | 1489     | 1489    | 1489   | 1489   |
    | Japanese         | 1489     | 1489    | 1489   | 1489   |
    | Chinese         | 1489     | 1489    | 1489   | 1489   |
  - validation

    | languageClass | culture1 | general | life1 | life2 |
    |---------------|----------|---------|--------|--------|
    | Vietnamese        | 308      | 308     | 308    | 308    |
    | English           | 0        | 308     | 308    | 308    |
    | Japanese         | 308      | 308     | 308    | 308    |
    | Chinese         | 308      | 308     | 308    | 308    |
  - test

    | languageClass | culture1 | general | life1 | life2 |
    |---------------|----------|---------|--------|--------|
    | Vietnamese        | 108      | 108     | 108    | 108    |
    | English           | 0        | 108     | 0      | 108    |
    | Japanese         | 108      | 108     | 108    | 108    |
    | Chinese         | 108      | 108     | 108    | 108    |


## Training Method

- Fine-tuning was performed using HuggingFace’s `transformers` library with the **`Seq2SeqTrainer`** API.


## Project Structure

```text
├── data.ipynb
├── data_analyze.ipynb
├── data_test_prepare.ipynb
├── train.py
└── test.py
```

### File Descriptions

- `data.ipynb`:  
  A notebook for initial testing of the fine-tuning process using the full dataset.

- `data_analyze.ipynb`:  
  Analyzes the distribution of languages and labels in the training set, rebalances them, and splits the dataset into train/validation/test subsets.

- `data_test_prepare.ipynb`:  
  Prepares the test dataset in the same way as the training set (rebalancing and splitting).

- `train.py`:  
  Fine-tunes the Whisper-base model using `transformers.Seq2SeqTrainer`.

- `test.py`:  
  Evaluates the fine-tuned model on the test set and outputs the results.



## Results

### Performance

- **Character Error Rate (CER)**: **2.1%**

### Model

- [View on HuggingFace](https://huggingface.co/icig/non-native-korean-speech-asr)


## Contribution

- JoonChul Kim - 100%
  - All development, implementation, and documentation were completed individually.


## License

This project is licensed under the [Apache License 2.0](LICENSE).

It uses the pretrained [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) model hosted on HuggingFace, which is also licensed under Apache 2.0.

For more information, see the [`NOTICE`](NOTICE) file.
