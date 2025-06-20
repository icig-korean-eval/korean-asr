# Foreign Korean Speech Recognition with Whisper

This repository fine-tunes OpenAI's Whisper model to **transcribe Korean speech spoken by non-native speakers**.  
The model is based on `whisper-base`, and it is trained using a public dataset provided by AI Hub.



## Project Overview

- **Model**: [OpenAI Whisper-Base](https://huggingface.co/openai/whisper-base)  
- **Dataset**:  
  [AI Hub – Foreigners’ Korean Speech Dataset for AI Learning](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=505)  
- **Author**: Juncheol Kim – 100% individual contribution



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
  Analyzes the distribution of languages and labels in the training set, rebalances them, and splits the dataset into train/validation subsets.

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



## License

This project is licensed under the [Apache License 2.0](license/LICENSE).

It uses the pretrained [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) model hosted on HuggingFace, which is also licensed under Apache 2.0.

For more information, see the [`NOTICE`](license/NOTICE) file.
