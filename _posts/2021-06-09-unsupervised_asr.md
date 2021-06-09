---
title: (Paper) Unsupervised Speech Recognition
categories: Speech_Recognition
tag: [tmp]

toc: true
toc_sticky: true

comments: true
---


이번 글에서는 [Unsupervised Speech Recognition](https://scontent-ssn1-1.xx.fbcdn.net/v/t39.8562-6/187874612_311717527241594_5668815448923437055_n.pdf?_nc_cat=102&ccb=1-3&_nc_sid=ae5e01&_nc_ohc=2rlpqCipZS4AX92nofj&_nc_ht=scontent-ssn1-1.xx&oh=ce7abb7bfc05f0269251cf8c936479e7&oe=60E59B95) 라는 논문을 요약해서 리뷰해 보려고 합니다. 
본 논문은 `Facebook AI Research (FAIR)` 에서 publish한 논문이며, 이를 다룬 Blog post, [High-performance speech recognition with no supervision at all](https://ai.facebook.com/blog/wav2vec-unsupervised-speech-recognition-without-supervision/)의 제목에서도 알 수 있듯이, `Supervised Finetuning` 을 "전혀" 안하고 음성인식이 가능하게 한 논문입니다.

???

네 사실 감이 안오실 수도 있는데요, 논문의 그림을 보시면 한순간에 "아 이렇게?" 라고 하실 수 있을 정도로 심플하니 바로 본론으로 가시죠.

---
< 목차 >
{: class="table-of-content"}
* TOC
{:toc}
---


## <mark style='background-color: #fff5b1'> Problem Definition and Contibution Points </mark>

본 논문에서 정의하는 Problem과 Contribution Point은 다음과 같습니다.

- 1.MIT CSAIL의 [Speech2Vec](https://arxiv.org/pdf/1803.08976) 등을 시작으로 FAIR는 Wav2Vec, VQ-Wav2Vec, `Wav2Vec 2.0` 등을 내왔다. 이는 [BERT](https://arxiv.org/pdf/1810.04805)와 유사한 방법론으로 특히 labeled data가 많아야 하지만 구축하는데 비용이 비싼 음성인식 분야에서 획기적인 성과를 거뒀다.
- 2.Self-Supervised Learning (SSL) 방법론을 사용한 Wav2Vec 2.0은 1번에서 언급한 것 처럼 unlabeled data를 무지막지하게 사용해서 사전학습을 했기 때문에, 파인튜닝을 할 때에는 심지어 10분정도의 labeld data만 있어도 어느정도 음성인식이 잘 되는 놀라운 결과를 보여줬지만, 이는 여전히 labeld data가 필요하다는 문제가 있다.
- 3.본 논문에서는 unlabeld speech, unlabeld text (두 음성, 텍스트 데이터는 전혀 연관이 없다. 정답 관계가 아니라는 것)를 사용해서 `Generative Adversarial Network (GAN)` 기반의 네트워크를 구성해서 `Totally Unsupervised ASR` 네트워크 (학습 방법론)을 제시한다.
- 4.Librispeech 데이터셋에서 supervised에 견주는 성능을 보여줬다. (근데 Language Model (LM)은 썼음)



## <mark style='background-color: #fff5b1'> Illustration of Wav2Vec-U </mark>

[wav2vec-u_figure1](/assets/images/unsupervised_asr/wav2vec-u_figure1.png)
*Fig. Overall Architecture of Wav2Vec-U*



## <mark style='background-color: #fff5b1'> Background </mark>

### <mark style='background-color: #dcffe4'> Pre-training : Wav2Vec 2.0 </mark>

[wav2vec2.0](/assets/images/unsupervised_asr/wav2vec2.0.png)
*Fig. 이미지 출처 : [Applying Wav2vec2.0 to Speech Recognition in Various Low-resource Languages](https://arxiv.org/pdf/2012.12121)*


[wav2vec2.0_codebook](/assets/images/unsupervised_asr/wav2vec2.0_codebook.png)
*Fig. 이미지 출처 : [UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data](https://arxiv.org/pdf/2101.07597)*


### <mark style='background-color: #dcffe4'> fine-tuning : End-to-End Supervised Learning </mark>





## <mark style='background-color: #fff5b1'> Proposed Method </mark>

### <mark style='background-color: #dcffe4'> Objective </mark>

### <mark style='background-color: #dcffe4'> Segmenting the Audio Signal </mark>

### <mark style='background-color: #dcffe4'> Pre-processing the Text Data </mark>

### <mark style='background-color: #dcffe4'> What features from Wav2Vec? </mark>

### <mark style='background-color: #dcffe4'> Unsupervised Cross-Validation Metric </mark>




## <mark style='background-color: #fff5b1'> Experiments and Results </mark>





## <mark style='background-color: #fff5b1'> Reference </mark>

