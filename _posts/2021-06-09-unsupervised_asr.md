---
title: (Paper) Improving Speech Recognition Using Consistent Predictions on Synthesized Speech
categories: Speech_Recognition
tag: [tmp]

toc: true
toc_sticky: true

comments: true
---


이번 글에서는 [Unsupervised Speech Recognition](https://scontent-ssn1-1.xx.fbcdn.net/v/t39.8562-6/187874612_311717527241594_5668815448923437055_n.pdf?_nc_cat=102&ccb=1-3&_nc_sid=ae5e01&_nc_ohc=2rlpqCipZS4AX92nofj&_nc_ht=scontent-ssn1-1.xx&oh=ce7abb7bfc05f0269251cf8c936479e7&oe=60E59B95) 라는 논문을 요약해서 리뷰해 보려고 합니다. 

[High-performance speech recognition with no supervision at all](https://ai.facebook.com/blog/wav2vec-unsupervised-speech-recognition-without-supervision/)

---
< 목차 >
{: class="table-of-content"}
* TOC
{:toc}
---


## <mark style='background-color: #fff5b1'> Problem Definition and Contibution Point </mark>

본 논문에서 정의하는 Problem과 Contribution Point은 다음과 같습니다.

- 1.요즘 음성 합성 성능이 많이 올라왔기 때문에 이러한 합성음 데이터를 사용해서 음성인식 성능을 높히려는 노력들이 많이 있었다.
- 2.하지만 이렇게 합성음을 사용해서 data augmentation 효과를 누리는 방법은 효과적이지 못했다. (참고 : [Training neural speech recognition systems with synthetic speech augmentation](https://arxiv.org/abs/1811.00707))
- 3.그렇기 때문에 우리는 합성음과 실제 음성이 유사한 형태를 이루도록 하는 `consistency loss`를 제안한다.
- 4.방법론은 핵심은 합성음과 실제 음성의 데이터 분포를 같게 한다는 것인데, 이전에 합성음을 사용해서 음성 인식 성능을 높히고자 했던 노력이 잘 안됐던 이유가 소량의 실제 음성 데이터의 분포와 합성음의 데이터 분포의 미스매치 때문이라고 생각하기 때문이다.
- 5.전체 960시간의 LibriSpeech 데이터셋을 사용해서 학습한 것과 비교해서 460시간 음성 + 500 합성음 (transcript 사용해서) 으로 학습한게 별로 차이가 안났다. (WER 0.2% 내외)
- 6.결론은 우리가 제안한 방법론을 바탕으로 무수히 많은 text만 있다면 오디오가 절반밖에 없는 상황에서도 좋은 성능을 낼 수 있을것이다. 


## <mark style='background-color: #fff5b1'> Generating Consistent Predictions </mark>

### <mark style='background-color: #dcffe4'> Unsupervised Data Augmentation (UDA) Learning </mark>


### <mark style='background-color: #dcffe4'> Consistency Loss (Modified UDA loss) </mark>



### <mark style='background-color: #dcffe4'> Overall Training Objective </mark>




### <mark style='background-color: #dcffe4'> Relationship to Speech Chain </mark>









## <mark style='background-color: #fff5b1'> Experiments </mark>




## <mark style='background-color: #fff5b1'> Reference </mark>

