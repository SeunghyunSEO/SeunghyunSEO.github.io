---
title: (Paper) Improving Speech Recognition Using Consistent Predictions on Synthesized Speech
categories: Speech_Recognition
tag: [tmp]

toc: true
toc_sticky: true

comments: true
---


이번 글에서는 [Improving Speech Recognition Using Consistent Predictions on Synthesized Speech](https://ieeexplore.ieee.org/document/9053831) 라는 논문을 요약해서 리뷰해 보려고 합니다. 

음성 합성을 통해 만든 샘플들을 가지고 음성인식에 사용해서 성능 개선을 가져오려는 방법론에 관심이 생겨서, 단순히 GAN으로 음성합성한 논문들부터 시작해 여기까지 왔네요.

ICASSP 2020년에 게재된 논문이고 Google에서 publish한 논문인데 어째서인지 논문 1저자인 Gary Wang 한분만 Simon Fraser University입니다... 검색해보니 동명이인의 구글러 [Gary Wang](https://scholar.google.co.kr/citations?hl=en&user=P8GaY4wAAAAJ&view_op=list_works&sortby=pubdate)가 있고 [이 분](https://scholar.google.com/citations?user=pR-3oMIAAAAJ&hl=en)이 계시는데 본 논문은 앞에 계신 분 scholar에 등재되었으나 논문에서의 적은 뒷분이신...


아무튼 해당 논문을 간단하게 리뷰해보도록 하겠습니다.

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

`Unsupervised Data Augmentation (UDA) Loss` 란 원래의 음성 $$x$$에 noise 등을 추가해서 $$\hat{x}$$을 만들고, 이를 기반으로 추론한 문장 $$y$의 분포를 유사하게 하는 인데요, 


$$
J_{UDA}(\theta) = \mathbb{E}_{x \in U} \mathbb{E}_{x^ \sim q(x^ \vert x)} D_{KL} (p_{\tilde{theta}} (y \vert x) \parallel p_{\theta} (y \vert \hat{x}) ) 
$$

여기서 $$q(\hat{x} \vert x)$$는 `Data Augmentation Function`입니다.
(이 loss는 [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/pdf/1904.12848) 에서 제안된 겁니다.)

`UDA`를 제안한 논문에서는 아래의 Objective를 통해서 최종적으로 네트워크를 학습했습니다. 

$$
{min}_{\theta} J(\theta) = \mathbb{E}_{x,y^{\ast} \in L} [p_{\theta}(y^{\ast} \vert x)] + \lambda J_{UDA} (\theta)
$$

![uda](/assets/images/asr_synthesized/uda.png)
*Fig. Network Architecture with Unsupervised Data Augmentation*


하지만 본 논문에서는 이는 어디까지나 Unsupervised Method이기 때문에 조금 변형해서 사용하려고 합니다.



### <mark style='background-color: #dcffe4'> Consistency Loss (Modified UDA loss) </mark>

`UDA loss`를 조금 변형해서 아래와 같이 만들었는데요, 바뀐 것은 $$q(\hat{x} \vert x)$$ 모듈이 $$q(\hat{x} \vert y^{\ast},z)$$가 됐다는 겁니다. 여기서 $$y^{\ast}$$는 원본 오디오의 타겟에 해당하는 문장이고, $$z$$는 원본 오디오에서 뽑은 `speaker conditioning information`이 담긴 벡터입니다.  

$$
J_{cons}(\theta) = \mathbb{E}_{x \in U} \mathbb{E}_{x^ \sim q(x^ \vert y^{\ast},z)} D_{KL} (p_{\tilde{theta}} (y \vert x) \parallel p_{\theta} (y \vert \hat{x}) ) 
$$



### <mark style='background-color: #dcffe4'> Overall Training Objective </mark>

다시, `UDA`를 제안한 논문에서는 아래의 Objective를 통해서 학습을 했는데요, 

$$
{min}_{\theta} J(\theta) = \mathbb{E}_{x,y^{\ast} \in L} [p_{\theta}(y^{\ast} \vert x)] + \lambda J_{UDA} (\theta)
$$

이 Objective에서는 supervised loss term이 한개지만 본 논문에서는 아래처럼 2개를 쓴다고 합니다.

$$
\begin{aligned}
&
J_{real}(\theta) = \mathbb{E}_{x,y^{\ast} \in L} [ p_{\theta} (y^{\ast} \vert x) ]
&\\

&
J_{tts}(\theta) = \mathbb{E}_{x,y^{\ast} \in L} [ p_{\theta} (y^{\ast} \vert \hat{x} \sim q( \hat{x} \vert y^{\ast}, z) ) ]
&\\
\end{aligned}
$$

그리고 최종 Objective는 아래와 같아지죠.

$$
{min}_{\theta} J(\theta) = \lambda_r J_{real}(\theta) + \lambda_t J_{tts}(\theta) + \lambda_c J_{cons}(\theta)
$$






### <mark style='background-color: #dcffe4'> Relationship to Speech Chain </mark>


![speech_chain](/assets/images/asr_synthesized/speech_chain.png)
*Fig. Speech - to - Text - to - Speech Chain.*





## <mark style='background-color: #fff5b1'> Data Augmentation, Network Reguralization and SpecAugment </mark>










## <mark style='background-color: #fff5b1'> Experiments </mark>


![asr_synthesized_table1](/assets/images/asr_synthesized/asr_synthesized_table1.png)
![asr_synthesized_table2](/assets/images/asr_synthesized/asr_synthesized_table2.png)
![asr_synthesized_table3](/assets/images/asr_synthesized/asr_synthesized_table3.png)
![asr_synthesized_table4](/assets/images/asr_synthesized/asr_synthesized_table4.png)
![asr_synthesized_table5](/assets/images/asr_synthesized/asr_synthesized_table5.png)









## <mark style='background-color: #fff5b1'> Reference </mark>

- [Improving Speech Recognition Using Consistent Predictions on Synthesized Speech](https://ieeexplore.ieee.org/document/9053831)

- [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/pdf/1904.12848) 

- [Training Neural Speech Recognition Systems with Synthetic Speech Augmentation](https://arxiv.org/abs/1811.00707)

- [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)
