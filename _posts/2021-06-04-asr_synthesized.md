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
J_{UDA}(\theta) = \mathbb{E}_{x \in U} \mathbb{E}_{x^ \sim q(x^ \vert x)} D_{KL} (p_{\tilde{\theta}} (y \vert x) \parallel p_{\theta} (y \vert \hat{x}) ) 
$$

여기서 $$q(\hat{x} \vert x)$$는 `Data Augmentation Function`입니다.
(이 loss는 [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/pdf/1904.12848) 에서 제안된 겁니다.)

`UDA`를 제안한 논문에서는 아래의 Objective를 통해서 최종적으로 네트워크를 학습했습니다. 

$$
{min}_{\theta} J(\theta) = \mathbb{E}_{x,y^{\ast} \in L} [p_{\theta}(y^{\ast} \vert x)] + \lambda J_{UDA} (\theta)
$$

![uda](/assets/images/asr_synthesized/uda.png)
*Fig. Network Architecture with Unsupervised Data Augmentation*

위의 그림을 보시면 수식 이해가 편하실 것 같은데요, 그림에서 왼쪽은 $$\mathbb{E}_{x,y^{\ast} \in L} [p_{\theta}(y^{\ast} \vert x)]$$ 부분으로 Supervised Learning을 하는것이고, 오른쪽은 현재의 모델 파라메터 $$\theta$$를 카피한 $$\tilde{\theta}$$로 원본 $$x$$를 given으로 추론한 $$p(y \vert x)$$와 노이즈를 추가하는 등의 Augmentation방법으로 만들어진 $$\hat{x}$$를 given으로 예측한 $$p(y \vert \hat{x})$$를 유사하게 만드는 겁니다. $$\tilde{\theta}$$ 쪽에는 그래디언트가 흐르지 않고 다른 쪽은 흐릅니다.
 

본 논문에서는 이를 조금 변형한 loss를 사용하려고 합니다.



### <mark style='background-color: #dcffe4'> Consistency Loss (Modified UDA loss) </mark>

논문에서는 `UDA loss`를 조금 변형해서 아래와 같이 만들었는데요, 바뀐 것은 $$q(\hat{x} \vert x)$$ 모듈이 $$q(\hat{x} \vert y^{\ast},z)$$가 됐다는 겁니다. 여기서 $$y^{\ast}$$는 원본 오디오의 타겟에 해당하는 문장이고, $$z$$는 원본 오디오에서 뽑은 `speaker conditioning information`이 담긴 벡터입니다.  

$$
\begin{aligned}
&
J_{UDA}(\theta) = \mathbb{E}_{x \in U} \mathbb{E}_{x^ \sim q(x^ \vert x)} D_{KL} (p_{\tilde{\theta}} (y \vert x) \parallel p_{\theta} (y \vert \hat{x}) ) 
& \\

&
J_{cons}(\theta) = \mathbb{E}_{x \in U} \mathbb{E}_{x^ \sim q(x^ \vert y^{\ast},z)} D_{KL} (p_{\tilde{\theta}} (y \vert x) \parallel p_{\theta} (y \vert \hat{x}) ) 
& \\
\end{aligned}
$$



### <mark style='background-color: #dcffe4'> Overall Training Objective </mark>

위의 consistency loss를 활용해서 본 논문에서는 아래의 Objective를 구성했는데요. 
이 Objective에서는 UDA와는 다르게 supervised loss term을 아래처럼 2개를 사용하고, 이와 더불어 consistency loss를 결합해서 최종 loss를 구성했습니다.

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

최종 `Objective`는 아래가 됩니다.

$$
{min}_{\theta} J(\theta) = \lambda_r J_{real}(\theta) + \lambda_t J_{tts}(\theta) + \lambda_c J_{cons}(\theta)
$$

즉 예를들어 10개의 `speech-text` 페어가 존재하고 10개의 `text only` 데이터가 존재한다고 하면 학습 방법은 다음과 같습니다.


- `speech-text pair` data
  - $$J_{real}(\theta)$$로 ASR학습을 한다. 정답 text를 기반으로 합성을 한 뒤에 이를 바탕으로 $$J_{tts}(\theta)$$, $$J_{cons}(\theta)$$ 를 구하고 이를 최소화 하는 방향으로 학습한다. 
  - 이렇게 할 경우 원본 음성, TTS Augmented 음성 두 개로 인식을 하는 효과를 누리는데, 이 와중에 TTS Augmented 음성은 실제 음성 데이터 분포와 크게 다르지 않게 제약이 걸려있기 때문에 네트워크 입장에선 더욱 리얼한 음성으로 학습하는 것이 된다. 
- `text only` data
  - $$J_{tts}(\theta)$$로만 학습을 한다.
  - 중요한 점은 speech-text pair 데이터로 부터의 학습 과정을 통해서 TTS 모델이 실제와 같은 (여기서 실제라함은 10개 페어 데이터 라고 해야할듯) 음성을 만들게 됐기 때문에, 네트워크 입장에서 괴리없이 학습할 수 있다. 







### <mark style='background-color: #dcffe4'> Relationship to Speech Chain </mark>

논문에서는 제안하는 Approach가 사실은 `Speech Chain`와 유사한 점이 있다고 쓰여있는데요, [Listening while Speaking: Speech Chain by Deep Learning](https://arxiv.org/pdf/1707.04879)라는 논문에서 제안된 Speech Chain은 아래와 같습니다.

![speech_chain2](/assets/images/asr_synthesized/speech_chain2.png)
*Fig. Speech Chain*

Speech Chain은 기본적으로 ASR TTS가 서로 반복하면서 학습됨으로써 둘 다 성능 향상을 가져올 것이라는 생각에 제안된 방법론으로 기계 번역 분야의 `back translation`과도 유사하다고 하는데요,
`ASR -> TTS (Speech-to-text-to-speech)`와 `TTS -> ASR (Text-to-Speech-to-Text)`로 나뉘어져 있다고 합니다.
다른 한쪽 모듈이 학습될 때 반대 모듈은 fix한다던가 하는 식으로 학습이 된다고 쓰여있고, 음성인식한 문장이 `argmax operator`를 사용하기 때문에 gradient가 전파되지 않는다는 문제점은 아래와 같이 `gumbel softmax`를 쓰는 방식으로 해결했다고 합니다.  


![speech_chain](/assets/images/asr_synthesized/speech_chain.png){: width="70%"}
*Fig. Speech-to-Text-to-Speech Chain.*

본 논문에서는 $$J_{tts}(\theta)$$부분이 `TTS -> ASR (Text-to-Speech-to-Text)`와 닮아있다고 얘기를 합니다.









## <mark style='background-color: #fff5b1'> Experiments </mark>

논문에는 이 밖에도 글을 논리적으로 전개하기 위한 문단들이 더 있는데요, 간단하게 요약하는 것 치고 글이 너무 길어질 것 같아 생략하고 바로 결과로 넘어가도록 하겠습니다.
본 논문에 사용된 ASR Network는 `LAS`이며 TTS Network는 `Tacotron 2` 입니다.


실험에 사용한 데이터셋은 잘 알려진 Public English Dataset인 `LibriSpeech`이며, 
이 중에서 대부분의 실험에 `Speech-text pair` 데이터로 사용된 것은 `train-clean-460`으로 460시간짜리 깨끗한 음성을 사용했고, 좀 더 어려운 데이터셋인 500시간짜리 noisy한 데이터 `train-other-500`는 Ablation Study를  나중에 따로 사용했습니다.

![asr_synthesized_table1](/assets/images/asr_synthesized/asr_synthesized_table1.png)

`Table 1`에 있는 첫 번째 실험 결과는 절반의 데이터인 460시간만으로 Seq2Seq 모델을 Supervised Leaning 했을 때의 결과입니다. 
음서인식의 성능을 나타내는 WER이 10입니다 (낮을수록 에러 없이 인식을 잘 했다는 것).

$$
J(\theta) = J_{real}(\theta) 
$$


여기에 TTS를 joint로 트레이닝 햇을 때 성능이 $$0.5$$정도 올랐다는 걸 알 수 있는데,

$$
J(\theta) = \lambda_r J_{real}(\theta) + \lambda_t J_{tts}(\theta) 
$$

$$
J_{tts}(\theta) = \mathbb{E}_{x,y^{\ast} \in L} [ p_{\theta} (y^{\ast} \vert \hat{x} \sim q( \hat{x} \vert y^{\ast}, z) ) ]
$$

$$ J_{tts}(\theta) $$ 에서도 합성으로 만들어낸 데이터를 이용해 파라메터 $$\theta$$를 업데이트 하기 때문에 성능 개선이 있는것은 당연합니다.


여기에 마지막으로 `consistency loss`를 추가해서 `합성음으로 데이터를 증강하긴 할건데, 실제음 (원래의 460 시간) 에 가까운 합성음으로 하고싶다.` 라는 효과를 누려서

$$
J(\theta) = \lambda_r J_{real}(\theta) + \lambda_t J_{tts}(\theta) + \lambda_c J_{cons}(\theta)
$$

최종적으로 $$8.3$$이라는 성능을 얻었습니다.
즉 첫 번째 실험은 `과연 TTS로 증강하는게 효과가있는가?` 와 consistency loss를 통해서 `실제 데이터와 합성 데이터의 분포를 매칭시킨다는 가정이 작용했는가?` 를 검증한셈이 됩니다.
 

![asr_synthesized_table2](/assets/images/asr_synthesized/asr_synthesized_table2.png)

`Table 2`의 결과는 TTS + Consistency loss를 통해서 합성음으로 데이터 증강을 한 것이, 단순히 음성에 가우시안 노이즈를 섞은 것 보다 얼마나 좋았는지를 나타냅니다.
당연히 제안한 기법이 더 좋네요. 
 
![asr_synthesized_table3](/assets/images/asr_synthesized/asr_synthesized_table3.png)

`Table 3`은 화자 정보를 담당하는 `d-vector`를 랜덤하게 셔플했을 때의 효과와 음성 합성 inference를 다양하게 바꿔가면서 성능 변화를 관찰했습니다.
화자 정보를 랜덤하게 주면 당연히 Augmentation을 하기 전 원본 데이터는 여자가말했는데, Augmentation한 데이터는 남성이 말했다던가 하는 다양성이 추가되기 때문에 더 잘될거라고 하며,  `inference`의 경우에는 Tacotron의 특성상 음성을 `autoregressive`하게 만들어 낼 때 conditioned vector를 이전까지 만들어진 벡터들로 하느냐 (full inference), ground truth 로 하느냐 (teacher forcing) 등의 차이를 실험해 본 것입니다.

![asr_synthesized_table4](/assets/images/asr_synthesized/asr_synthesized_table4.png)

`Table 4`는 SpecAugment와의 양립가능성에 대한 실험인데요, 결과적으로 다 같이 쓴게 제일 좋습니다.

![asr_synthesized_table5](/assets/images/asr_synthesized/asr_synthesized_table5.png)










## <mark style='background-color: #fff5b1'> Reference </mark>

- [Improving Speech Recognition Using Consistent Predictions on Synthesized Speech](https://ieeexplore.ieee.org/document/9053831)

- [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)

- [Listening while Speaking: Speech Chain by Deep Learning](https://arxiv.org/pdf/1707.04879)

- [Training Neural Speech Recognition Systems with Synthetic Speech Augmentation](https://arxiv.org/abs/1811.00707)
