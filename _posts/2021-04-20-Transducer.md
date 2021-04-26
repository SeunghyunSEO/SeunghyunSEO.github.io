---
title: (미완)Transducer
categories: Speech_Recognition
tag: [tmp]

toc: true
toc_sticky: true
---

---
< 목차 >
{: class="table-of-content"}
* TOC
{:toc}
---

[Transducer]((https://arxiv.org/pdf/1211.3711)) 모델이란 [Connectionist Temporal Classification (CTC)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) 을 제안한 Alex Graves 가 2012년에 처음 제안한 개념으로 CTC의 단점을 보완한 업그레이드 버전이라고 이야기 할 수 .
일반적인 경우 Recurrent Neural Networks (RNNs) 를 내부 모듈로 사용하기 때문에 RNN-Tranducer (RNN-T) 라고 부르곤 하지만, 최근에는 [Transformer-Transducer](https://arxiv.org/pdf/2002.02562) 가 제안되는 등 다양한 Variation이 존재하기 때문에 Transducer라고 부르도록 하겠습니다.


```
시작하기에 앞서, 본 포스트는 다른 자료들과 논문들의 그림을 상당 부분 참고하였음을 밝히며 출처는 Reference에 있습니다.
```

## <mark style='background-color: #fff5b1'> Common Approaches for Deep Learning based E2E ASR Model) </mark>
데이터와 적당한 모델만 있으면 되는 딥러닝 기반 End-to-End (E2E) 음성인식 (Automatic Speech Recognition, ASR) 모델과 달리 전통적인 ASR 모델은 음성에 대한 전문적인 지식 (Domain Knowledge) 딥러닝 기법과 비교해 상당히 복잡한 모델링이 요구되었습니다.

![traditional_asr](/assets/images/rnnt/shinji1.png)
*Fig. 딥러닝 기반 E2E ASR 모델이 제안되기 전의 음성인식 모델*

하지만 딥러닝기반 E2E기법들이 차례대로 등장하면서 CTC(2006~), Transducer(2012~), Attention-based Seq2Seq(2014~) 복잡한 모델링이 없이도 엄청난 음성 인식 성공률을 보여주며 음성인식의 인식 레벨은 한단계 업그레이드 되었습니다.

![e2e_asr](/assets/images/rnnt/asr.png)
*Fig. 일반적인 딥러닝 기반 E2E ASR 기법들, 왼쪽부터 차례대로 CTC, Tranducer, Attention 기반 기법들이다.*

(이미지 출처 : [Sequence-to-sequence learning with Transducers from Loren Lugosch](https://lorenlugosch.github.io/posts/2020/11/transducer/))


이 중 CTC, ATtention 기반 기법과 다르게, Transducer는 상대적으로 적게 연구가 되었지만 최근 실시간 음성인식 (Straeming-ASR) 의 중요성등이 대두되면서 주목을 받아왔습니다.
Transducer는 앞서 말한 것 처럼 CTC의 업그레이드 버전인데, CTC 또한 최근 Attention기반 기법과 비교해 성능이 뒤쳐지지 않는다는 논문들이 많이 나오고 있기도 합니다.
그렇기에 최근에는 Transducer와 Attention 기반 Sequence-to-Sequence(Seq2Seq) 모델 (LAS나 트랜스포머 기반 기법)을 같이 사용하는 Two-Pass 기법이 제안되기도 해왔습니다. 


아무튼 우리는 이제 CTC와 Attention 기반 기법에 대해 간단하게 알아보고 Tranducer 모델들에 대해서 알아보도록 하겠습니다.



### <mark style='background-color: #dcffe4'> CTC-based model (2006, 2014, ...) </mark>

음성인식같은 어떤 길이가 서로 다른 입력 Sequence를 출력 Sequence로 Mapping 시키는 task의 가장 큰 문제점은 Alignment가 맞지 않는다는 것입니다.
예를 들어 "안녕하세요, 저는 대학원생 입니다." 라고 녹음된 전화 음성이 있다고 해 보도록 하겠습니다.
이 음성의 길이가 2초일 경우 sampling rate가 8000이라면 16000차원인데, 이를 입력으로 사용해 디코딩 해야 할 정답 길이는 19차원(글자) 라는 미스 매치를 Miss-Alignment 문제라고 합니다.
이를 해결 하기 위해 2006년에 제안된 방법이 바로 Connectionist Temporal Classification (CTC) loss 입니다.

CTC는 복잡한 철학과 이론이 있지만, 짧게 요약하자면 아래와 그림으로 나타낼 수 있습니다.

![ctc](/assets/images/rnnt/shinji2.png)
*Fig. CTC 기반 ASR Model, 입력 음성에 대해서 인코딩을 하고 (보통 디코더에 비해서 인코더가 엄청 큽니다.) 디코더는 각 Encodede Representation Vectors를 토큰들로 하나씩 매핑해줍니다. 그리고는 <blank> 토큰과 중복되는 토큰들을 규칙에 따라 제거하여 최종 출력 문장을 만듭니다.*

모델이 하는 일은 입력을 인코더에 통과시켜 인코딩한 벡터들을 가지고 그 벡터들을 일일히 토큰(문자(char),단어(word) 등)으로 바꾸고 특정한 규칙에 의해 최종적으로 정답 Sentence를 만들어내는 것입니다.

![Wang_1](/assets/images/rnnt/Wang_1.png)
*Fig. CTC 알고리즘에서 정답 label sequence가 'cat'일 경우 가능한 모든 path들은 위와 같다. (총 7개 이며 정답 길이가 길수록 기하급수적으로 늘어난다.)*

```
디코딩 된 모든 토큰들의 예시 1 : "c - a a - t"
디코딩 된 모든 토큰들의 예시 2 : "c - a - t t -"

최종 출력 : "c a t"
```
`CTC Decoding Example`, 위에는 하나의 디코딩 예시만 들었지만 사실 위와 같은 규칙으로 최종 출력을 만들 수 있는 경우의 수는 많으며, CTC는 이러한 가능한 모든 alignment를 `잠재 변수(latent varialbe)`로 생각하여 학습하는 모델입니다.

![Wang_2](/assets/images/rnnt/Wang_2.png)
*Fig. Path examples in CTC.*



CTC를 사용한 모델은 요약하자면 아래와 같습니다.

- 인코더가 뱉은 각각의 최종 벡터들은 조건부 독립이라고 가정 (HMM과 비슷)하고 이들을 특수한 토큰 <Blank> 를 포함해 쭉 디코딩(예측)한다.
- 입력 X 와 출력 Y 사이의 Alignment를 다이나믹 프로그래밍을 사용해 효율적으로 찾아낸다.
- **<span style="color:#e01f1f">1번에서 말한 것 처럼 조건부 독립을 가정하기 때문에 만들어진 문장이 자연스럽지 않다.</span>** 즉, 이전과 이후에 어떤 토큰이 만들어 신경쓰지 않기 때문에 발음 그대로 만들어지는 경우도 많음 ex) `I ate food` 가 아니라 `I eight food` 가 만들어지기도 하는데, 이는 언어 모델 (Language Model, LM) 을 따로 사용하는 방법으로 해결 가능하긴 함. 
- 입력 길이가 출력 길이보다 길어야 합니다. 이는 음성 인식에서는 문제가 없어 보이지만, CNN layer 나 pooling layer 등 모델의 추론 시간을 단축시켜주는 강력한 모듈들을 사용하는데 제약을 줍니다.





### <mark style='background-color: #dcffe4'> Attention-based model (2014, ...) </mark>

Seq2Seq ASR 모델은 자연어 처리(NLP) 분야에서 제안된 기계 번역 (Neural Machine Translation, NMT)와 유사한 모델로 입력 시퀀스를 인코더를 통해 Hidden Reperesentation Vector들로 나타낸 뒤
이들을 바탕으로 디코더에서 토큰을 하나씩 디코딩 하고, 그렇게 만들어진 토큰들을 다음 디코딩 할 때 정보로 주어 또 디코딩을 하고 ... `디코딩이 끝났다`는 의미의 <EOS> 토큰을 뱉을 때 까지 계속 디코딩을 하는 
Autoregressive 디코딩을 하는 모델입니다. 
 
여기에 '과연 각 토큰들을 디코딩 할 때 인코더가 출력한 정보(벡터)들 중 어떠한 정보를 참조해서 디코딩 해야 할 까?' 라는 의문을 해결하여 Seq2Seq 성능을 대폭 증가시킨 Attention Mechanism을 추가한 것이 Attention 기반 Seq2Seq 모델이 되는 것입니다. 즉 Attention Mechanism이 각 토큰과 입력 음성을 어떻게 Align해야 하는지를 CTC와는 다른 방식으로 해결했다고 볼 수 있습니다.

![attention](/assets/images/rnnt/shinji3.png)
*Fig. 일반적인 Attention 기반 Seq2Seq Model, 인코딩 된 벡터들(Memory, Representation Vectors)의 정보를 이용하여(Attention Mechanism) 토큰을 하나씩 생성해 냅니다. CTC와 다르게 토큰을 하나씩 만들 때 이전까지 만들어진 토큰 정보를 음성 정보와 같이 이용합니다(Conditional).*

![attention_operation](/assets/images/rnnt/attention.png){: width="60%"}
*Fig. Attention Mechanism을 사용한 Seq2Seq의 이해 : `입력된 음성들 중 어느 부분에 집중해서 이번 토큰을 만들어내야 하는가?` 를 반영한 context vector와 이전까지 만들어진 토큰 정보를 이용해 최종적으로 토큰을 만들어냄*

Attention 기반 기법도 몇가지 특징이 있는데요, 이는 아래와 같습니다.

- Encoder가 전통적인 ASR모델의 Acoustic Model 중 DNN 파트를 담당하며, Decoder가 Language Model을, Attention이 HMM 파트를 담당한다고 볼 수 있다. (해석적?)
- 토큰을 출력할 때 CTC와 다르게 `조건부(Conditional)`로 이전 토큰들을 입력으로 주기 때문에 더욱 정확하고 말이 되는 문장을 출력할 수 있다. (추가적인 LM 없이)
- 하지만 어텐션 모델은 CTC와 다르게 Monotonic한 Alignment를 생성해야 한다는 제한이 없기 때문에 다양한 Alignment를 만들어 낼 수 있고, 이는 학습을 어렵게 한다.
- **<span style="color:#e01f1f">전체 음성에 대해서 어텐션을 수행하기 때문에 Straming(온라인, 실시간) 모델에 적합하지 않다. </span>**
- 어텐션을 계산하는 데 시간이 많이 소요되며, 조건부로 토큰을 받아 생성하는 `Autoregressive Decoding` 또한 시간을 많이 잡아먹는다.  


물론 어텐션을 사용한 모델이 음성인식이 요구하는 단조로운(Monotonic)한 alignment를 만들어내지 못하는 것은 아닙니다. 다만 초기 학습에 어렵다는 것이죠.

![alignment](/assets/images/rnnt/alignment.png){: width="80%"}
*Fig. 잘 학습된 Seq2Seq with Attention Model의 단조로운 (Monotonic) Alignment*



위에서 언급한 네번 문제를 해결하기 위해서 CTC와 Attention을 결합한 기법이 제안되기도 했습니다. 
모델은 아래와 같고, 이렇게 함으로써 CTC Loss가 학습 초기 Monotonic Alignment를 배우게끔 하여 더욱 전체 모델을 잘 학습할 수 있게 합니다. 
(추가적으로 두가지 모델을 결합한 형태이기 때문에 앙상블(Ensemble)한 효과를 간접적으로 누림으로써 성능을 올려줍니다.)

![hybrid](/assets/images/rnnt/shinji4.png){: width="85%"}

![hybrid2](/assets/images/rnnt/shinji5.png)
*Fig. CTC 기법과 Attention 기법을 합쳐 loss를 구성한 Hybrid 모델. 이는 Attetnion과 CTC만을 가지고 구성된 단일 모델들의 단점을 상호 보완한다.*

(이렇게 결합하는 방법 외에도 어텐션의 alignment를 monotonic하게 제안하기 위해서 [MoChA(Monotonic Chunk wise Attention)](https://arxiv.org/pdf/1712.05382) 이라는 논문이 제안되거나 가우시안을 사용한 어텐션 등이 제안되기도 하였습니다.)



## <mark style='background-color: #fff5b1'> Transducer-based model </mark>


### <mark style='background-color: #dcffe4'> RNN-Transducer(RNN-T) (2012, ...) </mark>

자 이제, 일반적인 딥러닝 기반 E2E ASR모델 기법들 중 두 가지를 간단하게 알아봤고 Transducer에 대해서 알아보도록 하겠습니다.


Tranduscer는 위에서 언급한 CTC의 문제점 중 출력 길이가 입력 길이보다 작아야 한다는 점과, 출력 토큰들의 조건부 독립 가정을 해결해 성능을 끌어올렸는데요,
수식으로 CTC와 Transducer를 먼저 생각해보도록 하겠습니다.


![ctc_rnnt_attention](/assets/images/rnnt/ctc_rnnt_attention.png)
*Fig. CTC-based Model, Seq2Seq with Attention Model and Transducer-based Model*

(이미지 출처 : [Streaming End-to-end Speech Recognition For Mobile Devices](https://arxiv.org/pdf/1811.06621))




#### 1. CTC

우선 `notation`에 대해서 확실히 하겠습니다.

- $$x=(x_1, \cdots, x_T)$$ 는 input acoustic frames 입니다. 음향 벡터들이죠. 각 벡터들은 $$x_t \in \mathbb{R}^d$$ 의 d가 80차원이며 (log-mel filterbank 사용) $$T$$는 시퀀스 길이를 나타냅니다.  
- 특수한 토큰 (special token) 으로 $$y_0 = <sos>$$ 가 있으며, $$blank$$를 나타내는 토큰은 $$<b>$$라고  합니다. 
- $$ Z $$ : Vocabulary, 이며 토큰 하나가 가질 수 있는 차원의 수 입니다. (A, B, C ... Z) 
- $$ Z' $$ : $$\{ Z \cup <b> \}$$, 원래의 Vocabulary 에 특수한 토큰들을 추가한 것입니다. (음... 제가 참고한 논문이 많아서 조금 혼동이 있겠으나 우선은 $$blank$$ 토큰만 추가한 것을 $$Z'$$라고 .)
- $$ U $$는 정답이 될 문장의 길이를 나타내고, $$y=(y_1,\cdots,y_U)$$ 는 각 frame x를 디코딩 해 만들어낸 독립적인 출력 토큰 벡터들 입니다., 여기서 $$y_u \in Z'$$ 입니다.
- 즉 $$y_t$$ 는 $$t$$번째 time-step의 출력 벡터이고, 이 벡터는 $$ y_t = (y_t^1, \cdots, y_t^{\vert Z+1 \vert}) $$ 으로 각 토큰이 될 확률들을 포함하고 있습니다. (예를 들어, $$y_t^1$$ 는 'a'일 확률, $$y_t^2$$ 는 'b'일 확률 ... $$y_t^{\vert V+1 \vert}$$ 는 $$<b>$$ 일 확률 입니다.)
 


notation이 위와 같을 때 CTC 수식은 아래와 같습니다.


$$ 
P(y|x) = \sum_{\hat{y} \in A_{CTC}(x,y)} \prod_{i=1}^{T} P(\hat{y_t} \vert x_1, \cdots  ,x_t) 
$$


where $$ \hat{y} = ( \hat{y_1}, \cdots, \hat{y_T} ) \in A_{CTC}(x,y) \subset { \{ Z \cup <b> \} }^T  $$


($$A_{CTC}(x,y)$$는 수식에서 말한 것과 같이 $$x,y$$간 가능한 alignment를 모두 포함하는 set입니다.)


수식을 조금 더 살펴보자면, 위의 수식에서 $$\sum$$ 수식 안의 $$\prod$$는 아래와 같은 의미를 가지고 있습니다. 

$$ 
p(\pi \vert X) = \prod_{t=1}^{T} y_t^{\pi_t}, \forall \pi \in Z'^{T}
$$

where $$ Z'^{T} $$ denote the collection of all sequences of length T that defined on the Vocabulary $$Z'$$.


즉 $$ p(\pi \vert X) $$ 는 음성을 입력받아 정답 시퀀스의 모든 가능한 path중 하나에 대한 확률 분포(probability distribution conditioned only speech input, $$X$$)이며, 

(여기서 output sequence의 길이 speech input sequence 의 길이 $$T$$와 같습니다. (다만 최종 인퍼런스 할 때 중복 레이블을 제거하는 등의 정해진 규칙을 따라 최종 출력 문장을 만들어 냅니다.)

![Wang_1](/assets/images/rnnt/Wang_1.png)

![Wang_2](/assets/images/rnnt/Wang_2.png)
*Fig. Path examples in CTC. 즉 위에서 말한 $$\pi$$는 예를 들자면, 'c-aat' 인 것.*

진짜 정답 $$L$$의 가능한 모든 path $$\pi \in B^{-1} (L)$$ 들의 $$ p(\pi \vert X) $$  확률 분포값을 모두 더한 것이 최종적으로 우리가 원하는 Likelihood가 되며 

$$ 
p(L|X) = \sum_{\pi \in B^{-1}(L)} p(\pi \vert X)
$$

우리의 목적은 이를 최대화 하는 `Maximum Likelihood Estimation` 문제를 푸는 것 입니다. (혹은 우리는 위의 수식에 negative log likelihood (nll)을 취한 후 이를 최소화 하는 이른 바 `CTC Loss`를 최소화 하는 방향으로 파라메터를 업데이트 합니다.) 


($$L$$의 확률을 계산 해 내는 것은 미분 가능 (differentiable) 하기 때문에 정답 레이블 $$L$$에 대한 확률을 구해내기만 하면 오차 역전파 (Error Backpropagation) 알고리즘으로 쉽게 학습이 가능합니다. 하지만 $$p(\pi \vert X)$$ 하나를 계산하는 것들은 쉽기 쉬워도, 이를 가능한 모든 path에 대해서 계산 해 더하는 것은 어렵기에 `Forward-Backward Algorithm` 과 `Dynamic Programming (DP)`를 사용해 효율적으로 CTC Loss를 계산해내야 합니다. ( 생략)) 


하지만 위의 수식에서 보시면 아시겠지만 (또한 글의 서두에서 이야기 했듯) CTC는 앞서 말한 것 처럼 매 토큰을 디코딩하는데 있어, 입력 음성 (acoustic input sequence)에 대한 정보만을 사용하는, 즉 각 출력 토큰들에 대해서는 독립인 (independent) 이른 바 `Acoustic-Only model` 라고 할 수 있습니다.





#### 2. Transducer

반면 Transducer의 수식은 아래와 같습니다.


$$ 
P(y|x) = \sum_{ \hat{y} \in A_{RNNT}(x,y) } \prod_{i=1}^{T+U} P( \hat{y_i} \vert x_1, \cdots, x_{t_i}, y_0, \cdots, y_{u_{i-1}} ) 
$$


where $$ \hat{y} = ( \hat{y}, \cdots, \hat{y_{T+U}} ) \in A_{RNNT}(x,y) \subset { \{ Z \cup <b>\} }^{T+U} $$


수식적으로 보기에도 CTC와 Transducer는 크게 다르지 않은데요, 그도 그럴것이 이들의 목적은 같습니다.

- 1. 두 알고리즘 모두 음성인식의 'forced segmentation alignment problem' 문제를 풀기 위한 loss function이다.
- 2. 두 알고리즘 모두 $$blank$$ 라는 특수한 토큰을 도입했다.
- 3. 두 알고리즘 모두 정답 label sequecne 대해 가능한 모든 alignment에 대한 (all possible path) 확률을 구하고, 이들을 합쳐서 정답 label sequence에 대한 최종 확률을 구해낸다.

하지만 이들이 가지고 있는 결정적인 차이점이 있습니다.

- 1. path를 만들어 내는 방식(process)가 다르다.
- 2. 그렇게 만들어진 path에 대한 확률을 계산하는 방식(path probability calculation methods)이 완전히 다르다.

입니다.


이 두가지 중 아래에 2번은 수식에서도 알 수 있는데요, CTC에서는 모든 생성되는 토큰들이 $$t=1$$부터 $$T$$까지 조건부 독립을 가정하고 만들어졌다면, Transducer는 어떠한 $$i$$번째 토큰을 만들어내는 데 음성과 이전까지 만들어진 토큰들을 조건부로 주어 디코딩하게 됩니다. 

Transducer는 `Prediction Network (Language Model)`을 따로 하나 더 두고 이것이 뱉은 벡터들과, 음성 인코더에서 뱉은 음성 벡터들을 종합하여 `Joint Network`가 최종 디코딩을 하게 된다는 것입니다. 즉 Acoustic 과 Language model을 jointly 학습하는 것입니다.


(두 개의 분리된 네트워크(Prediction Network, Joint network)가 존재 하고, 특히 Prediction Network는 텍스트만을 이용해서 따로 RNN-LM 학습하듯 학습 해 사용할 수 있다고 합니다.)


다시 그림을 볼까요? Transducer와 CTC를 일반적으로 아래처럼 비교하여 나타내곤 하는데,

![ctc_rnnt_attention](/assets/images/rnnt/ctc_rnnt_attention.png)
*Fig. CTC-based Model, Seq2Seq with Attention Model and Transducer-based Model*

(이미지 출처 : [Streaming End-to-end Speech Recognition For Mobile Devices](https://arxiv.org/pdf/1811.06621))

조금 와닿지 않는 것 같아서 아래의 그림을 사용하도록 하겠습니다.


![ctc_rnnt_attention2](/assets/images/rnnt/asr.png)
*Fig. Transducer Model은 CTC와 다르게 Joint Network와 Prediction Network가 존재한다.*

(이미지 출처 : [Sequence-to-sequence learning with Transducers from Loren Lugosch](https://lorenlugosch.github.io/posts/2020/11/transducer/))


위의 그림을 이용해서 조금 더 Transducer에 대해서 얘기해 보도록 하겠습니다. 일단 Transducer가 RNN-Transducer (RNN-T) 라고 하겠습니다.

RNN-T는 3개의 Sub Networks로 이루어져있는데, 이는 각각 

- Transcription Network : $$F(x)$$
- Prediction Network : $$P(y,g)$$
- Joint Network : $$J(f,g)$$

입니다. 

![lugosch_rnnt3](/assets/images/rnnt/lugosch_rnnt3.png)
*Fig. RNN-T Network 는 3가지의 Sub Networks를 가지고 있다.*


각 네트워크를 조금 더 디테일하게 살펴보자면,


`Transcription Network` ($$F(x)$$) : 이 네트워크는 인코더 네트워크고, 음성 입력 벡터(speech input vector)들을 특징 벡터 (feature vector)로 변환해주는 `Acoustic Model` 입니다. 길이 $$T$$의 $$X=\{x_1,\cdots,x_T\}$$를 받아 $$F=\{f_1,\cdots,f_T\}$$로 매핑해줍니다. t번째 벡터 $$f_t$$ 는 Vocabulary의 크기에 $$+1$$을 한 $$\vert V \vert +1$$ 차원의 벡터 입니다. 





`Prediction Network` ($$P(y,g)$$) : Language Model 역할을 하는, Decoder의 한 part 입니다. RNN 네트워크이며, 이 네트워크는 output label sequence 내의 interdependencies 를 모델링 합니다 (반대로 Transdcription Network는 acoustic input간 dependencies를 모델링 합니다). $$P(l)$$은 maintains a hidden state $$hidden_u$$ and an output value $$g_u$$ for any label location $$u \in [1, N]$$. 네트워크 내의 계산되는 과정(loop calculation process)은 아래와 같습니다.

$$
hidden_u = H(W_{ih} l_{u-1} + W_{hh} hidden_{u-1} + b_h)
$$

$$
g_u = W_{h0} hidden_u + b_0
$$

위의 수식이 의미하는 바는, output sequence $$l_{[1:u-1]}$$의 마지막 $$u-1$$ 번째 벡터를 $$l_u$$ 를 만들어 내는 데 사용한다는 겁니다.
즉 이전까지의 정보를 내포하고 있는 히든 벡터를 사용하는 거죠. ($$hidden_u$$를 계산하는 수식에 $$l_{u-1}$$ 뿐만 아니라 $$h_{u-1}$$도 사용한다는 것을 알 수 있으며, 이는 여태까지의 문맥 정보를 내포하고 있는 벡터임)

간단하게 표햔해서 이를 $$g_u = P(l_{[1:u-1]})$$ 로 나타낼 수 있으며, $$g_u$$는 마찬가지로 $$\vert V \vert +1$$ 차원의 벡터입니다. 






`Joint Network` ($$J(f,g)$$) : input과 ouput 사이의 alignment job을 하는 네트워크 입니다. 예를 들어서 $$t \in [1,T], u \in [1,N]$$ 에 대해서, Joint Network는 Transcription Network의 출력인 $$f_t$$ 와 Prediction Network의 출력인 $$g_u$$를 사용해서 output location, $$u$$에서의 최종 출력을 계산 해 냅니다.

$$
e(k,t,u) = exp(f_t^k + g_u^k)
$$

$$
p(k \in V' \vert t,u) = \frac{ e(k,t,u) }{ \sum_{k' \in V'} e(k',t,u) }
$$

(수식이 조금 보기 그런 것 같은데, 결론은 매 time-step 에서 각 토큰들에 대해 합이 1인 softmax distribution 으로 나타내겠다는 것이며, $$t,u$$ 두 가지가 조건부로 걸려 있기 때문에, $$t=1$$ 일 때, $$u=1$$ 부터 $$u=N$$ 까지 , $$t=2$$ 일 때, $$u=1$$ 부터 $$u=N$$ 까지, ... , $$t=T$$ 일 때, $$u=1$$ 부터 $$u=N$$ 까지 모든 확률을 구해낼 수 있습니다.)


![lugosch_rnnt4](/assets/images/rnnt/lugosch_rnnt4.png)
*Fig. Joiner (Joint Network)는 두 개의 벡터를 더해서 토큰을 생성한다. (여러 자료를 섞어서만들었기 때문에 term이 다를 수 있습니다.) 여기서는 $$\phi$$가 공백 토큰입니다.*


- $$t=1$$, $$u=0$$, $$y={}$$(empty list) 부터 시작한다.
- $$f_t$$와 $$g_u$$를 계산한다.
- $$f_t$$와 $$g_u$$를 사용해서 $$h_{t,u}$$ 를 계산한다 (그림에선 $$h_{t,u}$$ 수식에선 $$p(k \in V' \vert t,u)$$.
- 만약 $$h_{t,u}$$ 가 공백이 아니고 의미 있는 label 이라면 $$u = u + 1$$로 한 스텝 전진하고, label을 y에 append하고, 다시 predictor에 이를 넣는다. 만약 그게 아니라 공백 토큰이라면, 아무것도 y에 append 하지 않으며 (output하지 않으며) time step, $$t$$만 $$t=t+1$$로 한 스텝 전진한다.
- 만약 $$t=T+1$$이면 루프를 탈출하고, 아니면 2번으로 가서 반복한다.



![lugosch_rnnt4](/assets/images/rnnt/lugosch_rnnt4.png)
*Fig. Joiner (Joint Network)는 두 개의 벡터를 더해서 토큰을 생성한다. (여러 자료를 섞어서만들었기 때문에 term이 다를 수 있습니다.) 여기서는 $$\phi$$가 공백 토큰입니다.*


 


수식에서 볼 수 있듯, $$p(k \in V' \vert t,u)$$는 $$f_t$$와 $$g_u$$의 함수이며, $$f_t$$는 $$x_t$$로 부터 나오며, $$g_u$$는 sequence $$\{l_1, \cdots, l_{u-1}\}$$로 부터 나오기 때문에, Joint Network 는 다음과 같은 역할을 한다고 볼 수 있습니다 :


주어진 historical output sequence $$\{l_1,\cdots,l_{u-1}\}$$과 t번째 입력 $$x_t$$를 사용해서, u 번째 output location에 대한 label distribution $$P( l_u \vert \{l_1,\cdots,l_{u-1}\},x_t )$$를 계산합니다.
그리고 이것은 decoding process를 위한 확률 분포 정보 (probability distribution information)을 제공합니다.






![rnnt_lattice](/assets/images/rnnt/rnnt.png){: width="70%"}
*Fig. Lattice of Paths in RNNT*


![ctc_rnnt_lattice](/assets/images/rnnt/deepvoice3_alignment.png)
*Fig. Lattice of Paths in CTC vs RNNT*






RNN-Transducer가 디코딩 하는 방법은 간단하게 말해서 다음과 같습니다.

- $$t$$번째 input $$x_t$$ 를 읽을 때 마다, 모델은 empty label, "-"를 뱉기 전 까지 계속해서 label을 생성해 냅니다.
- 만약 empty label, "-"을 만나게 되면, RNN-T는 다음 input $$x_{t+1}$$ 를 사용해서 위의 프로세스를 모든 input sequence 벡터들을 읽을 때 까지 반복합니다.
- $$x_t$$ 가 만들어내는 아웃풋 subsequence가 최소 1개 이상일 것이기 때문에, CTC가 가지는 문제 중 하나인, '출력 시퀀스가 입력 시퀀스 보다 짧아야 잘 작용한다'를 해결할 수 있습니다.





그리고 RNN-T를 사용함으로써 우리가 얻을 수 있는 이득은 다음과 같습니다.

- Since one input data can generate a label sequence of arbitrary length, theoretically, the RNN-transducer can map input sequence to an output sequence of arbitrary length, whether it is longer or shorter than the input.
- Since the prediction network is an RNN structure, each state update is based on previous state and output labels. Therefore, the RNN-transducer can model the interdependence within output sequence, that is, it can learn the language model knowledge.
- Since Joint Network uses both language model and acoustic model output to calculate probability distribution, RNN-Transducer models the interdependence between input sequence and output sequence, achieving joint training of language model and the acoustic model.





RNN-T 알고리즘은 CTC를 개선시켰음에도, 몇 가지 단점이 존재하는데, 우선 첫 번째는 학습 시키기 쉽지 않다는 것 입니다. 그러므로 보통 각 Sub Networks를 사전 학습 (pre-training) 시켜야 합니다. 그 다음으로는, RNN-T의 계산 과정에 포함되어있는 unreasonable paths 인데요, 예를 들어 첫 번째 입력 $$x_t$$가 정답 레이블 sequence에 대한 토큰을 모두 출력해버리면 나머지 입력들이 모두 공백 토큰을 뽑아내게 되고 이는 알고리즘 상에는 문제가 없지만, 분명히 말이 안되는 거죠. 


이러한 문제점들은 분명 다른 논문들에서 개선이 되긴 했으나, 여전히 RNN-T가 모든 음성인식 task에서 만능인 것은 아닙니다. 

![Wang_3](/assets/images/rnnt/Wang_3.png)
*Fig. 각 ASR 알고리즘들의 특징(장단점)*






### <mark style='background-color: #dcffe4'> Neural Transducer (2015) </mark>

Alex Glaves에 의해 제안된 제안된 `RNN-Tranducer` 이후 Google에서 제안한 음성인식 논문 중 `Neural Transducer`(An Online Sequence-to-Sequence Model Using Partial Conditioning) 라는 논문이 있었습니다. 논문의 요지는 Seq2Seq 논문을 개선시켰다는 것인데요. 짧게 요약하자면, 'Attention을 사용한 Seq2Seq 모델은 전체 음성을 한번에 받아들이고 나서야 문장 추론을 시작하기 때문에, 연산을 하는데 시간이 오래걸리며 전체 음성을 다 봐야 한다는 단점이 있기 때문에 실시간 음성인식에 적합하지 않기 때문에 RNN-Transducer와 비슷 하지만, 두 개의 독립된 연산을 하는 네트워크를(Prediction Network, Joint Network) 가정하지 않고 하나의 모듈을 사용한다는 점과 어텐션 매커니즘을 Block 단위로(partially observed speech input, partially generated text output sequence) 적용한다는 점에서 차이가 있는 새로운 Transdcuer 모델을 제안한다' 입니다. 

![all_seq2seq](/assets/images/rnnt/all_seq2seq.png)
*Fig. 왼쪽부터 차례대로 CTC-based Model, Seq2Seq with Attention Model, RNN-T Model, RNN-T with Attention Model *


![neural_transducer](/assets/images/rnnt/neural_transducer.png)
*Fig. Attention-based Seq2Seq Model vs Neural Transducer, 본 논문에서는 CTC와의 비교 보다는 CTC이후 Seq2Seq 태스크에서 훨씬 성공적으로 평가받았던 Attention기반 Seq2Seq 모델과 제안하는 모델을 비교했다. Seq2Seq 모델(좌)을 보면 입력을 다 받고난 후에야 추론할 수 있음을  볼 수 있다. Neural Transducer(우) 모델을 잘 보면, 입력 음성 전체가 아닌 특정 단위(Blcok)에 대해서만 이전의 Transducer가 전달한 Hidden State과 함께 입력으로 사용해 토큰들을 예측하고 또 이를 다음 블럭 예측할 때의 Transducer에 전달한다.*

모델은 온라인 음성인식을 위해서 계속해서 `블럭 단위의 음성 벡터들(Blocks of speech inputs)`을 끊임없이 받고, 한번 블럭 단위의 음성을 받으면, `단어 덩어리(chunks of outputs)`를 뱉습니다. 
출력된 토큰은 때로는 아무것도 없을 수도 있으며(받은 음성이 뭣도 아니라고 판단해서), 토큰을 생성하는 것은 Seq2Seq 모델과 같은 방식으로 수행되지만, 블럭 단위에 대해서만 Attention을 진행하기 때문에 앞서 말한 것 처럼 전체 음성에 대해 Attention을 진행할 필요가 없다는 점에서 차별점이 있습니다.


또 다른 차이점이 있는데, 논문에서는 네트워크를 학습할 때 과연 '각 블럭마다의 정답을 어떻게 할당할 것인가?' 즉, 다시 말해서 '어떻게 입력과 출력의 alignment를 설정할 것인가?'에 대해서 이야기 합니다.
가장 일반적인 두 가지 Approach가 있는데 하나는 Alignment를 잠재 변수(latent variable)로 두고 계산 하는 것이며, 다른 하나는 다른 알고리즘을 통해서 Alignment를 만들고 이를 maximize하는 방향으로 학습하는 것입니다. `CTC`의 경우는 전자의 방법을 사용하는데, `Neural Transducer`는 음성 입력뿐만 아니라 출력, 즉 alignment에 대해서도 조건(condition)이 걸리기 때문에 사용할 수 없으며, 이를 해결해 학습하기 위해서 다이나믹 프로그래밍(Dynamic Programming)을 통해서 `Approximate Best Alignments`를 계산해 낼 수 있는지를 보여줍니다.



![neural_transducer2](/assets/images/rnnt/neural_transducer2.png){: width="80%"}
*Fig. Neural Transducer의 디테일한 다이어그램. 빨간 박스 부분의 음성에 대해서만 인코딩을 진행해 hidden vectors를 뽑고 이에 대해 Transducer가 최종적으로 토큰들을 출력한다. 여기서 Transducer는 Attention을 사용한 Seq2Seq와 같은 역할을 수행한다.*


#### 1. Model and Notation
  - $$x_{1,\cdots,L}$$ : 길이 $$L$$의 입력 음성 벡터들 (즉 매트릭스)
  - $$x_i$$ : $$i$$ 번째 featrue vector
  - $$W$$ : block size
  - $$\frac{L}{W}$$ : the number of blocks
  - $$ \tilde{y_{1,\cdots,S}} $$ : 정답(target) 시퀀스 (길이 $$S$$)
  - $$ \tilde{y_{i,\cdots,(i+k)}}$$ : Transducer가 매 블럭마다 예측하는 시퀀스, 즉 $$k$$개의 token을 생성함. 하지만 $$0 \leq k \leq M$$ 인데, 이 말은 즉 k가 0일 수도(이번 block에서는 아무 토큰도 안 나올 수도) 있다는 것을 의미함.
  - $$ <e> $$ symbol : 매 Transducer가 예측하는 시퀀스를 감싸는 symbol임 (vocab에 있음). 이 심볼은 트랜스듀서가 디코딩을 시작할지, 아니면 다음 블럭으로 넘어갈지를 나타냄. 만약 트랜스듀서가 생성한 토큰이 0개라면 이 심볼은 CTC의 $$<blank>$$와 유사한 역할을 함(is akin to).
  - $$ Y $$ : (음성 $$\rightarrow$$ 정답) 간 가능한 모든 경우의 수의 집합 (set of all alignments of output sequence)을 의미하며, 위에서 언급한 $$ \tilde{y_{1,\cdots,S}} $$ 는 이러한 다양한 alignment로 부터 산출될(transduced) 수 있음.
  - $$ y_{1,\cdots,(S+B)} \in Y $$ 는 가능한 alignment중 어떤 것도 가능한데, 여기서 $$y$$가 $$\tilde{y}$$보다 $$B$$ 만큼 큰(긴) 이유는 앞서 말한 것 처럼 블럭마다의 레이블에 $$<e>$$ 토큰이 꼈기 때문임


늘 그렇듯 머신러닝에서 우리가 원하는 것은 Likelihood를 최대화 하는 방향으로 네트워크 파라메터를 학습 하는 것이기 때문에, $$p(\tilde{y_{1,\cdots,S}} \vert X_{1,\cdots,L})$$ 와  $$p(y_{1,\cdots,(S+B)} \vert X_{1,\cdots,L})$$ 를 계산하는 방법에 대해서 알아보도록 할 것입니다.

먼저 output seqeuence $$y_{1,\cdots,e_b}$$ 


$$ 
p(y_{1,\cdots,e_b} \vert x_{1,\cdots,bW}) = p(y_{1,\cdots,e_1} \vert x_{1,\cdots,W}) \prod_{b'=2}^{b} p( y_{(e_{b'-1}+1),\cdots,e'} \vert x_{1,\cdots,b'W}, y_{1,\cdots,e_{b'-1}} ) 
$$



$$ 
p(y_{(e_{b-1}+1),\cdots,e_b} \vert x_{1,\cdots,bW}, y_{1,\cdots,e_{b-1}}) = \prod_{m=e_{b-1}+1}^{e_b} p(y_m \vert x_{1,bW}, y_{1,\cdots,(m-1)})  
$$


#### 2. Next Step Prediction



$$ s_m = f_{RNN} ( s_{m-1}, [c_{m-1},y_{m-1} ; \theta ] ) $$
$$ c_m = f_{context} (s_m, h_{((b−1)W +1),\cdots,bW} ; \theta ) $$
$$ h'_{m} = f_{RNN} (h'_{m-1}, [c_m;s_m] ; \theta) $$
$$ p(y_m \vert x_{1,\cdots,bW},y{1,\cdots,(m-1)}) = f_{softmax}(y_m;h'_m,\theta) $$


#### 3. Computing $$f_{context}$$



$$e_j^m = f_{attention} (s_m,h_(b-1)W+j;\theta)$$
$$\alpha_m = softmax([e_1^m;e_2^m;\cdots;e_W^m])$$
$$c_m=\sum_{j=1}^W \alpha_j^m h_(b-1)W+j $$


#### 4. Addressing End of Blocks

asd

#### 5. Training



$$ 
p(\tilde{y_{1,\cdots,S}} \vert x_{1,\cdots,L}) = \sum_{y \in Y} p(y_{1,\cdots,(S+B)} \vert x_{1,\cdots,L} )
$$




$$ 
\frac{\partial}{\partial{\theta}} log p(\tilde{y_{1,\cdots,S}} \vert x_{1,\cdots,L} ) \sum_{y \in Y} p(y_{1,\cdots,(S+B)} \vert x_{1,\cdots,L}, \tilde{y_{1,\cdots,S}} ) \frac{\partial}{\partial{\theta}} log p(y_{1,\cdots,(S+B)} \vert x_{1,\cdots,L}) 
$$


#### 6. Inference




$$ 
\tilde{y_{1,\cdots,S}} = argmax_{y_{1,\cdots,S'}, e_{1,\cdots,N}} \sum_{b=1}^{N} log p( y_{e_{(b-1) +1 }, \cdots, e_b } \vert x_{1,\cdots,bW}, y_{1,\cdots,e_{(b-1)}}) 
$$


### <mark style='background-color: #dcffe4'> Two-Pass End-to-End Speech Recognition (2019) </mark>

[Two-Pass End-to-End Speech Recognition](https://arxiv.org/pdf/1908.10992)은 아래의 그림과 같이 Transducer Decoder가 예측한 일정 부분의 시퀀스와 음향 정보를 조건부(conditional)로 하여 Seq2Seq 모델인 LAS의 Decoder 를 이용하여 한번 더 디코딩 해주는 모델입니다. 

![twopass](/assets/images/rnnt/twopass.png){: width="30%"}
*Fig. Two-Pass Decoding을 하는 Transducer 모델.*

본 논문에서는 특이하게 학습시에 두 가지 로스를 사용합니다.

첫 번째 로스는 $$x,y^{\ast}$$ 가 각각 입력 음성과 정답 시퀀스를 의미할 때, RNNT와 Seq2Seq(LAS) Loss를 결합(interpolation)한 loss이고

$$
L_{combined}(x,y^{\ast}) = \lambda L_{RNNT}(x,y^{\ast}) + (1-\lambda) L_{LAS}(x,y^{\ast})
$$

다른 로스는 ~입니다.

$$
L_{MWER}(x,y^{\ast}) + \lambda_{MLE} log P(y^{ast} \vert x)
$$





### <mark style='background-color: #dcffe4'> Transformer Transducer (2020) </mark>

![transformer_transducer](/assets/images/rnnt/transformer_transducer.png)
*Fig. Transformer Transducer.*





## <mark style='background-color: #fff5b1'> References </mark>

- Blog
  - [Sequence-to-sequence learning with Transducers from Loren Lugosch](https://lorenlugosch.github.io/posts/2020/11/transducer/)
- Paper
  - [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/pdf/1211.3711)
  - [Speech Recognition with Deep Recurrent Neural Networks](https://arxiv.org/pdf/1303.5778)
  - [A Neural Transducer](https://arxiv.org/pdf/1511.04868)
  - [Exploring Neural Transducers for End-to-End Speech Recognition](https://arxiv.org/pdf/1707.07413)
  - [Streaming End-to-end Speech Recognition For Mobile Devices](https://arxiv.org/pdf/1811.06621)
  - [Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss](https://arxiv.org/pdf/2002.02562)
  - [Two-Pass End-to-End Speech Recognition](https://arxiv.org/pdf/1908.10992)
  - [A Comparison of Sequence-to-Sequence Models for Speech Recognition](http://www.isca-speech.org/archive/Interspeech_2017/pdfs/0233.PDF)
  - [End-to-End Attention-based Large Vocabulary Speech Recognition](https://arxiv.org/pdf/1508.04395)
  - [Listen, Attend and Spell](https://arxiv.org/pdf/1508.01211)
  - [An Overview of End-to-End Automatic Speech Recognition](https://www.mdpi.com/2073-8994/11/8/1018/pdf)
- Others
  - [End-to-End Speech Recognition by Following my Research History from Shinji Watanabe](https://deeplearning.cs.cmu.edu/F20/document/slides/shinji_watanabe_e2e_asr_bhiksha.pdf)
