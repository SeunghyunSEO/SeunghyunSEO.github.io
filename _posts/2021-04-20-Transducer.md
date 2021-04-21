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

[Transducer]((https://arxiv.org/pdf/1211.3711)) 모델이란 [Connectionist Temporal Classification (CTC)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) 을 제안한 Alex Graves 가 2012년에 처음 제안한 개념으로 CTC의 단점을 보완한 업그레이드 버전이라고 보면 됩니다.
일반적인 경우 Recurrent Neural Networks (RNNs) 를 내부 모듈로 사용하기 때문에 RNN-Tranducer (RNN-T) 라고 부르곤 하지만, 최근에는 [Transformer-Transducer](https://arxiv.org/pdf/2002.02562) 가 제안되는 등 다양한 Variation이 존재합니다.


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

음성인식같은 Seq2Seq 모델의 가장 큰 문제점은 Alignment가 맞지 않는다는 것입니다.
예를 들어 "안녕하세요, 저는 대학원생 입니다." 라고 녹음된 전화 음성이 있다고 해 보도록 하겠습니다.
이 음성의 길이가 2초일 경우 sampling rate가 8000이라면 16000차원인데, 이를 입력으로 사용해 디코딩 해야 할 정답 길이는 19차원(글자) 라는 미스 매치를 Miss-Alignment 문제라고 합니다.
이를 해결 하기 위해 2006년에 제안된 방법이 바로 Connectionist Temporal Classification (CTC) loss 입니다.

CTC는 복잡한 철학과 이론이 있지만, 짧게 요약하자면 아래와 그림으로 나타낼 수 있습니다.

![ctc](/assets/images/rnnt/shinji2.png)
*Fig. CTC 기반 Model*

모델이 하는 일은 입력을 인코더에 통과시켜 인코딩한 벡터들을 가지고 그 벡터들을 일일히 토큰(문자(char),단어(word) 등)으로 바꾸고 특정한 규칙에 의해 최종적으로 정답 Sentence를 만들어내는 것입니다.

```
디코딩 된 모든 토큰들 : "A A _ _ P P P _ P P _ L _ E"
최종 출력 : "A P P L E"
```

CTC를 사용한 모델은 요약하자면 아래와 같습니다.

- 인코더가 뱉은 각각의 최종 벡터들은 조건부 독립이라고 가정 (HMM과 비슷)하고 이들을 특수한 토큰 <Blank> 를 포함해 쭉 디코딩(예측)한다.
- 입력 X 와 출력 Y 사이의 Alignment를 다이나믹 프로그래밍을 사용해 효율적으로 찾아낸다.
- **<span style="color:#e01f1f">1번에서 말한 것 처럼 조건부 독립을 가정하기 때문에 만들어진 문장이 자연스럽지 않다.</span>** 즉, 이전과 이후에 어떤 토큰이 만들어 신경쓰지 않기 때문에 발음 그대로 만들어지는 경우도 많음 ex) "I ate food" 가 아니라 "I eight food" 가 만들어지기도 하는데, 이는 언어 모델 (Language Model, LM) 을 따로 사용하는 방법으로 해결 가능하긴 함. 
- 입력 길이가 출력 길이보다 길어야 합니다. 이는 음성 인식에서는 문제가 없어 보이지만, CNN layer 나 pooling layer 등 모델의 추론 시간을 단축시켜주는 강력한 모듈들을 사용하는데 제약을 줍니다.





### <mark style='background-color: #dcffe4'> Attention-based model (2014, ...) </mark>

Seq2Seq ASR 모델은 자연어 처리(NLP) 분야에서 제안된 기계 번역 (Neural Machine Translation, NMT)와 유사한 모델로 입력 시퀀스를 인코더를 통해 Hidden Reperesentation Vector들로 나타낸 뒤
이들을 바탕으로 디코더에서 토큰을 하나씩 디코딩 하고, 그렇게 만들어진 토큰들을 다음 디코딩 할 때 정보로 주어 또 디코딩을 하고 ... 디코딩이 끝났다는 <EOS> 토큰을 뱉을 때 까지 계속 디코딩을 하는 
Autoregressive 디코딩을 하는 모델입니다. 여기에 '과연 각 토큰들을 디코딩 할 때 인코더가 출력한 정보(벡터)들 중 어떠한 정보를 참조해서 디코딩 해야 할 까?' 라는 의문을 해결하여 Seq2Seq 성능을 대폭 증가시킨 Attention Mechanism을 추가한 것이 Attention 기반 Seq2Seq 모델이 되는 것입니다. 즉 Attention Mechanism이 각 토큰과 입력 음성을 어떻게 Align해야 하는지를 CTC와는 다른 방식으로 해결했다고 볼 수 있습니다.

![attention](/assets/images/rnnt/shinji3.png)
*Fig. Attention 기반 Seq2Seq Model*

Attention 기반 기법도 몇가지 특징이 있는데요,

- Encoder가 전통적인 ASR모델의 Acoustic Model 중 DNN 파트를 담당하며, Decoder가 Language Model을, Attention이 HMM 파트를 담당한다고 볼 수 있다. (해석적?)
- 토큰을 출력할 때 CTC와 다르게 조건부로 이전 토큰들을 입력으로 주기 때문에 더욱 정확하고 말이 되는 문장을 출력할 수 있다. (추가적인 LM 없이)
- 하지만 어텐션 모델은 CTC와 다르게 Monotonic한 Alignment를 생성해야 한다는 제한이 없기 때문에 다양한 Alignment를 만들어 낼 수 있고, 이는 학습을 어렵게 한다.
- **<span style="color:#e01f1f">전체 음성에 대해서 어텐션을 수행하기 때문에 Straming(온라인, 실시간) 모델에 적합하지 않다. </span>**
- 어텐션을 계산하는 데 시간이 많이 소요되며, Autoregressive Decoding 또한 시간을 많이 잡아먹는다.  

입니다.


위에서 언급한 세번 문제를 해결하기 위해서 CTC와 Attention을 결합한 기법이 제안되기도 했습니다. 
모델은 아래와 같고, 이렇게 함으로써 CTC Loss가 학습 초기 Monotonic Alignment를 배우게끔 하여 더욱 전체 모델을 잘 학습할 수 있게 합니다. 
(추가적으로 두가지 모델을 결합한 형태이기 때문에 앙상블(Ensemble)한 효과를 간접적으로 누림으로써 성능을 올려줍니다.)

![hybrid](/assets/images/rnnt/shinji4.png)

![hybrid2](/assets/images/rnnt/shinji5.png)
*Fig. CTC 기법과 Attention 기법을 합쳐 loss를 구성한 Hybrid 모델. 이는 Attetnion과 CTC만을 가지고 구성된 단일 모델들의 단점을 상호 보완한다.*

(이렇게 결합하는 방법 외에도 어텐션의 alignment를 monotonic하게 제안하기 위해서 [MoChA(Monotonic Chunk wise Attention)](https://arxiv.org/pdf/1712.05382) 이라는 논문이 제안되거나 가우시안을 사용한 어텐션 등이 제안되기도 하였습니다.)



### <mark style='background-color: #dcffe4'> Transducer-based model (2012, 2013, 2018, ...) </mark>

자 이제, 일반적인 딥러닝 기반 E2E ASR모델 기법들 중 두 가지를 간단하게 알아봤고 Transducer에 대해서 알아보도록 하겠습니다.


Tranduscer는 위에서 언급한 CTC의 문제점 중 출력 길이가 입력 길이보다 작아야 한다는 점과, 출력 토큰들의 조건부 독립 가정을 해결해 성능을 끌어올렸는데요,
수식으로 CTC와 Transducer를 먼저 생각해보도록 하겠습니다.

우선 notation들에 대해서 확실히 하겠습니다.

- $$x=(x_1, \cdots, x_T)$$ 는 input acoustic frames 입니다. 음향 벡터들이죠. 각 벡터들은 $$x_t \in \mathbb{R}^d$$ 의 d가 80차원이며 (log-mel filterbank 사용) $$T$$는 시퀀스 길이를 나타냅니다.   
- $$ U $$는 정답이 될 문장의 길이를 나타내고, $$y=(y_1,\cdots,y_U)$$ 는 각 frame x를 디코딩 해 만들어낸 독립적인 출력 벡터들 이며, 여기서 $$y_u \in Z$$ 이며 Z는 문맥적으로 독립 (context-independent, CI)인 (phonemes, graphems or word-pieces)등을 나타냅니다.
- 특수한 토큰 (special token) 으로 $$y_0 = <sos>$$ 가 있으며, $$blank$$를 나타내는 토큰은 $$<b>$$라고  합니다.  

notation이 위와 같을 때 CTC 수식은 아래와 같습니다.

<center>
$$ P(y|x) = \sum_{\hat{y} \in A_{CTC}(x,y)} \prod_{i=1}^{T} P(\hat{y_t} \vert x_1, \cdots  ,x_t) $$
$$ where \space \hat{y} = (\hat{y_1}, \cdots, \hat{y_T}) \in A_{CTC}(x,y) \subset {\{ Z \cup <b> \}}^T $$ 
</center>

입력 음성과 정답 문장 간의 가능한 alignment들을 모두 생각하고 이를 모두 더한 확률을 구하는 것이죠.

하지만 CTC는 앞서 말한 것 처럼 매 토큰을 디코딩하는데 있어, 입력 음성 (acoustic input sequence)에 대한 정보만을 사용하는 이른 바 `acoustic-only model` 입니다.



반면 Transducer의 수식은 아래와 같습니다.

<center>
$$ P(y|x) = \sum_{ \hat{y} \in A_{RNNT}(x,y) } \prod_{i=1}^{T+U} P( \hat{y_i} \vert x_1, \cdots, x_{t_i}, y_0, \cdots, y_{u_{i-1}} ) $$
$$ where \space \hat{y} = (\hat{y}, \cdots, \hat{y_{T+U}}) \in A_{RNNT}(x,y) \subset {\{ Z \cup <b>\}}^{T+U} $$
</center>
  

CTC의 수식에서 모든 생성되는 토큰들이 $$t=1$$부터 $$T$$까지 조건부 독립을 가정하고 만들어졌다면, Transducer는 수식에서도 알 수 있듯이, $$i$$번째 토큰을 만들어내는 데 음성과 이전까지 만들어진 토큰들을 조건부로 주어 디코딩하게 됩니다. 

즉, Transducer는 박스로 나눠서 디코딩을 하는데, 예를 들어 $$i$$번째 박스를 디코딩 하는데 있어, 입력 음성 (acoustic inputs sequence)와 $$i-1$$ 번째 박스에서의 토큰들을 조건부로 주어 $$i$$번째 박스를 디코딩 하는, 즉 하나의 분리된 뉴럴 네트워크 모듈(RNN이라고 하겠습니다 우선)을 추가한 거죠. 즉 이는 `language model`을 따로 하나 더 두고 acoustic 과 language model을 jointly 학습하는 것이라고 볼 수 있습니다.  

Transducer와 CTC를 일반적으로 아래처럼 비교하여 나타내곤 하는데,

![ctc_vs_rnnt](/assets/images/rnnt/rnnt_model.png)
*Fig. CTC-based Model vs Transducer-based Model*

조금 와닿지 않는 것 같아서 아래의 그림을 사용하도록 하겠습니다.

![ctc_vs_rnnt2](/assets/images/rnnt/ctc_rnnt.png)
*Fig. Transducer Model은 CTC와 다르게 Joint Network와 Prediction Network가 존재한다.*

(이미지 출처 : [Sequence-to-sequence learning with Transducers from Loren Lugosch](https://lorenlugosch.github.io/posts/2020/11/transducer/))



아래는 Neural Transducer의 그림입니다.

![neural_transducer](/assets/images/rnnt/neural_transducer.png)
*Fig. neural transducer*

![neural_transducer2](/assets/images/rnnt/neural_transducer2.png)
*Fig. neural transducer*

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
- Others
  - [End-to-End Speech Recognition by Following my Research History from Shinji Watanabe](https://deeplearning.cs.cmu.edu/F20/document/slides/shinji_watanabe_e2e_asr_bhiksha.pdf)
