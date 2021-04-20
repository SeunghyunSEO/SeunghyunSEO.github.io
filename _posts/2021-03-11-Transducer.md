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
처음 되었을 때와 마찬가지로 일반적인 경우 Recurrent Neural Networks (RNNs) 를 내부 모듈로 사용하기 때문에 RNN-Tranducer (RNN-T) 라고 부르곤 하지만, 최근에는 [Transformer-Transducer](https://arxiv.org/pdf/2002.02562) 가 제안되는 등 다양한 Variation이 존재합니다.

## <mark style='background-color: #fff5b1'> Common Approaches for Deep Learning based E2E ASR Model) </mark>
데이터와 적당한 모델만 있으면 되는 딥러닝 기반 End-to-End (E2E) 음성인식 (Automatic Speech Recognition, ASR) 모델과 달리 전통적인 ASR 모델은 음성에 대한 전문적인 Domain Knowledge와 딥러닝 기법과 비교해 상당히 복잡한 모델링이 요구되었습니다.

![traditional_asr](/assets/images/rnnt/shinji1.png.png)
*Fig. 딥러닝 기반 E2E ASR 모델이 제안되기 전의 음성인식 모델*

하지만 딥러닝기반 E2E기법들이 차례대로 등장하면서 CTC(2006~), Transducer(2012~), Attention-based Seq2Seq(2014~) 복잡한 모델링이 없이도 엄청난 음성 인식 성공률을 보여주며 음성인식의 인식 레벨은 한단계 업그레이드 되었습니다.

![e2e_asr](/assets/images/rnnt/asr.png)
*Fig. 일반적인 딥러닝 기반 E2E ASR 기법들, 왼쪽부터 차례대로 CTC, Tranducer, Attention 기반 기법들이다.*

이 중 CTC, ATtention 기반 기법과 다르게, Transducer는 상대적으로 적게 연구가 되었지만 최근 실시간 음성인식 (Straeming-ASR) 의 중요성등이 대두되면서 주목을 받아왔습니다.
Transducer는 앞서 말한 것 처럼 CTC의 업그레이드 버전인데, CTC 또한 최근 Attention기반 기법과 비교해 성능이 뒤쳐지지 않는다는 논문들이 많이 나오고 있기도 합니다.
그렇기에 최근에는 Transducer와 Attention 기반 Sequence-to-Sequence(Seq2Seq) 모델 (LAS나 트랜스포머 기반 기법)을 같이 사용하는 Two-Pass 기법이 제안되기도 해왔습니다. 


아무튼 우리는 이제 CTC와 Attention 기반 기법에 대해 간단하게 알아보고 Tranducer 모델들에 대해서 알아보도록 하겠습니다.




### <mark style='background-color: #dcffe4'> CTC-based model (2006, 2014, ...) </mark>

![ctc](/assets/images/rnnt/shinji2.png)
*Fig. CTC 기반 Model*




### <mark style='background-color: #dcffe4'> Attention-based model (2014, ...) </mark>

![attention](/assets/images/rnnt/shinji3.png)
*Fig. Attention 기반 Seq2Seq Model*


![hybrid](/assets/images/rnnt/shinji4.png)

![hybrid2](/assets/images/rnnt/shinji5.png)
*Fig. CTC 기법과 Attention 기법을 합쳐 loss를 구성한 Hybrid 모델. 이는 Attetnion과 CTC만을 가지고 구성된 단일 모델들의 단점을 상호 보완한다.*


### <mark style='background-color: #dcffe4'> Transducer-based model (2012, 2018, ...) </mark>

![rnnt_model](/assets/images/rnnt/asr.png)
*Fig. CTC-based Model vs Transducer-based Model*


![neural_transducer](/assets/images/rnnt/neural_transducer.png)
*Fig. neural transducer*

![neural_transducer2](/assets/images/rnnt/neural_transducer2.png)
*Fig. neural transducer*

## <mark style='background-color: #fff5b1'> References </mark>

- Blog
  - 1. [Sequence-to-sequence learning with Transducers from Loren Lugosch](https://lorenlugosch.github.io/posts/2020/11/transducer/)
- Paper
  - 1. [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/pdf/1211.3711)
  - 2. [A Neural Transducer](https://arxiv.org/pdf/1511.04868)
  - 3. [Exploring Neural Transducers for End-to-End Speech Recognition](https://arxiv.org/pdf/1707.07413)
  - 4. [Streaming End-to-end Speech Recognition For Mobile Devices](https://arxiv.org/pdf/1811.06621)
  - 5. [Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss](https://arxiv.org/pdf/2002.02562)
- Others
  - 1. [End-to-End Speech Recognition by Following my Research History from Shinji Watanabe](https://deeplearning.cs.cmu.edu/F20/document/slides/shinji_watanabe_e2e_asr_bhiksha.pdf)
