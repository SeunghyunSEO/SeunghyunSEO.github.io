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

![asr](/assets/images/rnnt/asr.png)
*Fig. 일반적인 딥러닝 기반 End-to-End(E2E) 음성인식 (Automatic Speech Recognition, ASR) 모델들.*

### <mark style='background-color: #dcffe4'> CTC-based model (2006, 2014, ...) </mark>

### <mark style='background-color: #dcffe4'> Attention-based model (2014, ...) </mark>

### <mark style='background-color: #dcffe4'> Transducer-based model (2012, 2018, ...) </mark>


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
