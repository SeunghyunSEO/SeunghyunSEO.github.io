---
title: (Paper) Improving Speech Recognition Using Consistent Predictions on Synthesized Speech
categories: Speech_Recognition
tag: [tmp]

toc: true
toc_sticky: true

comments: true
---


이번 글에서는 [Improving Speech Recognition Using Consistent Predictions on Synthesized Speech](https://ieeexplore.ieee.org/document/9053831) 라는 논문을 요약해서 리뷰해 보려고 합니다. 

---
< 목차 >
{: class="table-of-content"}
* TOC
{:toc}
---


## <mark style='background-color: #fff5b1'> Problem Definition and Contibution Point </mark>

본 논문에서 정의하는 Problem과 Contribution Point은 다음과 같습니다.

- 1.요즘 음성 합성 성능이 많이 올라왔기 때문에 이러한 합성음 데이터를 사용해서 음성인식 성능을 높히려는 노력들이 많이 있었다.
- 2.하지만 이렇게 합성음을 사용해서 data augmentation 효과를 누리는 방법은 효과적이지 못했다.
- 3.그렇기 때문에 우리는 합성음과 실제 음성이 유사한 형태를 이루도록 하는 `consistency loss`를 제안한다,
- 4.전체 960시간의 LibriSpeech 데이터셋을 사용해서 학습한 것과 비교해서 460시간 음성 + 500 합성음 (transcript 사용해서) 으로 학습한게 별로 차이가 안났다. (WER 0.2% 내외)
- 5.결론은 우리가 제안한 방법론을 바탕으로 무수히 많은 text만 있다면 오디오가 절반밖에 없는 상황에서도 좋은 성능을 낼 수 있을것이다. 

입니다.


## <mark style='background-color: #fff5b1'> Generating Consistent Predictions </mark>

### <mark style='background-color: #dcffe4'> Unsupervised Data Augmentation (UDA) Loss </mark>

`Unsupervised Data Augmentation (UDA) Loss` 이란 원래의 음성 $$x$$에 noise 등을 추가해서 $$\hat{x}$$을 만들고, 이를 기반으로 추론한 문장 $$y$의 분포를 유사하게 하는 겁니다.


$$
J_{UDA}(\theta) = \mathbb{E}_{x \in U} \mathbb{E}_{x^ \sim q(x^ \vert x)} D_{KL} (p_{\tilde{theta}} (y \vert x) \parallel p_{\theta} (y \vert \hat{x}) ) 
$$



### <mark style='background-color: #dcffe4'> Consistency Loss (Modified UDA loss) </mark>

$$
J_{cons}(\theta) = \mathbb{E}_{x \in U} \mathbb{E}_{x^ \sim q(x^ \vert y^{\ast},z)} D_{KL} (p_{\tilde{theta}} (y \vert x) \parallel p_{\theta} (y \vert \hat{x}) ) 
$$



### <mark style='background-color: #dcffe4'> Overall Training Objective </mark>

$$
{min}_{\theta} J(\theta) = \mathbb{E}_{x,y^{\ast} \in L} [p_{\theta}(y^{\ast} \vert x)] + \lambda J_{UDA} (\theta)
$$



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



$$
{min}_{\theta} J(\theta) = \lambda_r J_{real}(\theta) + \lambda_t J_{tts}(\theta) + \lambda_c J_{cons}(\theta)
$$
