---
title: Classification (1/2)
categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> Regression VS Classification </mark>

아래의 표에서 볼 수 있듯이, 간단하게 생각하면 


1.입력값이 continuous 한데 결과값이 마찬가지로 continuous하면 Regression 문제라 할 수 있고,


2.입력값이 continuous 한데 결과값이 discrete하면 Classification 문제라 할 수 있습니다.


![image](https://user-images.githubusercontent.com/48202736/105444745-c7717380-5cb1-11eb-92a0-b618ad4d6b4f.png)

- <mark style='background-color: #fff5b1'> Logistic Regression </mark>

로지스틱 회귀 정의 추가해야할듯

본론)

앞서 Regression 문제에서와 마찬가지로 우리는 연속적인 입력 변수에 대해서 대상 y의 분포를 모델링 할 수 있습니다. 선형 회귀에서는 y를 가우시안 분포로 정의하여 문제를 풀었습니다.

자 이번에는 y의 분포를 베르누이 확률 분포로 모델링 해보도록 하겠습니다.

그 전에 우선 베르누이 분포에 대해서 remind를 해보도록 하겠습니다. 

- <mark style='background-color: #dcffe4'> Bernoulli Distribution </mark>

베르누이 분포에 대해 쉽게 설명하기 위해 그림을 먼저 보도록 하겠습니다.

![image](https://user-images.githubusercontent.com/48202736/105621207-7fd12000-5e48-11eb-9106-42a0a58fb4a2.png)

베르누이 분포는 '$$x=0/x=1$$' 이나 '성공/실패' 등 두 가지 가능한 경우에 대한 상황을 나타냅니다.

수식으로 나타내면 아래와 같은데, $$x=1$$일 확률이 $$\lambda$$ 이고, 베르누이 분포는 두 가지 경우에 대해서만 생각하기 때문에 반대로 $$x=0$$ 이 될 확률은 $$1-\lambda$$가 됩니다.
(예를 들어, 어떤 x(이미지 픽셀값)가 $$x=0$$(강아지)일 확률이 $$labmda$$(0.64)면 $$x=1$$(고양이)일 확률은 $$1-\lambda$$(1-0.64=0.34)가 됩니다.)

![image](https://user-images.githubusercontent.com/48202736/105621213-86f82e00-5e48-11eb-8f27-74ec370737da.png)

위의 수식을 보면 베르누이 분포를 한번에 $$\labmda^{x}(1-\lambda)^{1-x}$$로 표현하는 걸 알 수 있습니다. 이는 x가 1이면 $$\lambda$$가 되고 x가 0이면 $$(1-\lambda)$$가 되는 수식입니다.
이러한 의미를 가지는 베르누이 분포는 아래와 같이 쓰기도 합니다.

![image](https://user-images.githubusercontent.com/48202736/105621209-8495d400-5e48-11eb-8ab7-2095f20068c6.png)

이 때 추정하고자 하는 파라메터는 성공 확률(편의상 이렇게 말하겠습니다. 경우에 따라 다르게 말할 수 있을 것 같습니다.), $$\lambda$$가 되겠죠? (가우시안 분포에서 평균,$$\mu$$와 분산,$$\sigma^2$$를 찾는게 목적이듯)

- <mark style='background-color: #dcffe4'> Categorical Distribution </mark>

+) (나중에 다루겠지만, 미리 얘기하고 넘어가겠습니다.) 베르누이 분포와 분포로 Categorical(범주형) 분포가 있습니다.

![image](https://user-images.githubusercontent.com/48202736/105621216-96777700-5e48-11eb-99c4-400cf91c6405.png)

![image](https://user-images.githubusercontent.com/48202736/105621218-9aa39480-5e48-11eb-88ed-ac911e2e2a76.png)

![image](https://user-images.githubusercontent.com/48202736/105621220-9c6d5800-5e48-11eb-8d87-bc128a488378.png)


다시 본론으로 돌아가서 

![image](https://user-images.githubusercontent.com/48202736/105444774-d3f5cc00-5cb1-11eb-93e4-f280a7328d92.png)

- <mark style='background-color: #dcffe4'> Maximum Likelihood </mark>

![image](https://user-images.githubusercontent.com/48202736/105038606-e3eb9100-5aa2-11eb-9b1d-070d4e6edd32.png)

- <mark style='background-color: #dcffe4'> Optimization </mark>

![image](https://user-images.githubusercontent.com/48202736/105444895-128b8680-5cb2-11eb-91a7-dac84df8707d.png)
![image](https://user-images.githubusercontent.com/48202736/105444900-14ede080-5cb2-11eb-9347-cb53fbfa5a5a.png)
![image](https://user-images.githubusercontent.com/48202736/105444909-18816780-5cb2-11eb-9c32-403825583254.png)

- <mark style='background-color: #fff5b1'> Gradient Based Optimization </mark>

- <mark style='background-color: #dcffe4'> Steepest Gradient Descent  </mark>

![image](https://user-images.githubusercontent.com/48202736/105444927-23d49300-5cb2-11eb-8336-1e13ccfd1901.png)
![image](https://user-images.githubusercontent.com/48202736/105444938-2b943780-5cb2-11eb-9947-df2dcf6cb9d4.png)
![image](https://user-images.githubusercontent.com/48202736/105444942-2f27be80-5cb2-11eb-858c-a23f594829c2.png)


- <mark style='background-color: #dcffe4'> Newton’s Method </mark>

![image](https://user-images.githubusercontent.com/48202736/105444950-3353dc00-5cb2-11eb-8f62-291d69813e45.png)
![image](https://user-images.githubusercontent.com/48202736/105444958-35b63600-5cb2-11eb-8cd4-939b22e844f2.png)
![image](https://user-images.githubusercontent.com/48202736/105444964-38b12680-5cb2-11eb-834d-b3c6475a09b7.png)

- <mark style='background-color: #dcffe4'> Optimization for Logistic Regression </mark>

![image](https://user-images.githubusercontent.com/48202736/105444986-41a1f800-5cb2-11eb-97fe-d287609a3a77.png)


![image](https://user-images.githubusercontent.com/48202736/105444997-48c90600-5cb2-11eb-8445-1b32c228bdc9.png)

- <mark style='background-color: #fff5b1'> Multiclass Logistic Regression </mark>

![image](https://user-images.githubusercontent.com/48202736/105445180-9e9dae00-5cb2-11eb-96cc-e8ac1453fee7.png)

- <mark style='background-color: #fff5b1'> References </mark>

1. [Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.](http://www.computervisionmodels.com/)
