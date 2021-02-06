---
title: Classification (2/?) - Bayesian logistic regression
categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> Bayesian Logistic Regression </mark>

이번에는 Bayesian Logistic Regression에 대해 알아보도록 하겠습니다.

![cls1](https://user-images.githubusercontent.com/48202736/107110376-b2d4d400-688a-11eb-832b-4cbb5babc175.png)
*Fig. Logistic Regression (Classification)의 여러 변형*

우리가 흔히 머신러닝 방법론을 통해 분류, 회귀를 하는 방식은 크게 3가지가 있었습니다.

이전 글들에서 많이 설명을 했기 때문에 이번에는 간략하게 요약만 하고 넘어가도록 하겠습니다.

> 1. Maximum Likelihood (ML) : likelihood 를 정의하고 이를 최대화하는 단 하나의 값(점)을 찾는다. (점 추정) <br>
> 2. Maximum A Posteriori (MAP) : likelihood를 정의하고 추가로 파라메터에 대한 prior를 정의해서 베이즈룰을 통해 posterior를 구한 뒤 이를 최대화하는 단 하나의 값(점)을 찾는다. (점 추정) <br>
> 3. Bayesian Approach : 2번의 posterior 분포를 구하고 점 추정 하지 않고(분포 그대로), 추론 시 파라메터에 대해(posterior 분포를) 전부 적분해서 사용한다. <br>

분류 문제를 풀 경우에 우리는 일반적으로 ML,MAP 방식을 사용할 수 있지만 이는 어떤 부분에서는 문제가 있을 수 있습니다. 

예를들어 MAP 방식으로 Decision Boundary를 정하는 것은 아래와 같은 문제가 있을 수 있는데, 
MAP는 posterior 분포로부터 최대값을 나타내는 단 하나의 파라메터만을 주어진 학습 데이터를 통해 구해서 쓰는 것이기 때문에 Decision Boundary 근처에 어떤 데이터가 주어졌을 때 이를 class1이라고 과잉 확신 하는 경우가 있습니다. (예를 들어 class1 :0.9, class2:0.1) 

<img width="1182" alt="bayesian_cls" src="https://user-images.githubusercontent.com/48202736/107111166-0c400180-6891-11eb-93d1-6f6a16fba8b4.png">
*Fig. MAP(좌) vs Bayesian Approach(우), 이미지 출처 : [A Bayesian graph convolutional network for reliable prediction of molecular properties with uncertainty quantification](https://pubs.rsc.org/en/content/articlepdf/2019/sc/c9sc01992h)*

이는 Decision boundary가 아래의 그림같이 생겼기 때문인데요,

![cls5-1](https://user-images.githubusercontent.com/48202736/107110383-b8321e80-688a-11eb-9d60-901e301a7f81.png)
*Fig. Decision Boundary는 왼쪽과 같은 logistic 함수를 사용해서 만들기 때문입니다. decision boundary가 출력이 0.5인 부분이라고 하면 이 값을 전후로 굉장히 높은 확률로 class를 확신해서 분류하게 됩니다. *

하지만 우리는 위의 
그렇기 때문에 우리는 가능한 decision boundary 를 많이 그려보고 이에 대해 

- <mark style='background-color: #fff5b1'> Laplace Approximation </mark>

![cls2](https://user-images.githubusercontent.com/48202736/107110380-b700f180-688a-11eb-8e65-ce0e99f29e0e.png)
*Fig. Posterior를 간단한 어떤 다루기 쉬운 분포로 근사한다.*

![cls3](https://user-images.githubusercontent.com/48202736/107110381-b700f180-688a-11eb-937e-d3340fba0dc5.png)
*Fig. 근사해서 구한 분포와 실제 분포는 크게 다르지 않음을 알 수 있다.*

- <mark style='background-color: #fff5b1'> Bayesian Inference </mark>

*integral 수식*

- <mark style='background-color: #fff5b1'> Approximation of Integral </mark>

![cls4](https://user-images.githubusercontent.com/48202736/107110382-b7998800-688a-11eb-9cca-bd4eccac089c.png)

- <mark style='background-color: #fff5b1'> ML Solution vs Bayesian Solution </mark>

![cls5-1](https://user-images.githubusercontent.com/48202736/107110383-b8321e80-688a-11eb-9d60-901e301a7f81.png)
*Fig. Maximum Likelihood로 단순히 Decision Boundary 하나를 '점 추정(point estimation)'한 결과*

![cls5](https://user-images.githubusercontent.com/48202736/107110384-b8cab500-688a-11eb-89af-1f1033e883c6.png)
*Fig. Maximum Likelihood vs Bayesian Approach*



- <mark style='background-color: #fff5b1'> References </mark>

1. [Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.](http://www.computervisionmodels.com/)
