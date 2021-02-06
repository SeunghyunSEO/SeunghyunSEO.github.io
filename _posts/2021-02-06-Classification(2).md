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

분류 문제를 풀 경우에 데이터가 충분히 많으면 어느 방법을 써도 크게 문제가 없지만, 데이터가 적을 경우에는 특히 ML방식은 문제가 있을 수 있습니다. 

ML 방식으로 Decision Boundary
(이미지 출처 : [A Bayesian graph convolutional network for reliable prediction of molecular properties with uncertainty quantification](https://pubs.rsc.org/en/content/articlepdf/2019/sc/c9sc01992h))

이번 글에서는 Logistic Regression의 Over-Confident를 막아주는 Baywsian Logistic Regression

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



