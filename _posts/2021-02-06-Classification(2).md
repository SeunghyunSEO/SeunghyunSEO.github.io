---
title: Classification (2/?) - Bayesian logistic regression
categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> Bayesian logistic regression </mark>

![cls1](https://user-images.githubusercontent.com/48202736/107110376-b2d4d400-688a-11eb-832b-4cbb5babc175.png)
*Fig. Logistic Regression (Classification)의 여러 변형*

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



