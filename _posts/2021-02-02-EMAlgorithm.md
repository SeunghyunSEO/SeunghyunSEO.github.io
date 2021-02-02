---
title: EM Algorithm
categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> Models with hidden variables </mark>

![image1](https://user-images.githubusercontent.com/48202736/106545430-b85db180-654c-11eb-809d-e2a9727670df.png)

```Key idea```는 Pr(x) 분포에 우리가 알 수 없는 hidden variable h를 추가해서 생각해보자는 겁니다.

![hidden](https://user-images.githubusercontent.com/48202736/106546381-6b7ada80-654e-11eb-8dbb-dc703b8c3a5b.png)

<center>$$ Pr(x) = \int Pr(x,h) dh $$</center>

우리는 위의 식을 또한 아래와 같이 쓸 수도 있겠죠

<center>$$ Pr(x|\theta) = \int Pr(x,h|\theta) dh $$</center>

이제 위의 pdf를 어떻게하면 데이터에 맞게 제대로 피팅할 수 있을까요? 

<center>$$ \hat{\theta} = \arg \max_{\theta} [ \sum_{i=1}^{I} log [\int Pr(x_i,h_i|\theta) dh_i] ]  $$</center>

잘 알려진 방법중에 하나는, log likelihood에 대한 lower bound를 정의하고 그 bound를 반복을 통해 증가시키는 겁니다.

<center>$$ B[\{ q_i(h_i) \}, \theta] = \sum_{i=1}^{I} \int q_i(h_i) log[\frac{Pr(x,h_i|\theta)}{q_i(h_i)}]dh_{1...I} $$</center>
<center>$$ \leq [ \sum_{i=1}^{I} log [\int Pr(x_i,h|\theta) dh_i] ]  $$</center>

- <mark style='background-color: #fff5b1'> (E)xpectation and (M)aximization </mark>

- <mark style='background-color: #fff5b1'> Lower Bound </mark>

![lowerbound1](https://user-images.githubusercontent.com/48202736/106545443-bc89cf00-654c-11eb-9be5-301120d70938.png)

- <mark style='background-color: #fff5b1'> E-Step & M-Step </mark>

![em1](https://user-images.githubusercontent.com/48202736/106545453-c01d5600-654c-11eb-9912-9b3dac3d146a.png)

- <mark style='background-color: #fff5b1'> Mixture of Gaussian (MoG) Example </mark>

