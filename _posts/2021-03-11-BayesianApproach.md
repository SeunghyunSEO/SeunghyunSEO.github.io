---
title: MLE & MAP(3) - Bayesian Approach
categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> Bayesian Approach </mark>

MAP에서와 마찬가지로 다음의 두 관계식에 대해서 적어두고 시작하도록 하겠습니다. 

> 1. $$likelihood : p(x\mid\theta)$$ <br>
> 2. $$posterior \propto likelihood \times prior : p(\theta \mid x) \propto p(x \mid \theta)p(\theta)$$ <br> 

`posterior`란 `likelihood`에 대해 `분포를 나타내는 변수들이 실제로는 '???'한 값을 가질 확률이 높던데?` 라는 사전 정보 prior를 추가한 것이었고
이렇게 만들어진 posterior분포에서 최대값을 리턴하는 파라메터 하나만을 취하는 것이 `MAP`였고 모든 파라메터를 고려하는게 `Bayesian Approach`였죠. 


![map vs mle](https://user-images.githubusercontent.com/48202736/106485046-89ffb800-64f3-11eb-815e-c7ac0ea84f5f.png)

Maximum Likelihood와는 다르게 Maximum A Posterior는 우리가 추정하고자하는 분포의 파라메터에 대해 사전 분포(prior) 정보를 도입합니다.

그래서 데이터 개수가 적을 때 prior의 도움으로 완전히 엉뚱한 분포를 예측하지 않게 도와주는 역할을 하기도 했죠.

하지만 단순히 posterior를 구한 뒤, 그 중에 가장 큰 값만을 취하는 방식은 (즉 점 추정하는 방식) 

![bayesian1](https://user-images.githubusercontent.com/48202736/106484849-5b81dd00-64f3-11eb-8266-8aa2b32ce4e1.png)
![bayesian2](https://user-images.githubusercontent.com/48202736/106484856-5cb30a00-64f3-11eb-8b16-74e18f508e80.png)
![bayesian3](https://user-images.githubusercontent.com/48202736/106484860-5de43700-64f3-11eb-86fa-384ce5e36ae3.png)


사실 $$poseterior$$에 대해 좀 더 얘기해보자면 이는 베이즈룰을 사용해서 유도한 식으로 원래는 아래와 같이 적분 계산을 해야합니다.

<center>$$posterior :  p(\phi \mid X,W) = \frac{p(W \mid X, \phi)p(\phi)}{p(W \mid X)}$$</center>

