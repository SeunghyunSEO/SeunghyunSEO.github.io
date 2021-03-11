---
title: MLE & MAP(3) - Bayesian Approach
categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> Bayesian Approach </mark>

이제 ML, MAP 대해서 알아봤으니, 이제 Bayesian Approach에 대해서 알아봐야겠죠? 

이제는 다음의 수식이 익숙하실 겁니다 ㅎㅎ

> 1. $$likelihood : p(x\mid\theta)$$ <br>
> 2. $$posterior \propto likelihood \times prior : p(\theta \mid x) \propto p(x \mid \theta)p(\theta)$$ <br> 

![map vs mle](https://user-images.githubusercontent.com/48202736/106485046-89ffb800-64f3-11eb-815e-c7ac0ea84f5f.png)

Maximum Likelihood와는 다르게 Maximum A Posterior는 우리가 추정하고자하는 분포의 파라메터에 대해 사전 분포(prior) 정보를 도입합니다.

그래서 데이터 개수가 적을 때 prior의 도움으로 완전히 엉뚱한 분포를 예측하지 않게 도와주는 역할을 하기도 했죠.

하지만 단순히 posterior를 구한 뒤, 그 중에 가장 큰 값만을 취하는 방식은 (즉 점 추정하는 방식) 

![bayesian1](https://user-images.githubusercontent.com/48202736/106484849-5b81dd00-64f3-11eb-8266-8aa2b32ce4e1.png)
![bayesian2](https://user-images.githubusercontent.com/48202736/106484856-5cb30a00-64f3-11eb-8b16-74e18f508e80.png)
![bayesian3](https://user-images.githubusercontent.com/48202736/106484860-5de43700-64f3-11eb-86fa-384ce5e36ae3.png)


사실 $$poseterior$$에 대해 좀 더 얘기해보자면 이는 베이즈룰을 사용해서 유도한 식으로 원래는 아래와 같이 적분 계산을 해야합니다.

<center>$$posterior :  p(\phi \mid X,W) = \frac{p(W \mid X, \phi)p(\phi)}{p(W \mid X)}$$</center>

