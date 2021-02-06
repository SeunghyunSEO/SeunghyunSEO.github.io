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
*Fig. 1. Logistic Regression (Classification)의 여러 변형*

우리가 흔히 머신러닝 방법론을 통해 분류, 회귀를 하는 방식은 크게 3가지가 있었습니다.

이전 글들에서 많이 설명을 했기 때문에 이번에는 간략하게 요약만 하고 넘어가도록 하겠습니다.

> 1. Maximum Likelihood (ML) : likelihood 를 정의하고 이를 최대화하는 단 하나의 값(점)을 찾는다. (점 추정) <br>
> 2. Maximum A Posteriori (MAP) : likelihood를 정의하고 추가로 파라메터에 대한 prior를 정의해서 베이즈룰을 통해 posterior를 구한 뒤 이를 최대화하는 단 하나의 값(점)을 찾는다. (점 추정) <br>
> 3. Bayesian Approach : 2번의 posterior 분포를 구하고 점 추정 하지 않고(분포 그대로), 추론 시 파라메터에 대해(posterior 분포를) 전부 적분해서 사용한다. <br>

분류 문제를 풀 경우에 우리는 일반적으로 ML,MAP 방식을 사용할 수 있지만 이는 어떤 부분에서는 문제가 있을 수 있습니다. 

예를들어 MAP 방식으로 Decision Boundary를 정하는 것은 아래와 같은 문제가 있을 수 있는데, 
MAP는 posterior 분포로부터 최대값을 나타내는 단 하나의 파라메터만을 주어진 학습 데이터를 통해 구해서 쓰는 것이기 때문에 Decision Boundary 근처에 어떤 데이터가 주어졌을 때 이를 class1이라고 과잉 확신 하는 경우가 있습니다. 

(예를 들어, class1 :90%, class2:10%) 

<img width="1182" alt="bayesian_cls" src="https://user-images.githubusercontent.com/48202736/107111166-0c400180-6891-11eb-93d1-6f6a16fba8b4.png">
*Fig. 2. MAP(좌) vs Bayesian Approach(우), 이미지 출처 : [A Bayesian graph convolutional network for reliable prediction of molecular properties with uncertainty quantification](https://pubs.rsc.org/en/content/articlepdf/2019/sc/c9sc01992h)*

이는 Decision boundary가 아래의 그림같이 생겼기 때문인데요,

![cls5-1](https://user-images.githubusercontent.com/48202736/107110383-b8321e80-688a-11eb-9d60-901e301a7f81.png)
*Fig. 3. Decision Boundary는 왼쪽과 같은 logistic 함수를 사용해서 만들기 때문입니다. decision boundary가 출력이 0.5인 부분이라고 하면 이 값을 전후로 굉장히 높은 확률로 class를 확신해서 분류하게 됩니다. *

이렇게 극단적으로 클래스를 나눠주지 말고 *Fig. 2*의 오른쪽 그림처럼 결정 경계 근처에 존재하는 테스트 데이터에 대해서 분류기가 주는 불확실성을 조금 더 표현해 줬으면 좋지 않을까요?

(예를 들어, class1 : 54%, class2:56%)

이러한 생각 때문에 우리는 Bayesian Approach를 통해서 조금 더 자연스러운 Inference를 하고싶은 겁니다.

- <mark style='background-color: #dcffe4'> 수식으로 보는 Bayesian Logistic Regression </mark>

자 이제 수식적으로 접근해보도록 합시다.

> <mark style='background-color: #dcffe4'> Notation </mark> <br>
> $$ x $$ : input state, 데이터 입력값 <br>
> $$ w $$ : world state, x에 대응하는 값 <br>
> $$ \theta $$ : parameter, 우리가 알고싶은, 추정하려는 값 <br>

우리는 이진 분류 문제에 대해서 likelihood를 베르누이 분포로 모델링 하는것을 이전의 글에서 충분히 이해했습니다.

![ber1](https://user-images.githubusercontent.com/48202736/106453016-04b4dd00-64cc-11eb-9278-625d36eaa5be.png)
{: style="width: 60%;" class="center"}
*Fig. 4. Bernoulli Distribution*

그렇기 때문에 우리는 likelihood를 다음과 같이 나타낼 수 있었죠.

<center>$$ Pr(w|X,\phi) = \prod_{i=1}^{I} \lambda^{w_i}(1-\lambda)^{1-w_i} $$</center>

<center>$$ Pr(w|X,\phi) = \prod_{i=1}^{I} (\frac{1}{1+exp[-\phi^T x_i]})^{w_i}(1-\frac{1}{1+exp[-\phi^T x_i]})^{1-w_i} $$</center>

<center>$$ Pr(w|X,\phi) = \prod_{i=1}^{I} (\frac{1}{1+exp[-\phi^T x_i]})^{w_i}(\frac{exp[-\phi^T x_i]}{1+exp[-\phi^T x_i]})^{1-w_i} $$</center>

우리가 Bayesian Logistic Regression을 하기 위해 해야할 것은 아래의 식이므로 (파라메터에 대해 전부 적분),
우리는 posterior또한 구해야 합니다.

<center>Pr(w^{\ast} \vert x^{\ast}, X, W) = \int Pr(w^{\ast} \vert x^{\ast} \phi) Pr(\phi | X, W) d\phi <center>
  

사후분포(posterior)를 구하기 위해 구하고자 하는 파라메터에 대한 사전 분포(prior)를 도입해줍시다.

<center>$$ Pr(\Phi) = Norm_{\phi}[0,\sigma_p^2 I] $$</center> 

(주의할점은 이 prior는 likelihood와 conjugate 관계가 아니기 때문에 둘을 곱하면 더럽게 계산이 될 수도 있습니다.)

베이즈룰을 적용해서 이제 posterior를 구해보도록 하죠.

<center>Pr(\phi \vert X, w) = \frac{ Pr(w \vert X, \phi) Pr(\phi) }{ Pr(w \vert X) }</center>

하지만 여기서 문제가 있습니다.

바로 posterior를 위한 closed form solution이 따로 존재하지 않는다는 것인데요. (바로 베이즈룰을 사용할 때 분모에 있는 수식을 구하기 위한 적분을 계산할 수 없음)

이 posterior를 구해야 아래의 식을 편하게 적분할 수 있을텐데,

<center>Pr(w^{\ast} \vert x^{\ast}, X, W) = \int Pr(w^{\ast} \vert x^{\ast} \phi) Pr(\phi | X, W) d\phi <center>
  
그럴 수 없기때문에 우리는 approximation을 통해서 실제 posterior와 근사한 어떤 함수를 도입할겁니다.

(+ 회귀 문제에서 간단한 베이지안 선형 회귀를 할 경우에서는 Posterior가 계산이 쉽기 때문에 이런 문제는 없었습니다) 

- <mark style='background-color: #fff5b1'> Posterior Approximation </mark>

- <mark style='background-color: #dcffe4'> Laplace Inference </mark>

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
