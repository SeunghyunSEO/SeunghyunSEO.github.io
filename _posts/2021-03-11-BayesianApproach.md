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


![bayesian1](/assets/images/Bayesian/bayesian1.png)
*Fig. MLE vs MAP*

MAP가 MLE의 단점을 보완하긴 했지만 아직도 부족합니다.
왜냐하면 posterior 분포가 `0.3의 확률로 mean=1, variance=0.7`일 때 가장 큰 값을 리턴한다면 0.3밖에 안되는 확신으로 파라메터를 정한 것이기 때문에 이렇게 구한 분포가 별로 좋을 리 없기 때문이죠.


그래서 `베이지안 방법론 (Bayesian Approach)`는 모든 파라메터에 대해서 고려해서 결과를 산출하자는 것인데요, 

![bayesian3](/assets/images/Bayesian/bayesian3.png)
*Fig. MAP vs Bayesian*

사실 poseterior를 제대로 계산하려면 `베이즈 룰 (Bayes' Rule)`을 사용해서 유도한 식으로 원래는 아래와 같이 적분 계산을 해야합니다.

$$
& posterior \propto likelihood \times prior : p(\theta \vert x) \propto p(x \vert \theta)p(\theta) & \\
& p(\theta \vert x) = \frac{p(x \vert \theta)p(\theta)}{p(x)} & \\
$$

그동안 하던대로 `likelihood`는 `Normal 분포`로 정하고, 마찬가지로 계산을 용이하게 하기 위해서 `prior`, $$p(\theta)$$는 `Normal Inverse Gamma 분포`로 정합니다. 그러면 위의 수식은 아래와 같이 표현할 수 있습니다.

$$
& p(\theta \vert x) = \frac{p(x \vert \theta)p(\theta)}{p(x)} & \\

& p(\mu,\sigma^2 \vert x_{1,\cdots, I}) = \frac{ \prod_{i=1}^I p(x_i \vert \mu, \sigma^2) p(\mu,\sigma^2) }{ p(x_{1,\cdots,I})  } & \\

&  = \frac{ Norm_{x_i} [\mu,\sigma^2] NormInvGam_{\mu,\sigma^2}[\alpha,\beta,\gamma,\delta] }{ p(x_{1,\cdots,I} & \\

& = \frac{ \kappa(\alpha,\beta,\gamma,\delta,x_{1,cdots,I}) NormInvGam_{\mu,\sigma^2}[\tilde{\alpha},\tilde{\beta},\tilde{\gamma},\tilde{\delta}] }{ p(x_{1,\cdots,I} & \\

& = NormInvGam_{\mu,\sigma^2} [\tilde{\alpha},\tilde{\beta},\tilde{\gamma},\tilde{\delta}] & \\
$$

여기서 

$$
& \tilde{\alpha} = \alpha + \frac{I}{2} & \\
& \tilde{\gamma} = \gamma + I & \\
& \tilde{\delta} = \frac{ \gamma \delta + \sum_i x_i }{ \gamma +I } & \\
& \tilde{\beta} = \frac{\sum_i x_i^2}{2} + \beta + \frac{\gamma \delta^2}{2} - \frac{ (\gamma \delta + \sum_i x_i )^2}{ 2(\gamma +I) } & \\
$$

입니다.


수식이 복잡해보이지만 likelihood에 대해 `Conjugage prior`를 사용했기 때매 그래도 간단한 분포가 도출된 것입니다.
이제 우리는 모든 파라메터 $$\theta=\mu,\sigma^2$$에 대해 고려해서 분포를 추정하면 되는겁니다 (`predictive density`).


즉 원래의 
샘플들로 부터 구한 posterior가 있고, 새로운 데이터 포인트 $$x^{\ast}$$가 들어오면 이 데이터는 어떤 확률을 가지는지를 이미 구해진 파라메터로 이루어진 분포의 값을 읽는게 아니고, `새로운 데이터 포인트가 들어올 때 마다 적분을 해서` 값을 리턴하는거죠.


$$
& p(x^{\ast} \vert x_{1,\cdots,I}) = \int p(x^{\ast} \vert \theta) p(\theta \vert x_{1,\cdots,I}) d \theta & \\
& p(x^{\ast} \vert x_{1,\cdots,I}) = \int \int p(x^{\ast} \vert \mu,\sigma^2) p(\mu,\sigma^2 \vert x_{1,\cdots,I}) d \mu \sigma & \\
$$

사실 이 `predictive density`를 구하는 적분식도 데이터가 많아지고 분포가 복잡해지면 계산이 굉장히 어려워지는데요, 지금의 경우에는 할만합니다. (하지만 생략하겠습니다 글이 너무 길어져서...)

![bayesian4](/assets/images/Bayesian/bayesian4.png)
*Fig. 베이지안 방법론은 posterior로 부터 샘플링한 (가능한) 모든 파라메터를 이용해 테스트 데이터의 값을 추론을 하는 것이다.*


이러한 베이지안 방법론은 실제 분류,회귀 문제를 풀 때도 적용이 될 수 있는데요, 이 때 발생하는 계산에 대한 문제 `(intractable posterior)`등을 해결하기 위해서 여러 알고리즘들이 제안되기도 했습니다.


이제 머신러닝/딥러닝의 핵심 개념 중 하나인 MLE, MAP 그리고 Bayesian Approach까지 감을 잡았으니 앞으로는 회귀, 분류문제에 대해서도 다양하게 알아보고 고전 머신러닝 알고리즘들 (퍼셉트론, SVM 등)을 넘어 최신 딥러닝 기법 (DNN, ...)에 대해서도 다뤄보려고 합니다.






## <mark style='background-color: #fff5b1'> References </mark>

1. [Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.](http://www.computervisionmodels.com/)

2. [Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)

