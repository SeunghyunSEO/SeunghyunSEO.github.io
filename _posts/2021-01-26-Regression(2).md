---
title: Regression (2/2)
categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> ML solution for Modeling Gaussian Dist over Output, W </mark>

우리는 이전에 회귀 문제, 그 중에서도 선형 회귀 문제를 푸는 방법에 대해 알아봤습니다.

가장 먼저 출력($$w$$)에 대한 분포를 가우시안 분포로 정의하고 $$likelihood$$인 $$Pr(y \mid x,\theta)$$ 를 최대화 하는 Maximum likelihood 방법이나,

$$\theta$$에 대한 $$prior$$를 하나 더 정의해서 $$likelihood$$와 곱해서 구한 $$posterior$$, $$Pr(\theta \mid x,y)$$ 를 최대화 하는 Maximum A Posterior 방법을 사용했습니다.

![image](https://user-images.githubusercontent.com/48202736/105039364-d1be2280-5aa3-11eb-9f2e-f3ff85d367a4.png)

위의 그림은 MAP로 최적의 파라메터를 구했을 때의 그림입니다.


그치만 사실 뭔가 불편합니다.


뭐가 불편하냐면 그것은 모든 x 에 대해 y 분포가 제 각기 다른데도 불구하고 (데이터 개수 다름), 우리가 찾은 직선은 전구간에 걸쳐 다 똑같은 굵기라는 것입니다. 


이는 다르게 말하면 전 구간에 있어 동일한 confidence를 가지고 있다, 즉 데이터가 없는 부분에서 over-confident 하다는 문제를 보인다는 것입니다.


이를 해결하기 위해서 어떻게할까요? 당장 생각할 수 있는 방법은 $$posterior$$ 가장 큰 값 하나만 구하는 MAP를 사용하지 말고, 
한발 더 나아가 가능한 모든 파라메터에 대해 적분하는 Bayesian 방법을 사용하는 것입니다.


- <mark style='background-color: #fff5b1'> Bayesian Regression </mark>

우리가 잘 아는 관계식이 하나 있습니다.

![image](https://user-images.githubusercontent.com/48202736/105039371-d387e600-5aa3-11eb-8b54-2d9f2b31601e.png)

> 1. $$likelihood : p(x\mid\theta)$$ <br>
> 2. $$posterior \propto likelihood \times prior : p(\theta \mid x) \propto p(x \mid \theta)p(\theta)$$ <br> 

이전에 선형 회귀 문제에서 파라메터를 구하던 것과 비슷하게 접근해봅시다.

$$likelihood$$ 와 $$prior$$를 모두 가우시안 분포로 정의하겠습니다. 
(왜냐면 $$posterior$$를 구해서 적분을 하게 될 텐데, $$posterior$$는 두 분포 $$likelihood$$ 와 $$prior$$의 곱이기 때문입니다. 그리고 두 가우시안 분포의 곱은 가우시안 분포기 때문이죠.)

<center>$$ likelihood : Pr(w|X) = Norm_w[X^T\phi,\sigma^2I] $$</center>

<center>$$ prior : Pr(\phi) = Norm_\phi[0,\sigma_p^2I] $$</center>

여기서 헷갈리지 말아야 할 점은 $$prior$$에 존재하는 분산은 $$\sigma_p^2$$라는 것입니다.

위의 사후 확률을 구하는 관계식을 이용해서 $$posterior$$를 구하면 다음과 같습니다.

<center>$$ posterior : Pr(\theta|X,w) = Norm_\phi[\frac{1}{\sigma^2} A^{-1}Xw, A^{-1}] $$</center>
<center>$$ where A = \frac{1}{\sigma^2} XX^T + \frac{1}{\sigma_p^2}I $$</center>

![image](https://user-images.githubusercontent.com/48202736/105039396-dc78b780-5aa3-11eb-8cdd-c37caca058e6.png)

위의 그림의 왼쪽은 원래 추정하고자 했던 $$\phi$$의 사전 확률인 $$prior$$가 가우시안 분포를 나타내고, 
오른쪽은 $$\phi$$의 분포를 나타냅니다. (variance는 나중에 다룰 예정)


- <mark style='background-color: #dcffe4'> Inference </mark>

자 우리가 학습 데이터 $$X,Y$$ pair, 즉 입력 데이터 $$X$$와 이에 해당하는 정답 $$W$$를 가지고 있다고 합시다. 


그리고 우리는 정의한 모델을 통해 어떤 학습 데이터에 존재하지 않는 $$x^{\ast}$$에 대응하는 $$w^{\ast}$$를 찾고 싶습니다.


'$$x^{\ast}$$에 대응하는 $$w^{\ast}$$' 이는 수식적으로 아래와 같이 표현할 수 있습니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,w) $$</center>

이는 marginalization 테크닉을 통해 아래와 같이 나눌 수 있습니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,w) = \int Pr(w^{\ast}|x^{\ast},\phi) Pr(\phi|X,w) d\phi $$</center>

우리는 이미 $$Pr(w \mid x,\phi)$$와 $$Pr(\phi \mid X,w)$$에 대해 정의를 했습니다. 

계속해서 전개해보겠습니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,w) = \int Norm_{w^{\ast}}[\phi^T x^{\ast},\sigma^2] Norm_{\phi}[\frac{1}{\sigma^2} A^{-1}Xw, A^{-1}] d\phi $$</center>

<center>$$ = Norm_{w^{\ast}}[\frac{1}{\sigma^2}x^{\ast T}A^{-1}Xw,x^{\ast T}A^{-1}x^{\ast} + \sigma^2] $$</center>

이렇게 하면 결과는 아래와 같이 됩니다.

![image](https://user-images.githubusercontent.com/48202736/105039401-dedb1180-5aa3-11eb-9922-10d47a5cbdd8.png)

a)는 추정하고자 하는 파라메터 $$\phi_0,\phi_1$$의 분포를 나타내는 것입니다.
원래 MAP는 여기서 최대가 되는 값 하나만을 학습을 통해 구했으나, 이제는 b)처럼 가능한 파라메터 $$\phi^{1}$$, $$\phi^{2}$$, $$\phi^{3}$$ ... 에 대해서 모두 생각을 해보자는 것이죠.


b)는 파라메터 $$\phi_0,\phi_1$$가 어떤 값이냐에 따라서 선형 회귀의 직선이 어떻게 표현되는지를 나타냅니다.


c)는 말 그대로 위의 식 처럼 가능한 파라메터 $$\phi$$에 대해서 모두 적분한 결과입니다.  


- <mark style='background-color: #fff5b1'> Why Integral over all parameter? </mark>

위에서부터 글을 읽으시면서 "근데 왜 marginalization을 해야 하는거지? 왜 적분을 해야하는거지?"라는 생각을 하시는 분이 계실 것 같습니다.


그것은 이렇게 해석할 수 있을 것 같습니다.


본래 우리가 알고 싶은 것은 $$Pr(Y \mid X)$$ 분포입니다. 


그래서 완벽한 분포를 알고 있다면 여기에 결과를 알고싶은 데이터 포인트 $$x^{\ast}$$를 집어넣으면 당연히 모든 데이터에 대한 분포를 설명하는 $$Pr(Y \mid X)$$를 가지고 있기 때문에
우리는 $$Pr(y^{\ast} \mid x^{\ast} X)$$ 라는 정답을 알 수 있습니다.


하지만 우리는 주어진 문제에 대해 모든 데이터셋을 가지고 있지 않고 학습 데이터만을 가지고 있죠.


우리는 문제를 완벽하게 설명하는 함수가 무엇인지는 알 수 없지만 이와 유사한 근사 함수라도 알고 싶은겁니다. 
그래서 $$\theta$$ 라는 파라메터를 도입해 $$Pr(Y \mid X, \theta)$$ 라는 분포를 만들어 내고, 가장 그럴듯하게 데이터를 설명하는 분포의 파라메터 $$\theta$$를 찾아내는 겁니다..


하지만 $$\theta$$를 학습으로 찾아낸다 한들 임의로 정한 $$\theta$$ 분포를 가져다 쓰는게 마음에 들지 않아 이것마저 없애버리고 싶은겁니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,w) = \int Pr(w^{\ast}|x^{\ast},\phi) Pr(\phi|X,w) d\phi $$</center>

그래서 우리가 구하고 싶은 $$Pr(w^{\ast} \mid x^{\ast},X,w)$$ 를 위와 같은 식으로 나타내고 파라메터에 대해서 적분을 해서 없애버리는 겁니다.


- <mark style='background-color: #fff5b1'> Limitation </mark>

베이지안 방법으로 문제를 푸는 것은 상당히 괜찮은 접근으로 보입니다.


하지만 위에서 유도한 식과 달리 실제로는 $$posterior$$ 를 제대로 구하기 쉽지 않습니다. (위에서는 $$posterior$$ 에 대한 관계식 정도만 언급했음)

> 1. $$likelihood : p(x\mid\theta)$$ <br>
> 2. $$posterior \propto likelihood \times prior : p(\theta \mid x) \propto p(x \mid \theta)p(\theta)$$ <br> 

위와 같이 $$posterior$$ 를 간단한 관계식으로 표현했지만, 사실 $$posterior$$를 구하기 위한 Bayes' Rule은 조금 더 복잡합니다.

<center>$$posterior :  p(\theta \mid X,W) = \frac{p(W \mid X, \theta)p(\theta)}{p(W \mid X)}$$</center>

<center>$$p(W \mid X) = \int p(W|X,\theta)p(\theta)d\theta$$</center>

위에서 보시는 거와 같이 사실은 베이즈 룰에서는 분모, normalizer term 혹은 model evidence 라고 하는 적분 term이 중요합니다.

이 적분을 계산하는 것은 'marginalising the likelihood over ω' 혹은 'marginal likelihood'라고도 합니다.


우리가 예시로 든 간단한 Bayesian Linear Regression 문제에서는 $$likelihood$$와 $$prior$$를 둘 다 conjugate 관계인 가우시안 분포로 적당히 가정했기 때문에 적분을 계산하는게 상대적으로 쉽습니다.


하지만 조금만 복잡해져도 (예를들어 basis function이 고정되어 있지 않은 Basis Function Regression이라던가, Neural Network 라던가...) 위의 적분은 계산하기가 쉽지 않습니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,w) = \int Pr(w^{\ast}|x^{\ast},\phi) Pr(\phi|X,w) d\phi $$</center>

즉, 위의 식에서 $$Pr(\phi \mid X,w)$$를 제대로 구할 수 없다는 것이죠.


이런 경우를 'The true posterior cannot usually be evaluated analytically.' 혹은 'the true posterior is intractable' 하다고 합니다.

이를 해결하기 위해서는 $$true \space posterior$$와 유사하지만 계산하기 쉬운 분포로 근사(approximate)를 해서 문제를 풉니다.

<br><br>

다음에는 위에서 언급한 Bayesian Classification, Bayesian Neural Network 더 나아가 Bayesian Deep Learning에 대해서 더 알아보도록 하겠습니다.


- <mark style='background-color: #fff5b1'> References </mark>

1. [Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.](http://www.computervisionmodels.com/)

2. [Gal, Yarin. "Uncertainty in deep learning." University of Cambridge 1, no. 3 (2016): 4.](https://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html)

