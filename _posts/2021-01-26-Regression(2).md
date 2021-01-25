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


뭐가 불편하냐면 그것은 모든 x 에 대해 y 분포가 제 각기 다른데도 불구하고, 우리가 찾은 직선은 전구간에 걸쳐 다 똑같은 굵기라는 것입니다. 


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

<center>$$ posterior : Pr(\theta|X,w) = Norm_\phi[\frac{1}{sigma^2} A^{-1}Xw, A^{-1}] $$</center>
<center>$$ where A = \frac{1}{\sigma^2} XX^T + \frac{1}{\sigma_p^2}I $$</center>

![image](https://user-images.githubusercontent.com/48202736/105039396-dc78b780-5aa3-11eb-8cdd-c37caca058e6.png)

확실히 해야 할 것은 우리가 찾고싶은 것은 선형회귀의 gradient를 나타낼 $$\phi$$와 $$\sigma^2$$라는 점입니다. ($$\sigma_p^2$$는 그냥 정해졌습니다.)

위의 그림의 왼쪽은 원래 추정하고자 했던 $$\phi$$의 사전 확률인 $$prior$$가 가우시안 분포를 나타내고, 
오른쪽은 $$prior$$를 그림으로는 안나와 있지만 가우시안 분포를 가지는 $$likelihood$$ 곱했을 때 나오는 실제 추정하고자 하는 파라메터, $$\phi$$의 분포를 나타냅니다. (variance는 나중에 다룰 예정)


자 우리가 학습 데이터 $$X,Y$$ pair, 즉 입력 데이터 $$X$$와 이에 해당하는 정답 $$W$$를 가지고 있다고 합시다. 


그리고 우리는 정의한 모델을 통해 어떤 학습 데이터에 존재하지 않는 $$x^{\ast}$$에 대응하는 $$w^{\ast}$$를 찾고 싶습니다.


'$$x^{\ast}$$에 대응하는 $$w^{\ast}$$' 이는 수식적으로 아래와 같이 표현할 수 있습니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,w) $$</center>

이는 marginalization 테크닉을 통해 아래와 같이 나눌 수 있습니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,w) = \int Pr(w^{\ast}|x^{\ast},\phi) Pr(\phi|X,w) d\phi $$</center>

우리는 이미 $$Pr(w|x,\phi)$$와 $$Pr(\phi|X,w)$$에 대해 정의를 했습니다. 

계속해서 전개해보겠습니다.

<center>$$ = \int Norm_{w^{\ast}}[\phi^T x^{\ast},\sigma^2] Norm_{\phi}[\frac{1}{\sigma^2} A^{-1}Xw, A^{-1}] d\phi $$</center>

<center>$$ Norm_{w^{\ast}}[\frac{1}{\sigma^2}x^{\ast T}A^{-1}Xw,x^{\ast T}A^{-1}x^{\ast} + \sigma^2] $$</center>

이렇게 하면 결과는 아래와 같이 됩니다.

![image](https://user-images.githubusercontent.com/48202736/105039401-dedb1180-5aa3-11eb-9922-10d47a5cbdd8.png)

a)는 추정하고자 하는 파라메터 $$\phi_0,phi_1$$의 분포를 나타내는 것입니다.
원래 MAP는 여기서 최대가 되는 값 하나만을 학습을 통해 구했으나, 이제는 b)처럼 가능한 파라메터 $$\phi^{1}$$, $$\phi^{2}$$, $$\phi^{3}$$ ... 에 대해서 모두 생각을 해보자는 것이죠.


b)는 파라메터 $$\phi_0,phi_1$$가 어떤 값이냐에 따라서 선형 회귀의 직선이 어떻게 표현되는지를 나타냅니다.


c)는 말 그대로 위의 식 처럼 가능한 파라메터 $$\phi$$에 대해서 모두 적분한 결과입니다.  

- <mark style='background-color: #fff5b1'> Non-Linear Regression </mark>

![image](https://user-images.githubusercontent.com/48202736/105039371-d387e600-5aa3-11eb-8b54-2d9f2b31601e.png)


![image](https://user-images.githubusercontent.com/48202736/105039467-ef8b8780-5aa3-11eb-994b-9e82c2569038.png)


![image](https://user-images.githubusercontent.com/48202736/105039492-f6b29580-5aa3-11eb-89ad-72bed31ccea3.png)

- <mark style='background-color: #fff5b1'> Kernelization and Gaussian processes </mark>

![image](https://user-images.githubusercontent.com/48202736/105039371-d387e600-5aa3-11eb-8b54-2d9f2b31601e.png)
![image](https://user-images.githubusercontent.com/48202736/105039516-fe723a00-5aa3-11eb-986c-3a245635e6c6.png)
![image](https://user-images.githubusercontent.com/48202736/105039578-10ec7380-5aa4-11eb-991c-a2f0639e6446.png)

- <mark style='background-color: #fff5b1'> Sparse linear regression </mark>

![image](https://user-images.githubusercontent.com/48202736/105039371-d387e600-5aa3-11eb-8b54-2d9f2b31601e.png)

![image](https://user-images.githubusercontent.com/48202736/105341731-ca715300-5c22-11eb-92ae-7424f092c401.png)
![image](https://user-images.githubusercontent.com/48202736/105039605-1a75db80-5aa4-11eb-950e-e6e9a75f20ba.png)

- <mark style='background-color: #fff5b1'> Dual Linear Regression </mark>

![image](https://user-images.githubusercontent.com/48202736/105039618-1d70cc00-5aa4-11eb-9bf1-4f79f5156595.png)

- <mark style='background-color: #fff5b1'> Sparse linear regression </mark>

![image](https://user-images.githubusercontent.com/48202736/105039371-d387e600-5aa3-11eb-8b54-2d9f2b31601e.png)
![image](https://user-images.githubusercontent.com/48202736/105341781-d9580580-5c22-11eb-8331-bac2dcc117ee.png)
