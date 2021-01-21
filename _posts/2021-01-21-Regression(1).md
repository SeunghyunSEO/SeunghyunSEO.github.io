---
title: Regression (1/2)
categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> Regression </mark>

아래의 표에서 볼 수 있듯이, 간단하게 생각하면 


1.입력값이 continuous 한데 결과값이 마찬가지로 continuous하면 Regression 문제라 할 수 있고,


2.입력값이 continuous 한데 결과값이 discrete하면 Classification 문제라 할 수 있다.

![image](https://user-images.githubusercontent.com/48202736/105357223-da476200-5c37-11eb-9612-eaebab62a743.png)

- <mark style='background-color: #fff5b1'> Linear Regression </mark>

1차원 x값에 대해서 이에 대응하는 y값이 존재하는 데이터를 생각해보자.
우리의 목적은 이 데이터를 가장 잘 설명하는 직선 하나를 찾는것이다. 

<img src="https://user-images.githubusercontent.com/48202736/105359057-4fb43200-5c3a-11eb-9268-3f6d5f5c3241.png" width="70%" title="제목"/>

데이터는 x 1차원, y 1차원이니 총 2차원 평면에 뿌려져있고, 우리는 중고등학교때 직선의 방정식을 구하기 위해서는 y절편 하나, 직선의 기울기 하나, 이렇게 딱 두가지만 알면 된다고 알고있다.

<center>$$y=ax+b$$</center>

그러니까 우리가 데이터로부터 학습을 통해 찾아야 될 직선은 a랑 b인 것이다.



여기에 조금 더 보태보자, 우리가 직선의 방정식만 찾으면 어떤 $$x_i$$에 대응하는 $$y_i$$ 는 한 점일텐데, 그렇게 생각하지말고 앞으로는 $$x_i$$에 대응하는게 분포라고 찾는 일이라고 생각해보자.
쉽게 $$x_i$$에 대응하는 $$y_i$$가 가우시안 분포를 따른다고 생각하자.

![image](https://user-images.githubusercontent.com/48202736/105039350-cc60d800-5aa3-11eb-96ec-465f01c3fc46.png)

이 때 $$y_i$$의 평균과 분산이 있을텐데 평균은 $$y_i=ax_i+b$$ 를 따르는 것이다.
그렇다면 우리가 추정하고자 하는 회귀 모양은 위의 그림 (b) 같이 된다.


마치 빔을 쏘는 것 처럼 됐다.
마찬가지로 직선의 방정식을 구하는게 맞긴 맞다. 근데 이제 분포를 곁들인...

- <mark style='background-color: #fff5b1'> 수식으로 보는 Linear Regression </mark>

우리가 위에서 w (혹은 y인데 책에서는 같은 의미로 world state, w를 사용했다.)에 대해서 가우시안 분포를 가정했기 때문에 
우리가 모델링 하고자 하는 분포는 다음과 같다. 

<center>$$ Pr(w_i \mid x_i,\theta) = Norm_{w_i}[\phi_0 + \phi^T x_i, \sigma^2] $$</center>

(각 $$x_i$$에 대응하는 $$y_i$$의 분포인 것이다.)

x가 1차원이지만 notation을 쉽게 만들기 위해서 모든 $$x_i$$에 1을 붙혀보자.

<center>$$ x_i \leftarrow [1 \space x_{i}^{T}]^T $$</center>

그리고 $$\phi$$도 합쳐서 표현하자.

<center>$$ \phi \leftarrow [\phi_0 \space \phi^{T}]^T $$</center>

그러면 위의 모델링 하고자 하는 분포를 아래처럼 다시 쓸 수 있다.

<center>$$ Pr(w_i \mid x_i,\theta) = Norm_{w_i}[\phi^T x_i, \sigma^2] $$</center>

자 이제 우리는 모든 x,y data pair에 대한 식을 위처럼 얻게 되었다.

 - <mark style='background-color: #dcffe4'> likelihood </mark>

우리가 찾고싶은 것은 전체 데이터셋에 대한 $$likelihood$$이다. 

이는 각각의 분포를 전부 곱한것과 같기 때문에 아래와 같이 쓸 수 있다.

<center>$$ Pr(w \mid X) = Norm_{w}[X^T \phi, \sigma^2I] $$</center>

<center>$$ where X = [x_1,x_2, ... x_I] \space and \space w=[w_1,w_2,...,w_I]^T $$</center>

이제 우리는 $$likelihood$$를 가지고 있으니 그전에 다룬 Maximim likelihood를 통해 데이터에 딱 맞는 원하는 파라메터를 구해보자.

<center>$$ \hat{\theta} = argmax_{\theta}[Pr(w|X,\theta)] = argmax_{\theta}[logPr(w|X,\theta)] $$</center>
 
우리가 구하고자 하는 파라메터 $$\theta$$는 지금은 $$\phi_0$$, $$\phi_1$$, $$\sigma^2$$ 세 개 이므로 위의 식을 다시 쓰면, 
(물론 로그를 먼저 likelihood인 가우시안 분포의 식을 풀어 쓰고 log를 취해줌.)

<center>$$ \hat{\phi}, \hat{\sigma^2} = argmax_{\phi,\sigma^2}[ -\frac{Ilog[2\pi]}{2} - \frac{Ilog[\sigma^2]}{2} - \frac{(w-X^T\phi)^T(w-X^T \phi)}{2\sigma^2} ] $$</center>
  
이제 늘 그랬듯이 미분해서 0인 지점을 찾으면 우리는 likelihood를 가장 크게하는, 그러니까 현재 데이터를 가장 likely하게 표현하는 세 가지 파라메터를 구할 수 있다.

 - <mark style='background-color: #dcffe4'> solution </mark>

위의 방법대로 풀면 우리가 Maximum likelihood 방법을 통해 구한 솔루션은 아래와 같게 됩니다.

<center>$$ \hat{\phi} = (XX^T)^{-1}Xw $$</center>

먼저 구한 평균을 결정하는 파라메터들을 통해 분산 마저 구합니다.
  
<center>$$ \hat{\phi} = \frac{(w-X^T\phi)^T(w-X^T \phi)}{I} $$</center>

직선의 방정식을 구하고, 거기에 균일한 분산을 곁들인거죠.


- <mark style='background-color: #fff5b1'> 가우시안 분포를 가정한 ML solution과 MSE의 관계  </mark> 

어떤 분들은 위의 솔루션이 맘에 들지 않을 수도 있습니다.

왜냐하면 대부분의 머신러닝 교재,강의에서 커브 피팅(혹은 직선 피팅)을 할 때 $$X(x_1,x_2...)$$와 $$X(y_1,y_2...)$$ 데이터가 쭉 존재할 때, 
Mean Squared Error (MSE) 를 통해 해를 구하는 방식을 얘기하기 때문입니다.


어떤 의미냐 하면 아래의 커브 피팅(곡선 피팅) 의 예시를 보시면, 

![image](https://user-images.githubusercontent.com/48202736/105365680-d9b3c900-5c41-11eb-9325-86c051a53e4b.png)

목적은 우리가 구하려는 파라메터는 곡선을 나타내는데 (직선의 방정식 얘기하다가 갑자기 곡선으로 넘어와서 햇갈리실 수 있지만 매커니즘은 같습니다.)
그 곡선과 실제 y값과의 차이(error)가 존재하고, 이를 계산해서 줄이는 방식으로 파라메터를 점차적으로 학습 (gradient descent) 혹은 한방에(closed-form solution) 구하겠다. 입니다.

```주어진 데이터와 잘 맞는 직선 혹은 곡선의 방정식을 찾겠다``` 라는 목적은 같지만 차이(error)를 계산하는 식을 도입해서 문제를 풀자는 겁니다.

여기서 regression을 할 때 널리 알려진 MSE Loss는 다음과 같이 나타낼 수 있습니다.

<center>$$ Loss(\theta) = \frac{1}{2} \sum_{i=1}^{I}{ \{ f(x_i,\theta)-y_i \} }^2 $$</center>

일반적으로 loss를 줄여야 하는 방향으로 업데이트 해야 하기 때문에 식을 다시 쓰면 아래와 같습니다.

<center>$$ \hat{\theta} = argmin_{\theta}\frac{1}{2} \sum_{i=1}^{I}{ \{ f(x_i,\theta)-y_i \} }^2 $$</center>


그런데 이는 사실 위에서 구한 식과 같습니다.

<center>$$ \hat{\phi}, \hat{\sigma^2} = argmax_{\phi,\sigma^2}[ -\frac{Ilog[2\pi]}{2} - \frac{Ilog[\sigma^2]}{2} - \frac{(w-X^T\phi)^T(w-X^T \phi)}{2\sigma^2} ] $$</center>

이를 다시쓰면 다음과 같습니다.

<center>$$ \hat{\phi}, \hat{\sigma^2} = argmax_{\phi,\sigma^2}[ - \frac{1}{2}\frac{1}{\sigma^2}{\sum_{i=1}^{N}{\{f(x_i,\phi)-w_i\}}^2} -\frac{Nlog[2\pi]}{2} - \frac{Nlog[\sigma^2]}{2}] $$</center>

여기서 분산에 대한 식은 다 떼어버리고 생각하면 이는 MSE loss식과 같습니다. (argmax, argmin의 차이가 이것마저 같게 하면 아예 동일합니다.)

<center>$$ 1. \space \hat{\theta} = argmin_{\theta}\frac{1}{2} \sum_{i=1}^{I}{ \{ f(x_i,\theta)-y_i \} }^2 $$</center>

<center>$$ 2. \space \hat{\phi} = argmax_{\phi}[ - \frac{1}{2}\frac{1}{\sigma^2}{\sum_{i=1}^{N}{\{f(x_i,\phi)-w_i\}}^2}] $$</center>

결과적으로 노이즈가 가우시안 분포를 가진다는 가정하에, 즉 가우시안 분포로 y를 모델링 한 경우 $$likelihood$$를 maximize하는 방법이 일반적인 회귀에 쓰이는 MSE를 최소화 하는것과 같다는 걸 알 수 있습니다.


- <mark style='background-color: #fff5b1'> MAP로 Linear Regression </mark> 

우리는 앞서 ML과 MAP의 차이에 대해서 공부했었습니다. 사후 확률(posterior)는 likelihood에 prior 정보를 추가해 데이터가 별로 없을 때 likelihood를 약간 보정해주는 느낌이라고 설명했었습니다.

<center>$$posterior \propto likelihood \times prior$$</center>

<center>$$ Pr(\theta \mid x,w) \propto Pr(w \mid x,\theta) \times Pr(\theta)$$</center>

여기서는 x,y가 데이터로 주어졌고 \theta 를 통해 찾는 것이기 때문에 y와 $$\theta$$가 조건부 확률의 어디에 있느냐가 중요합니다.


prior는 $$\theta$$에 대한 사전 정보가 되고 ( $$likelihood$$가 가우시안 분포이기 때문에 mean, variance가 어떤 값을 가질 확률이 어떻다~ 를 나타내는 분포가 된다. )


prior를 적절히 다음과 같이 0 mean 가우시안 분포로 고르고

<center>$$ Pr(\theta \mid \alpha) = Norm_{\theta}[0,\sigma^2] $$</center>

<center>$$ Pr(\theta \mid \alpha) = \frac{1}{\sqrt{2\pi\sigma^2}}exp[-0.5\frac{(\theta-0)^2}{\sigma^2}] $$</center>

posterior를 최대화 하는 solution을 구하면

이는 다음을 최소화 하는 것과 같은 solution을 구할 수 있게 됩니다.

<center>$$ \frac{1}{2\sigma^2} \sum_{i=1}^{I}{ \{ f(x_i,\phi)-y_i \} }^2 + \frac{\alpha}{2}{\phi^T \phi} $$</center>

즉 파라메터에 사전 확률을 넣어 계산하는 베이지안 관점으로 문제를 푸는것이 우리가 잘 알고있는 MSE Loss로 선형 회귀 문제를 풀 때, 곡선(혹은 직선)의 오버피팅을 막기위해 weight decay 정규화 제약식을 추가하는 것과 같은 매우 좋은 효과를 가져온다는 것입니다.

- <mark style='background-color: #fff5b1'> Further Study </mark>

지금까지 이야기 한 것 외에도, 모든 가능한 파라메터를 다 고려하는 Bayesian Regression 방법과

Non-linear Regression 등등의 다양한 업그레이드 버전이 있습니다. 

이것들은 2편에서 다루도록 하겠습니다.

![image](https://user-images.githubusercontent.com/48202736/105039371-d387e600-5aa3-11eb-8b54-2d9f2b31601e.png)


- <mark style='background-color: #fff5b1'> Bayesian Regression </mark>



![image](https://user-images.githubusercontent.com/48202736/105039371-d387e600-5aa3-11eb-8b54-2d9f2b31601e.png)
![image](https://user-images.githubusercontent.com/48202736/105039396-dc78b780-5aa3-11eb-8cdd-c37caca058e6.png)

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
