---
title: Classification (1/2)
categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> Regression VS Classification </mark>

아래의 표에서 볼 수 있듯이, 간단하게 생각하면 


1.입력값이 continuous 한데 결과값이 마찬가지로 continuous하면 Regression 문제라 할 수 있고,


2.입력값이 continuous 한데 결과값이 discrete하면 Classification 문제라 할 수 있습니다.


![image](https://user-images.githubusercontent.com/48202736/105444745-c7717380-5cb1-11eb-92a0-b618ad4d6b4f.png)

- <mark style='background-color: #fff5b1'> Logistic Regression </mark>

로지스틱 회귀 정의 추가해야할듯

본론)

앞서 Regression 문제에서와 마찬가지로 우리는 연속적인 입력 변수에 대해서 대상 y의 분포를 모델링 할 수 있습니다. 선형 회귀에서는 y를 가우시안 분포로 정의하여 문제를 풀었습니다.

자 이번에는 y의 분포를 베르누이 확률 분포로 모델링 해보도록 하겠습니다.

그 전에 우선 베르누이 분포에 대해서 remind를 해보도록 하겠습니다. 

- <mark style='background-color: #dcffe4'> Bernoulli Distribution </mark>

베르누이 분포에 대해 쉽게 설명하기 위해 그림을 먼저 보도록 하겠습니다.

![image](https://user-images.githubusercontent.com/48202736/105621207-7fd12000-5e48-11eb-9106-42a0a58fb4a2.png)

베르누이 분포는 '$$x=0/x=1$$' 이나 '성공/실패' 등 두 가지 가능한 경우에 대한 상황을 나타냅니다.

수식으로 나타내면 아래와 같은데, $$x=1$$일 확률이 $$\lambda$$ 이고, 베르누이 분포는 두 가지 경우에 대해서만 생각하기 때문에 반대로 $$x=0$$ 이 될 확률은 $$1-\lambda$$가 됩니다.
(예를 들어, 어떤 x(이미지 픽셀값)가 $$x=0$$(강아지)일 확률이 $$lambda$$(0.64)면 $$x=1$$(고양이)일 확률은 $$1-\lambda$$(1-0.64=0.34)가 됩니다.)

![image](https://user-images.githubusercontent.com/48202736/105621213-86f82e00-5e48-11eb-8f27-74ec370737da.png)

위의 수식을 보면 베르누이 분포를 한번에 $$\lambda^{x}(1-\lambda)^{1-x}$$로 표현하는 걸 알 수 있습니다. 이는 x가 1이면 $$\lambda$$가 되고 x가 0이면 $$(1-\lambda)$$가 되는 수식입니다.
이러한 의미를 가지는 베르누이 분포는 아래와 같이 쓰기도 합니다.

![image](https://user-images.githubusercontent.com/48202736/105621209-8495d400-5e48-11eb-8ab7-2095f20068c6.png)

이 때 추정하고자 하는 파라메터는 성공 확률(편의상 이렇게 말하겠습니다. 경우에 따라 다르게 말할 수 있을 것 같습니다.), $$\lambda$$가 되겠죠? (가우시안 분포에서 평균,$$\mu$$와 분산,$$\sigma^2$$를 찾는게 목적이듯)

- <mark style='background-color: #dcffe4'> Categorical Distribution </mark>

+) (나중에 다루겠지만, 미리 얘기하고 넘어가겠습니다.) 베르누이 분포와 분포로 Categorical(범주형) 분포가 있습니다.

![image](https://user-images.githubusercontent.com/48202736/105621216-96777700-5e48-11eb-99c4-400cf91c6405.png)

Categorical 분포는 베르누이 분포와 크게 다르지 않지만, 발생 가능한 케이스가 두가지 이상이 된 경우에 대한 이야기를 합니다.
아래의 수식은 Categorical 분포에 대한 수식을 나타냅니다.

![image](https://user-images.githubusercontent.com/48202736/105621218-9aa39480-5e48-11eb-88ed-ac911e2e2a76.png)

베르누이 분포에서는 두 가지 케이스중 한 가지 케이스로 분류될 확률 $$\lambda$$만 찾아내면 확률 분포의 합이 1이 되는것을 이용하여 자동으로 나머지 케이스로 분류될 확률이 정해졌습니다.

하지만 Categorical 분포는 예를들어 x=0~x=4 (5가지 가능한 경우)에 대한 확률을 전부 다 찾아내야 함으로 전체 경우에 대한 가능한 확률 값을 추정해야 합니다. (근데 이제 $$\sum{\lambda_{0...I}} = 1$$ 이 되는...)

이를 베르누이 분포처럼 아래와 같이 표현할 수 있습니다. (굵은 $$\lambda$$는 스칼라가 아니라 아니라 벡터) 

![image](https://user-images.githubusercontent.com/48202736/105621220-9c6d5800-5e48-11eb-8d87-bc128a488378.png)

- <mark style='background-color: #dcffe4'> Back to Logistic Regression </mark>

다시 본론으로 돌아가서 Logistic Regression에 대해 얘기해보겠습니다.

> <mark style='background-color: #dcffe4'> Notation </mark> <br>
> $$ x $$ : input state, 데이터 입력값 <br>
> $$ w $$ : world state, x에 대응하는 값 <br>
> $$ \theta $$ : parameter, 우리가 알고싶은, 추정하려는 값 <br>

Logistic Regression은 설명한 대로 베르누이(Bernolli) 확률 분포로 world state, $$w$$를 모델링 합니다.

<center>$$ Pr(w|\lambda) = Bern_w[\lambda] $$</center>

근데 여기서 $$\lambda$$ 를 입력 데이터 x에 대한 함수로 표현을 해봅시다.

<center>$$ Pr(w|\phi_0,\phi,x) = Bern_w[sig[a]] $$</center>

<center>$$ where, a = \phi_0 + \phi^T x$$</center> and <center>$$sig[a] = \frac{1}{1+exp[-a]} $$</center>

여기서 $$a$$는 $$x$$에 $$\phi$$를 통해 선형 결합한 것이고 

$$sig[a]$$는 a를 sigmoid 라는 함수에(활성 함수라고도 함) 넣은 결과값입니다.

sigmoid 함수는 아래의 그림처럼 $$[-\infty,\infty]$$ 사이의 입력값을 $$[0,1]$$ 사이의 값으로 매핑해줍니다.

![image](https://user-images.githubusercontent.com/48202736/105623021-afd4ef00-5e59-11eb-892d-4916ee66edb1.png)

(이미지 출처 : [link](https://en.wikipedia.org/wiki/Sigmoid_function#:~:text=4%20Applications-,Definition,refer%20to%20the%20same%20object.))

```
입력값을 굳이 sigmoid 함수에 넣어 최종 출력값으로 나타내는 것은 여러가지 의미가 있습니다. 지금은 다루지 않지만 크게 다른 개념이 아닌 Neural Network가 인간의 신경망을 motive로 만들었다는 것 같이, sigmoid가 neural firining을 나타내는 것이라고 하는 사람도 있고, 아니면 확률로 모델링 하기 위해서 출력값을 확률처럼 나타내기 위해서 라는 말도 있습니다. (sigmoid가 0~1 사이의 값을 뱉기 때문)
```

아무튼 위의 과정을 다시 그림으로 나타내면 아래와 같습니다.

```
* 아래의 오른쪽 그림이 저는 처음 봤을 때 좀 햇갈렸습니다. 편하게 그림이 3차원 이라고 생각하시면 될 것 같습니다. Pr(y|x), x, w 세 가지 축이 있는거죠. 
```

![image](https://user-images.githubusercontent.com/48202736/105444774-d3f5cc00-5cb1-11eb-93e4-f280a7328d92.png)




- <mark style='background-color: #dcffe4'> Decision Boundary </mark>

조금 더 notation을 깔끔하게 써보겠습니다.

<center>$$ Pr(w|\phi_0,\phi,x) = Bern_w[sig[a]] $$</center>

를 더 쉽게 쓰기 위해서 입력 데이터 x (1차원, 2차원 ... 전부 가능, 우리가 잘 아는 MNIST 손글씨 데이터라면 784차원)에 1을 붙혀봅시다.
그러면 데이터셋의 데이터 하나 하나는 다음과 같이 표현할 수 있습니다.

<center>$$ x_i \leftarrow [1 \space x_{i}^{T}]^T $$</center>

추정하고자 하는 파라메터도 간단하게 쓰기위해서 offset과 gradient 벡터, $$\phi_0$$ 과 $$\phi$$를 붙혀봅니다.

<center>$$ \phi \leftarrow [\phi_0 \space \phi^{T}]^T $$</center>

이렇게하면 notation을 깔끔하게 다시 쓸 수 있습니다.

<center>$$ Pr(w|\phi_0,\phi,x) = Bern_w[sig[a]] $$</center>
<center>$$ \downarrow $$</center>
<center>$$ Pr(w|\phi,x) = Bern_w[\frac{1}{1+exp[- \phi^T x]}] $$</center>

학습이 잘 된 상태를 가정해봅시다.

아래의 그림의 왼쪽은 입력 데이터 x가 1차원, 오른쪽은 x가 2차원인 경우입니다.

![image](https://user-images.githubusercontent.com/48202736/105038606-e3eb9100-5aa2-11eb-9b1d-070d4e6edd32.png)

분류 문제를 푼다고 생각할 때, 베르누이 확률 분포가 0.5가 되는 지점을 두 클래스가 어떤 것인지 결정하는 'Decision Boundary'라고 생각하면 그 선을 기준으로 클래스를 나누게 될 것입니다.

현재는 파란(cyan?)색이 decision boundary입니다.

```
* 사실 2차원 데이터도 아래의 그림처럼 생각하는게 더 직관적입니다. z축이 클래스를 나누는 베르누이 분포의 확률 분포가 될 겁니다.
```

![image](https://user-images.githubusercontent.com/48202736/105623182-14dd1480-5e5b-11eb-9512-121dc3549fef.png)

(출처 : [link](https://medium.com/analytics-vidhya/ml-from-scratch-logistic-regression-gradient-descent-63b6beb1664c))

Logistic Regression은 사실 간단히 말해서 한마디로 정리할 수 있습니다. 


바로 "데이터로부터 최적의 Decistion Boundary를 정하는 것" 입니다.


- <mark style='background-color: #dcffe4'> Maximum Likelihood </mark>

이제 어떻게 하면 위의 그림처럼 데이터로부터 최적의 Decision Boundary를 찾아낼 수 있는지,

그러니까 최적의 파라메터 $$\phi$$ (1차원이면 $$\phi_0, \phi_1$$) 를 찾아낼 수 있을지를 알아봅시다.


어떻게 해야할까요?


네 맞습니다. 이전에 ML, MAP, Bayesian에 대해서 배우셨다면 이것들 중 하나를 쓰면 됩니다. 어떤 방법으로 파라메터를 추정해도 상관 없습니다.

그 중 Maximum Likelihood (ML) 방법을 쓴다고 생각하면 다음과 같이 잘 아시는 것 처럼 다음의 방법을 통해 최적의 파라메터를 구할 수 있습니다.
 
> 1.$$likelihood$$를 정의한다.<br>
> 2.전체 식에 log를 취한다. <br>
> 3.미분을 취해 0인 지점을 찾는다.<br>

한번 해볼까요?

 
$$likelihood$$ 다음과 같이 나타낼 수 있습니다.

<center>$$ Pr(w|X,\phi) = \prod_{i=1}^{I} \lambda^{w_i}(1-\lambda)^{1-w_i} $$</center>

<center>$$ Pr(w|X,\phi) = \prod_{i=1}^{I} (\frac{1}{1+exp[-\phi^T x_i]})^{w_i}(1-\frac{1}{1+exp[-\phi^T x_i]})^{1-w_i} $$</center>

<center>$$ Pr(w|X,\phi) = \prod_{i=1}^{I} (\frac{1}{1+exp[-\phi^T x_i]})^{w_i}(\frac{exp[-\phi^T x_i]}{1+exp[-\phi^T x_i]})^{1-w_i} $$</center>

이제 Logarithm을 취해봅시다.

<center>$$ L = \sum_{i=1}^{I} w_i log[\frac{1}{1+exp[-\phi^T x_i]}] + \sum_{i=1}^{I}(1-w_i)log[\frac{exp[-\phi^T x_i]}{1+exp[-\phi^T x_i]}] $$</center>

마지막으로 미분을 해봅시다.

<center>$$ \frac{\partial L}{\partial \phi} = -\sum_{i=1}^{I}(\frac{1}{1+exp[-\phi^T x_i]} - w_i)x_i = - \sum_{i=1}^{I}(sig[a_i]-w_i)x_i $$</center>

이제 한방에 최적 해를 구할 수 있을까요?


답은 아닙니다. 안타깝게도 Logistic Regression은 $$\phi$$를 x와 w에 대해 한방에 정리해 그 식을 최소화 하는 해를 구할 수 없습니다. 즉 닫힌 형태의 해(Closed-form Solution)를 바로 구할 수 없다는 것입니다.

이는 다른 말로는 Analytic Solution이 존재하지 않는다고 할 수도 있습니다. (반대로는 Numerical Solution)


(이에 대해 더 궁금하신분들은 구글링을 해보시길 바랍니다 ㅎ...)


(임시 참조 링크, 나중에 정리해야함... [참조1](http://wanochoi.com/?p=5061), [참조2](https://stats.stackexchange.com/questions/949/when-is-logistic-regression-solved-in-closed-form)

- <mark style='background-color: #dcffe4'> Optimization </mark>

Logistic Regression이 닫힌 형태의 해가 존재하지 않기 때문에 이제는 다른 방법을 사용해 최적의 해를 찾아야 합니다. "Iterative Non-linear Opitmization"이 바로 그 방법이라고 할 수 있는데, 이는 반복적으로 추정하고자 하는 파라메터를 점진적으로 업데이트 해나가며 결국 최적해에 가까워지게 만드는 것.

```
* 하지만 위의 경우처럼 closed-form solution이 존재하지 않는 경우에만 최적화 기법으로 해를 구할 수 있는건 아닙니다. 이전에 다뤘던 Linear Regression은 Closed-form Solution이 존재했지만 마찬가지로 Iterative하게 파라메터를 업데이트 해서 최적 해를 찾을 수 있습니다. 그리고 최적화 기법을 사용한다고 해서 꼭 단 하나의 최적의 솔루션을 찾을 수 있는 것은 아닙니다, 그에 가까운 해를 찾을 수가 있을 뿐이죠.  또한 운이 나쁘면 못찾을 ...
```


![image](https://user-images.githubusercontent.com/48202736/105444895-128b8680-5cb2-11eb-91a7-dac84df8707d.png)
![image](https://user-images.githubusercontent.com/48202736/105444900-14ede080-5cb2-11eb-9347-cb53fbfa5a5a.png)
![image](https://user-images.githubusercontent.com/48202736/105444909-18816780-5cb2-11eb-9c32-403825583254.png)

- <mark style='background-color: #fff5b1'> Gradient Based Optimization </mark>

- <mark style='background-color: #dcffe4'> Steepest Gradient Descent  </mark>

![image](https://user-images.githubusercontent.com/48202736/105444927-23d49300-5cb2-11eb-8336-1e13ccfd1901.png)
![image](https://user-images.githubusercontent.com/48202736/105444938-2b943780-5cb2-11eb-9947-df2dcf6cb9d4.png)
![image](https://user-images.githubusercontent.com/48202736/105444942-2f27be80-5cb2-11eb-858c-a23f594829c2.png)


- <mark style='background-color: #dcffe4'> Newton’s Method </mark>

![image](https://user-images.githubusercontent.com/48202736/105444950-3353dc00-5cb2-11eb-8f62-291d69813e45.png)
![image](https://user-images.githubusercontent.com/48202736/105444958-35b63600-5cb2-11eb-8cd4-939b22e844f2.png)
![image](https://user-images.githubusercontent.com/48202736/105444964-38b12680-5cb2-11eb-834d-b3c6475a09b7.png)

- <mark style='background-color: #dcffe4'> Optimization for Logistic Regression </mark>

![image](https://user-images.githubusercontent.com/48202736/105444986-41a1f800-5cb2-11eb-97fe-d287609a3a77.png)


![image](https://user-images.githubusercontent.com/48202736/105444997-48c90600-5cb2-11eb-8445-1b32c228bdc9.png)



- <mark style='background-color: #fff5b1'> Intuitive Animation for Logistic Regression (Binary Classification) </mark>

Logistic Regression으로 분류 문제를 푼 경우 최적화를 통해 구한 $$\phi$$는 아래의 직선을 나타냅니다.


여기서 0.5의 확률을 나타내는 지점에 선을 그으면 그게 곧 Decision Boundary가 되는 것입니다. 

![image](https://user-images.githubusercontent.com/48202736/105623179-11e22400-5e5b-11eb-9ffd-173549eb6785.png)

사실 이 그림은 3D로 봐야 와닿습니다. (입력 데이터 x 2차원, 정답 y 1차원) 

![image](https://user-images.githubusercontent.com/48202736/105623182-14dd1480-5e5b-11eb-9512-121dc3549fef.png)

Logistic Regression이 학습되는 과정을 애니메이션으로 재생하면 아래와 같습니다. 

![logistic_regression_animation](https://user-images.githubusercontent.com/48202736/105623202-405fff00-5e5b-11eb-8831-bb4748282789.gif)

(출처 : [link](https://medium.com/analytics-vidhya/ml-from-scratch-logistic-regression-gradient-descent-63b6beb1664c))


- <mark style='background-color: #fff5b1'> ML vs MAP for Classification </mark>

asd

- <mark style='background-color: #fff5b1'> Multiclass Logistic Regression </mark>

여러개의 클래스를 분류하는 문제의 경우 이진 클래스를 분류하는 경우와 크게 다르지 않습니다. 다만 출력값을 Bernoulli 분포가 아닌 Categorical 분포로 모델링 하면 됩니다.

다시 한 번 remind 하자면, Categorical 분포는 아래와 같이 생겼습니다. (그림은 클래스가 5개인 경우에 해당)

![image](https://user-images.githubusercontent.com/48202736/105621216-96777700-5e48-11eb-99c4-400cf91c6405.png)

이럴경우 $$likelihood$$는 수식으로는 다음과 같이 쓸 수 있습니다.

<center>$$ Pr(w|x) = Cat_w[\lambda[x]] $$</center>

여기서 $$\lambda$$는 전체 합이 합이 1이 되는 각 class들에 대한 확률 값을 나타내는 벡터이고 수식으로 나타내면 다음과 같습니다.

<center>$$ \lambda_n = softmax_n[a_1,a_2,...,a_N] = \frac{exp[a_N]}{sum_{m=1}^{N}exp[a_m]} $$</center>

```
sigmoid 함수가 입력 벡터를 0~1 사이의 값으로 매핑해주듯, softmax 함수는 입력 벡터를 마찬가지로 0~1 사이의 값으로 매핑해주지만, 전체 클래스의 합이 1이 되도록 해줍니다. 
```

<center>$$ ex) \space \lambda_1 + \lambda_2 + \lambda_3 = \frac{exp[a_1]}{sum_{m=1}^{3}exp[a_m]} + \frac{exp[a_2]}{sum_{m=1}^{3}exp[a_m]} + \frac{exp[a_3]}{sum_{m=1}^{3}exp[a_m]} $$</center>

<center>$$ \lambda_1 + \lambda_2 + \lambda_3 = \frac{exp[a_1]+exp[a_2]+exp[a_3]}{sum_{m=1}^{3}exp[a_m]} = 1 $$</center>

우리가 추정하고자 하는 파라메터는 n개의 벡터들 $$\theta_n$$ 입니다.

<center>$$ a_1 = \theta_{1}^{T} x $$</center>

<center>$$ a_2 = \theta_{2}^{T} x $$</center>

<center>$$ a_n = \theta_{n}^{T} x $$</center>

이렇게 모델링해서 최적 해를 구하게 되면 각각의 클래스에 대한 Decision Boundary를 구할 수 있게 된다고 생각할 수 있습니다. 

![image](https://user-images.githubusercontent.com/48202736/105445180-9e9dae00-5cb2-11eb-96cc-e8ac1453fee7.png)

마찬가지로 해를 구하기 위해서는 $$likelihood$$에 $$log$$를 취한 $$log \space likelihood$$를 최대화 하면 됩니다.


이는 마찬가지로 Closed-form Solution이 존재하지 않기 때문에, Iterative Non-Linear Optimization을 하면 구할 수 있습니다. 



- <mark style='background-color: #ffdce0'> Modeling Bernoulli Distribution over Model Output VS Binary Cross Entropy  </mark>

이 글을 통해 처음 Logistic Regression 혹은 분류 문제를 접하신 분들이 아니라면, 아마 데이터를 클래스 2개로 분류하는 경우인, 이진 분류 문제를 풀 때 Loss Function으로 (Binary) Cross Entropy 를 쓴다는걸 아실겁니다. 


하지만 이는 이번 글에서 설명한 Bernoulli 분포로 출력값을 모델링하고 이것의 $$likelihood$$를 구한다음 log를 취해 ML문제를 푸는것과 다르지 않습니다. 아니, 같습니다.


다시 생각해보면 우리가 구했던, $$log \space likelihood$$는 아래와 같았습니다.

<center>$$ Pr(w|X,\phi) = \prod_{i=1}^{I} \lambda^{w_i}(1-\lambda)^{1-w_i} $$</center>

<center>$$ Pr(w|X,\phi) = \prod_{i=1}^{I} (sig[a])^{w_i}(1-sig[a])^{1-w_i} $$</center>

<center>$$ log \space likelihood = \sum_{i=1}^{I} w_i log[sig[a]] + (1-w_i)log[1-sig[a]] $$</center>

(Binary) Cross Entropy의 경우는 어떨까요? 


마찬가지로 activation function이 sigmoid일때 아래와 같이 쓸 수 있습니다.

<center>$$ BCE \space Loss = \sum_{i=1}^{I} w_i log[sig[a]] + (1-w_i)log[1-sig[a]] $$</center>

흔히들 쓰는 notation으로 정답이 $$t_i$$이고 모델의 예측 값이 $$y_i$$인 경우로 다시 쓰면 아래와 같습니다.

<center>$$ BCE \space Loss = \sum_{i=1}^{I} t_i log (y_i) + (1-t_i)log(1-y_i) $$</center>


어떤가요?? 한눈에 봐도 알 수 있듯이 Bernoulli 분포로 모델링한 $$likelihood$$를 maximize하는것이 곧 Binary Cross Entropy(BCE) Loss를 minimize 하는 것과 완벽하게 동치임을 알 수 있습니다.


(크로스 엔트로피는 정보이론 (+ KL divergence) , 확률분포 모델링 등 다양한 관점에서 해석을 할 수 있으니 다른 관점도 생각해보시길 바랍니다.) 


이는 회귀문제에서 잘 알려진 Mean Squared Error (MSE) Loss를 minimize하는 것이 출력 값을 Gaussian 분포로 모델링한 $$likelihood$$를 maximize하는 것과 수식적으로 완벽하게 일치하는것과 같습니다. 


마찬가지로, Categorical 분포로 모델링한 $$log \space likelihood$$를 최대화 하는 것은 Cross Entropy(CE) Loss를 최소화 하는것과 동일합니다. 


<center>$$ likelihood = \prod_{i=1}^{I} Cat_{w_i}[\lambda[x_i]] $$</center>

<center>$$ CE \space Loss = \sum_{i=1}^{I} - t_i log (y_i) $$</center>

- <mark style='background-color: #ffdce0'> Classification의 또다른 관점 </mark>

앞서 배웠던 Classification들은 그게 이진 클래스 분류던, 다중 클래스 분류던 동일한 과정을 겪었는데, 이는 입력값과 클래스를 구분하는 Decision boundary가 될 파라메터와의 내적을 하는 것이었습니다.

아래의 그림을 볼까요, 우리는 개,고양이,배 ... 등 다양한 클래스의 이미지가 포함된 데이터셋을 분류하는 task인 다중 분류 문제를 풀고 싶습니다.

아래는 이미지 분류에서 유명한 딥러닝 모델인 ResNet을 사용한 것인데요.

데이터가 $$ Batch \space size \times Width \times Height $$ 의 3차원 행렬인(채널을 무시해버렸네요 1이나 없는걸로 하겠습니다...) Tensor 모양으로 되어있고 이게 네트워크로 들어간다고 생각하겠습니다. 

예시가 딥러닝이라 생소하실 수도 있는데, 딥러닝이나 머신러닝이나 결국 우리가 풀고자하는 문제의 결과값을 내주는 어떤 최상의 $$y=f(x)$$ 가 있다고 생각하고, 주어진 데이터를 통해서 이 oracle function의 approximate function을 찾는 문제인것은 똑같으니 걱정 안하셔도 됩니다.

![decision2](https://user-images.githubusercontent.com/48202736/105625959-3c8aa780-5e70-11eb-8779-adf9c1176a86.png)

이미지가 들어가서 유의미한 특징값들을 추출해주는 단계를 거치고 마지막에 $$Batch \space  size \times hidden \space size$$ 형태가 됩니다. 

(이 때부터 다시 생각해봅시다. 지금 마지막 단계가 그냥 순수 입력값이라고 생각하고 여기에 클래스를 구분지어 줄 파라메터를 곱한 뒤 softmax함수를 취하고 등등... 앞서 다중 분류를 할 때와 똑같이 생각하면 됩니다.) 

아까는 이 클래스를 구분지어줄 파라메터들(Categorical Distribution의 여러 카테고리에 해당하는)과 입력 값을 곱하는 것(벡터간 내적하는 것)이 각각의 클래스의 Decision Boundary들을 아래와 같이 그어준다고 생각했는데,

![image](https://user-images.githubusercontent.com/48202736/105445180-9e9dae00-5cb2-11eb-96cc-e8ac1453fee7.png)

이는 다르게 생각해보면 각 클래스에 해당하는 어떤 벡터가 존재하고, 입력 이미지인 입력 벡터와 이 클래스들를 표현하는 벡터들과의 벡터간 유사도들을 전부 계산해내는 것이라고 생각할 수도 있습니다.

위의 그림에서는 10개의 클래스가 존재합니다. 한 이미지에 대해서 10개의 클래스를 나타내는 벡터들과 각각 내적을 통해 내적 값(scalar)들을 10개 계산할 수 있는데 (10차원이 됨) 이들 중에서 가장 큰 값으로 이미지가 분류되는 것이죠. 이는 즉 "이 이미지는 '개'를 대표하는? 표현하는 class vector와 가장 유사했다, 그래서 '개'다"라는 결론을 내줍니다.

![image](https://user-images.githubusercontent.com/48202736/105625681-22e86080-5e6e-11eb-9979-e53aee737fad.png)

이렇게 분류 문제를 분류할 클래스 벡터들과의 내적으로 생각하게 되면, 내적을 $$\cos{\theta}$$에 재밌는 짓을 해서 클래스를 구분짓는 경계에 다양한 variation을 줄 수 있습니다. ( Large Margine Loss, LDAM Loss 등) 




- <mark style='background-color: #fff5b1'> References </mark>

1. [Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.](http://www.computervisionmodels.com/)
