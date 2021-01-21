---
title: Regression
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

우리가 찾고싶은 것은 전체 데이터셋에 대한 likelihood이다. 

이는 각각의 분포를 전부 곱한것과 같기 때문에 아래와 같이 쓸 수 있다.

<center>$$ Pr(w \mid X) = Norm_{w}[X^T \phi, \sigma^2I] $$</center>

<center>$$ where X = [x_1,x_2, ... x_I] w=[w_1,w_2,...,w_I]^T $$<center>

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
