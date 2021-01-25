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
