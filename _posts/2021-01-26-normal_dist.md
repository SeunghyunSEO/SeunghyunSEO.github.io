---
title: Gaussian distribution (a.k.a normal distribution)
categories: Probability&Statistics
tag: 

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> Why Gaussian Distribution? </mark>


- <mark style='background-color: #fff5b1'> Univariate Gaussian Distribution </mark>

가우시안(Gaussian) 분포 혹은 정규(Normal) 분포는 변수가 1개냐 2개냐 ... 여러개냐에 따라서 (일변량)Univariate 분포 혹은 이변량(Bivariate), 다변량(Multivariate) 분포로 나눠 생각할 수 있습니다.

Univariate Gaussian Distribution의 모양은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/48202736/106379093-fd6ad200-63ec-11eb-9b6f-f8ad3b5448c1.png)

일변량 정규 분포는 1개의 연속적인 변수에 대한 확률 값을 뱉는 함수가 되고,

수식으로 나타내면 아래와 같이 쓸 수 있습니다.

<center>$$ Pr(x) = Norm_x[\mu,\sigma^2] $$</center>

<center>$$ Pr(x) = \frac{1}{\sqrt{w\pi\sigma^2}}exp[-\frac{1}{2} (x-\mu)^2 / \sigma^2 ] $$</center>

데이터 x를 표현하는 위의 분포의 파라메터는 평균(mean, $$\mu$$), 분산(variance, $$\sigma^2$$) 두가지 뿐입니다.

> $$\mu$$ : 평균(mean) <br>
> $$\sigma^2$$ : 분산(variance) ($$\sigma^2 > 0$$) <br>

- <mark style='background-color: #fff5b1'> Multivariate Gaussian Distribution </mark>

이제 조금 변수의 개수를 늘려서 생각해볼까요? 다변량 정규분포의 식은 아래와 같습니다.

<center>$$ Pr(x) = \frac{1}{ (2\pi)^{D/2} \left | {\Sigma}^{1/2} \right | }exp[-\frac{1}{2} (x-\mu)^T {Sigma}^{-1} (x-\mu) ] $$</center>

이를 간단하게 아래와 같이 표현하곤 합니다.

<center>$$ Pr(x) = Norm_x[\mu,\Sigma] $$</center>

다변량 정규분포는 역시 두 가지 파라메터로 주어진 분포를 표현합니다.

> $$\mu$$ : mean 벡터 (a vector containing mean position) <br>
> $$\Sigma$$ : 대칭이며 양의 정부호 행렬인, 공분산 행렬 (a symmetric “positive definite” covariance matrix) <br>

여기서 Positive definite의 정의는 아래와 같습니다.

> Positive definite: $$z^T {Sigma} z$$ is positive for any real is positive for any real. <br>

다변량 정규 분포는 변수의 개수가 2개 3개 ... 무수히 많아질 수 있지만

우리가 일반적으로 visualize하는데 한계가 존재하니 변수가 2개인 경우에 대해서 생각해봅시다. ($$x=[x1, x2]$$인 경우)

- <mark style='background-color: #dcffe4'> Bivariate Gaussian Distribution </mark>

변수가 2개인 경우인 이변량 정규 분포는 아래와 같이 생겼습니다.

![image](https://user-images.githubusercontent.com/48202736/106379277-1b850200-63ee-11eb-85aa-aceece871413.png)

이를 수식으로 나타내면 tmp.

- <mark style='background-color: #dcffe4'> Covariance Matrix </mark>

![image](https://user-images.githubusercontent.com/48202736/106379122-1d01fa80-63ed-11eb-964c-e9ed83014c02.png)

![image](https://user-images.githubusercontent.com/48202736/106379117-1a9fa080-63ed-11eb-9dd2-cc24ca2be8b3.png)

![image](https://user-images.githubusercontent.com/48202736/106379146-4458c780-63ed-11eb-8342-1eae6d8770d2.png)

![image](https://user-images.githubusercontent.com/48202736/106379155-4b7fd580-63ed-11eb-9f4c-0f4857763616.png)

![image](https://user-images.githubusercontent.com/48202736/106379156-4f135c80-63ed-11eb-941c-871ec0ffa780.png)


- <mark style='background-color: #fff5b1'> Conditional Gaussian Distribution </mark>

![image](https://user-images.githubusercontent.com/48202736/106379157-520e4d00-63ed-11eb-91f3-957b610e1eb1.png)

- <mark style='background-color: #fff5b1'> Marginal Gaussian Distribution </mark>

![image](https://user-images.githubusercontent.com/48202736/106379160-55093d80-63ed-11eb-9da8-4cdbac065b18.png)
![image](https://user-images.githubusercontent.com/48202736/106379163-58042e00-63ed-11eb-98de-0b82c005de7c.png)

![image](https://user-images.githubusercontent.com/48202736/106379164-5a668800-63ed-11eb-993b-ea09b72ac61a.png)
![image](https://user-images.githubusercontent.com/48202736/106379166-5c304b80-63ed-11eb-8e3c-761669998966.png)


- <mark style='background-color: #fff5b1'> Conjugate Distribution  </mark>

- <mark style='background-color: #dcffe4'> Inverse Gamma  </mark>

![image](https://user-images.githubusercontent.com/48202736/106379109-0a87c100-63ed-11eb-8e80-c8d642f20d22.png)
![image](https://user-images.githubusercontent.com/48202736/106379111-0c518480-63ed-11eb-8269-26244ea6bfea.png)

- <mark style='background-color: #dcffe4'> Normal X Nomral  </mark>

![image](https://user-images.githubusercontent.com/48202736/106379170-5fc3d280-63ed-11eb-95d1-7e2d91119b90.png)
