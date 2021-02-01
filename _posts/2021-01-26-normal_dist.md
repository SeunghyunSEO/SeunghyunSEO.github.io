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

Univariate Gaussian Distribution의 모양과 수식은 아래와 같습니다.

![normal1](https://user-images.githubusercontent.com/48202736/106457506-6aa46300-64d2-11eb-849c-3f76dac4ca70.png)
{: style="width: 60%;" class="center"}

![normal2](https://user-images.githubusercontent.com/48202736/106457512-6bd59000-64d2-11eb-8fc3-b12bdd89be6a.png)
{: style="width: 60%;" class="center"}


일변량 정규 분포는 1개의 연속적인 변수에 대한 확률 값을 뱉는 함수가 되고,

수식으로 나타내면 아래와 같이 쓸 수 있습니다.

<center>$$ Pr(x) = \frac{1}{\sqrt{w\pi\sigma^2}}exp[-\frac{1}{2} (x-\mu)^2 / \sigma^2 ] $$</center>

데이터 x를 표현하는 위의 분포의 파라메터는 평균(mean, $$\mu$$), 분산(variance, $$\sigma^2$$) 두가지 뿐입니다.

> $$\mu$$ : 평균(mean) <br>
> $$\sigma^2$$ : 분산(variance) ($$\sigma^2 > 0$$) <br>

- <mark style='background-color: #fff5b1'> Multivariate Gaussian Distribution </mark>

이제 조금 변수의 개수를 늘려서 생각해볼까요? 다변량 정규분포의 식은 아래와 같습니다.

![multivariate1](https://user-images.githubusercontent.com/48202736/106457529-709a4400-64d2-11eb-8f09-9a8d18343e00.png)
{: style="width: 60%;" class="center"}

<center>$$ Pr(x) = \frac{1}{ (2\pi)^{D/2} \left | {\Sigma}^{1/2} \right | }exp[-\frac{1}{2} (x-\mu)^T {Sigma}^{-1} (x-\mu) ] $$</center>

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

![cov1](https://user-images.githubusercontent.com/48202736/106457537-742dcb00-64d2-11eb-9418-ad604f5bba3b.png)
{: style="width: 80%;" class="center"}

![cov2](https://user-images.githubusercontent.com/48202736/106457539-755ef800-64d2-11eb-96d1-4acacd7edf93.png)
{: style="width: 80%;" class="center"}

![cov3](https://user-images.githubusercontent.com/48202736/106457544-76902500-64d2-11eb-8493-7685385ac389.png)
{: style="width: 60%;" class="center"}

![cov4](https://user-images.githubusercontent.com/48202736/106457546-77c15200-64d2-11eb-8f11-8b9e20de6d86.png)
{: style="width: 60%;" class="center"}

![cov5](https://user-images.githubusercontent.com/48202736/106457553-7859e880-64d2-11eb-88e7-d3888e8f43da.png)
{: style="width: 60%;" class="center"}

![cov6](https://user-images.githubusercontent.com/48202736/106457557-798b1580-64d2-11eb-9a74-9bb91d598ac3.png)

![cov7](https://user-images.githubusercontent.com/48202736/106457562-7b54d900-64d2-11eb-840d-c679d9fc872a.png)
{: style="width: 60%;" class="center"}



- <mark style='background-color: #fff5b1'> Conditional Gaussian Distribution </mark>


- <mark style='background-color: #fff5b1'> Marginal Gaussian Distribution </mark>


- <mark style='background-color: #fff5b1'> Conjugate Distribution  </mark>

- <mark style='background-color: #dcffe4'> Inverse Gamma  </mark>


- <mark style='background-color: #dcffe4'> Normal X Nomral  </mark>
