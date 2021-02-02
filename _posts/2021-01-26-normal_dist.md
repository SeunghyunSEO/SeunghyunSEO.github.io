---
title: Gaussian distribution (a.k.a normal distribution)
categories: Probability&Statistics
tag: 

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> Why Gaussian Distribution? </mark>

왜 우리는 가우시안 분포(정규 분포라고도 알려져 있음)에 대해서 잘 알아야할까요? 

![normal1](https://user-images.githubusercontent.com/48202736/106457506-6aa46300-64d2-11eb-849c-3f76dac4ca70.png)
{: style="width: 60%;" class="center"}

정규분포는 일반적으로 통계학에서 가장 많이 사용되는 연속확률분포로, 회귀분석 등 대부분의 통계 기법에 사용됩니다.

우선 이름이 정규분포가 된 이유는 아래와 같다고 합니다.

```
라플라스(Pierre S. Laplace)와 가우스(Karl F. Gauss)가 이 분포를 정립할 당시에는 모든 자료의 분포가 이 종형 곡선과 가까운 형태를 지니고 있어야 정상이고, 
그렇지 않은 경우에는 자료수집 과정에 이상이 있다고 생각했기 때문에 이 분포에 정규(Normal)이라는 이름을 붙이게 되었다는 것이다. 
```
(출처 : [link](https://blog.naver.com/definitice/220950767553))

일반적으로 '자연에서 발생하는 수많은 현상들이 가우시안 분포를 따른다(성인의 신장 분포, 혈압 분포, 지능지수 등).', '(수학적으로)다루기 쉽다'(=통계 분석이 쉽다)를 이유로 들어 정규분포가 중요하다고 합니다.

또한 우리가 반복적인 실험통해 샘플링을 거듭해 표본이 커질수록 결국 확률분포는 정규분포를 따르게 된다는 것도 가우시안 분포의 중요성을 설명합니다.(Central Limit Theorm)


또한 우리가 어떤 가우시안 분포를 따르지 않는 데이터의 분포를 가지고 있다고 하더라도 여기에 적절한 제곱근을 취하거나 log를 취해주면 이를 가우시안으로 변형(transform)해서 쓸 수도 있기 때문에, (결국 모든 분포는 가우시안으로...?)

실험적으로나 이론적으로 가우시안 분포를 잘 아는것은 중요하다고 할 수 있습니다.


혹은 반 농담으로 어떤 교수님은 다음과 같은 이유로 통계학에서 가우시안 분포를 쓴다고도 얘기합니다. 

```
수학자 : 공학자들이 실험적으로 증명했던데?
공학자 : 수학자들이 수식적으로 증명했던데?
```

어쨌든 가우시안 분포는 중요한 의미를 지니고 있으니 이 분포에 대해서 이해도를 높히는것이 통계/머신러닝을 하는데에 있어 유용할 것 같으니 알아보도록 하. 

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

<center>$$ Pr(x) = \frac{1}{ (2\pi)^{D/2} {\left | {\Sigma} \right |}^{1/2} }exp[-\frac{1}{2} (x-\mu)^T {\Sigma}^{-1} (x-\mu) ] $$</center>

다변량 정규분포는 역시 두 가지 파라메터로 주어진 분포를 표현합니다.

> $$\mu$$ : $$D$$차원의 mean 벡터 (a vector containing mean position) <br>
> $$\Sigma$$ : 대칭이며 $$D \times D$$차원의 양의 정부호 행렬인 공분산 행렬 (a symmetric “positive definite” covariance matrix) <br>

(그리고 $${\mid {\Sigma} \mid}$$는 공분산 행렬의 행렬식입니다.)


여기서 Positive definite의 정의는 아래와 같습니다.

> Positive definite: $$z^T {\Sigma} z$$ is positive for any real is positive for any real. <br>

다변량 정규 분포는 변수의 개수가 2개 3개 ... 무수히 많아질 수 있지만

우리가 일반적으로 visualize하는데 한계가 존재하니 변수가 2개인 경우에 대해서 생각해봅시다. ($$x=[x1, x2]$$인 경우)

- <mark style='background-color: #dcffe4'> Bivariate Gaussian Distribution </mark>

변수가 2개인 경우인 이변량 정규 분포는 아래와 같이 생겼습니다.

![image](https://user-images.githubusercontent.com/48202736/106379277-1b850200-63ee-11eb-85aa-aceece871413.png)

이를 수식으로 나타내면 tmp.

- <mark style='background-color: #dcffe4'> Covariance Matrix </mark>

우리는 앞서 다변량 정규분포가 어떤 형태를 띄고있으며, 수식으로 어떻게 정의되어있는지 살펴봤습니다.
 
 
다변량 정규분포의 파라메터는 두가지, Mean 벡터와, 변수들간의 상관 variance를 모아둔 Covariance Matrix, 이 두가지였습니다.


이 중에서도 Covariance Matrix는 굉장히 중요한 의미를 갖기 이에 대해 한번 알아보도록 하겠습니다.


Covariance Matrix는 행렬의 모양에 따라서 아래와 같이 나눌 수 있습니다.

![cov1](https://user-images.githubusercontent.com/48202736/106457537-742dcb00-64d2-11eb-9418-ad604f5bba3b.png)
{: style="width: 80%;" class="center"}

![cov2](https://user-images.githubusercontent.com/48202736/106457539-755ef800-64d2-11eb-96d1-4acacd7edf93.png)
{: style="width: 80%;" class="center"}

변수간 상관관계를 가장 잘 표현하는, 가장 복잡한 모델이 일반적인 Covariance Matrix, 즉 Full Covariance Matrix라고 할 수 있고(e,f),


이보다 조금 표현력이 떨어지지만 대각 성분에만 variance가 존재하는 Diagonal Covariance Matrix가 (c,d)에 나타나 있습니다.


그리고 이중에서 가장 간단하다고 할 수 있는 $$\sigma^2 I$$, Spherical Covariance Matrix의 수식과 그림이 (a,b)가 됩니다. 

![cov3](https://user-images.githubusercontent.com/48202736/106457544-76902500-64d2-11eb-8493-7685385ac389.png)
{: style="width: 60%;" class="center"}

![cov4](https://user-images.githubusercontent.com/48202736/106457546-77c15200-64d2-11eb-8f11-8b9e20de6d86.png)
{: style="width: 60%;" class="center"}

![cov5](https://user-images.githubusercontent.com/48202736/106457553-7859e880-64d2-11eb-88e7-d3888e8f43da.png)

![cov6 5](https://user-images.githubusercontent.com/48202736/106458941-785ae800-64d4-11eb-8d90-bb732b8bfd37.png)
{: style="width: 60%;" class="center"}

![cov6](https://user-images.githubusercontent.com/48202736/106457557-798b1580-64d2-11eb-9a74-9bb91d598ac3.png)

![cov7](https://user-images.githubusercontent.com/48202736/106457562-7b54d900-64d2-11eb-840d-c679d9fc872a.png)
{: style="width: 60%;" class="center"}


- <mark style='background-color: #fff5b1'> Conditional Gaussian Distribution </mark>
![cond1](https://user-images.githubusercontent.com/48202736/106459113-b8ba6600-64d4-11eb-9e9e-27ea3106bd8a.png)
{: style="width: 60%;" class="center"}

![cond2](https://user-images.githubusercontent.com/48202736/106459115-b9eb9300-64d4-11eb-9561-88beb5020302.png)
{: style="width: 60%;" class="center"}

- <mark style='background-color: #fff5b1'> Marginal Gaussian Distribution </mark>
![marg1](https://user-images.githubusercontent.com/48202736/106459120-bb1cc000-64d4-11eb-8947-9be05ce6ef70.png)
{: style="width: 60%;" class="center"}

- <mark style='background-color: #fff5b1'> Conjugate Distribution  </mark>

- <mark style='background-color: #dcffe4'> Normar Inverse Gamma X Univariate Normal  </mark>
![normal1](https://user-images.githubusercontent.com/48202736/106457506-6aa46300-64d2-11eb-849c-3f76dac4ca70.png)
{: style="width: 60%;" class="center"}

![normal2](https://user-images.githubusercontent.com/48202736/106457512-6bd59000-64d2-11eb-8fc3-b12bdd89be6a.png)
{: style="width: 60%;" class="center"}

![inverse1](https://user-images.githubusercontent.com/48202736/106459196-da1b5200-64d4-11eb-9cbb-a2dd3dc5a3ab.png)

![inverse2](https://user-images.githubusercontent.com/48202736/106459199-db4c7f00-64d4-11eb-9738-f1254d5d8a05.png)
{: style="width: 60%;" class="center"}

- <mark style='background-color: #dcffe4'> Normal X Nomral  </mark>
![joint1](https://user-images.githubusercontent.com/48202736/106459122-bd7f1a00-64d4-11eb-87ee-82c04b877996.png)
{: style="width: 60%;" class="center"}

- <mark style='background-color: #dcffe4'> Normal inverse Wishart X Multivariate Normal  </mark>
![wishart1](https://user-images.githubusercontent.com/48202736/106459372-1b136680-64d5-11eb-9c1e-dd81432ee24c.png)
{: style="width: 60%;" class="center"}

![wishart2](https://user-images.githubusercontent.com/48202736/106459378-1c449380-64d5-11eb-93d6-f1db1394458a.png)

![wishart3](https://user-images.githubusercontent.com/48202736/106459380-1d75c080-64d5-11eb-9c75-4fb0d25752a8.png)
{: style="width: 60%;" class="center"}



- <mark style='background-color: #fff5b1'> Maximum Likelihood for the Gaussian </mark>

- <mark style='background-color: #dcffe4'> Sequential Estimation  </mark>
