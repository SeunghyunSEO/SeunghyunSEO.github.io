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

또한 우리가 반복적인 실험통해 샘플링을 거듭해 표본이 커질수록 그 표본의 평균에 대한 확률분포가 정규분포를 따른다거나 하는 내용들 가우시안 분포의 중요성을 설명합니다.(Central Limit Theorm)


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


- <mark style='background-color: #dcffe4'> Covariance Matrix </mark>

우리는 앞서 다변량 정규분포가 어떤 형태를 띄고있으며, 수식으로 어떻게 정의되어있는지 살펴봤습니다.
 
 
다변량 정규분포의 파라메터는 두가지, Mean 벡터와, 변수들간의 상관 variance를 모아둔 Covariance Matrix, 이 두가지였습니다.


이 중에서도 Covariance Matrix는 굉장히 중요한 의미를 갖기 이에 대해 한번 알아보도록 하겠습니다.


Covariance Matrix는 행렬의 모양에 따라서 아래와 같이 나눌 수 있습니다.

![cov1](https://user-images.githubusercontent.com/48202736/106457537-742dcb00-64d2-11eb-9418-ad604f5bba3b.png)
{: style="width: 80%;" class="center"}

![cov2](https://user-images.githubusercontent.com/48202736/106457539-755ef800-64d2-11eb-96d1-4acacd7edf93.png)
{: style="width: 80%;" class="center"}

1.변수간 상관관계를 가장 잘 표현하는, 가장 복잡한 모델이 일반적인 Covariance Matrix, 즉 Full Covariance Matrix라고 할 수 있고(e,f),


2.이보다 조금 표현력이 떨어지지만 대각 성분에만 variance가 존재하는 Diagonal Covariance Matrix가 (c,d)에 나타나 있습니다.


3.그리고 이중에서 가장 간단하다고 할 수 있는 $$\sigma^2 I$$, Spherical Covariance Matrix의 수식과 그림이 (a,b)가 됩니다. 

- <mark style='background-color: #dcffe4'> Property of Covariance Matrix </mark>

Covariance Matrix가 어떤 모양을 가지느냐에 따라 정규 분포가 가지는 특성에 대해 살펴볼까요?


우선 Full Covariance Matrix를 제외한 Diagonal Covariance Matrix를 따르는 두 변수 $$x1,x2$$의 이변량 정규분포는

아래와 같은 수식으로 두 변수의 독립적인 분포(정규 분포)의 곱으로 나타낼 수 있습니다. (독립성)

![cov5](https://user-images.githubusercontent.com/48202736/106457553-7859e880-64d2-11eb-88e7-d3888e8f43da.png)

그리고 Full Covariance는 아래의 그림과 같이 어떤 Diagonal Covariance와 회전 행렬(Rotation Matrix)의 곱으로 Decomposition할 수도 있습니다.  

![cov6 5](https://user-images.githubusercontent.com/48202736/106458941-785ae800-64d4-11eb-8d90-bb732b8bfd37.png)
{: style="width: 60%;" class="center"}

(여기서 분홍색이 Full Covariance고, 녹색이 Diagonal Covariance가 됩니다.)

이는 수식으로 나타내면 다음과 같습니다.

![cov6](https://user-images.githubusercontent.com/48202736/106457557-798b1580-64d2-11eb-9a74-9bb91d598ac3.png)

또한 우리가 예를들어 $$Pr(x) = Norm_x[\mu,\Sigma]$$라는 분포를 가지고 있을 때, $$y=Ax+b$$라는 선형 변환을 할 경우

결과 $$y$$의 분포 또한 가우시안 분포를 따르게 됩니다.

<center>$$Pr(x) = Norm_x[\mu,\Sigma]$$</center>
<center>$$y=Ax+b$$</center>
<center>$$Pr(y) = Norm_y[A\mu + b, A^T \Sigma A]$$</center>

(Can be used to generate data from arbitrary Gaussians from standard one)

![cov7](https://user-images.githubusercontent.com/48202736/106457562-7b54d900-64d2-11eb-840d-c679d9fc872a.png)
{: style="width: 60%;" class="center"}


- <mark style='background-color: #fff5b1'> Marginal Gaussian Distribution </mark>

가우시안 분포의 주변 분포(Marginal distribution) 또한 가우시안 분포가 되는 특성을 가지고 있습니다.

![marg1](https://user-images.githubusercontent.com/48202736/106459120-bb1cc000-64d4-11eb-8947-9be05ce6ef70.png)
{: style="width: 60%;" class="center"}



- <mark style='background-color: #fff5b1'> Conditional Gaussian Distribution </mark>

그리고 가우시안 분포의 조건부 분포(Conditional distribution) 또한 가우시안 분포가 되는 특성이 있습니다.

![cond1](https://user-images.githubusercontent.com/48202736/106459113-b8ba6600-64d4-11eb-9e9e-27ea3106bd8a.png)
{: style="width: 80%;" class="center"}

![cond2](https://user-images.githubusercontent.com/48202736/106459115-b9eb9300-64d4-11eb-9561-88beb5020302.png)
{: style="width: 80%;" class="center"}





- <mark style='background-color: #fff5b1'> Conjugate Distribution  </mark>

이번에는 가우시안 분포의 공액 분포(Conjugate Distribution)에 대해서 알아보도록 하겠습니다.

왜 이 특성이 중요하냐면, 우리가 posterior를 계산할 때 likelihood x prior의 꼴로 계산하게 되는데 (적분을 해서 evidence로 나눠주기 까지 해야합니다)

여기서 Conjugate 관계인 likelihood와 prior를 고르면 계산이 훨씬 수월하기 때문입니다.


우리는 우선 likelihood가 가우시안인 경우를 가정하고 이와 conjugate관계인 prior들이 무엇이 있는지에 대해서 알아보도록 하겠습니다.

(이와 관련된 수식들은 옆의 링크에서 찾아볼 수 있습니다 : [link](https://en.wikipedia.org/wiki/Conjugate_prior#cite_note-murphy-10))

- <mark style='background-color: #dcffe4'> Normal Inverse Gamma X Univariate Normal  </mark>

1. Univariate Normal Distribution과 Normal Inverse Gamma는 conjugate 관계입니다.

이 두 분포를 곱하게 되면 하나의 새로운 (constant) x Normal Inverse Gamma 분포가 됩니다. (수식은 생략하겠습니다.)


아래는 두 분포의 모양입니다.

![normal1](https://user-images.githubusercontent.com/48202736/106457506-6aa46300-64d2-11eb-849c-3f76dac4ca70.png)
{: style="width: 60%;" class="center"}

![normal2](https://user-images.githubusercontent.com/48202736/106457512-6bd59000-64d2-11eb-8fc3-b12bdd89be6a.png)
{: style="width: 60%;" class="center"}

![inverse1](https://user-images.githubusercontent.com/48202736/106459196-da1b5200-64d4-11eb-9cbb-a2dd3dc5a3ab.png)

![inverse2](https://user-images.githubusercontent.com/48202736/106459199-db4c7f00-64d4-11eb-9738-f1254d5d8a05.png)
{: style="width: 60%;" class="center"}


- <mark style='background-color: #dcffe4'> Normal X Nomral  </mark>

2. Normal Distribution과 Normal Distribution도 두개를 곱하면 새로운 Normal Distribution이 되는 conjugate 관계입니다. (수식은 생략하겠습니다.)

(신기한 점은 마찬가지로 Normal x Normal = (constant) * Normal 이 되는데 이 때의 constant 또한 Normal 의 형태를 갖게 된다는 겁니다.)

아래는 두 분포의 곱을 나타낸 그림입니다..

![joint1](https://user-images.githubusercontent.com/48202736/106459122-bd7f1a00-64d4-11eb-87ee-82c04b877996.png)
{: style="width: 60%;" class="center"}

- <mark style='background-color: #dcffe4'> Normal inverse Wishart X Multivariate Normal  </mark>

3. Multivariate Normal Distribution과 Normal inverse Wishart Distribution도 Conjugate 관계입니다.

마찬가지로 수식을 계산하면 새로운 Normal inverse Wishart Distribution로 딱 맞아 떨어지게 돼서 이후 posterior를 계산하기 쉬워집니다. (마찬가지로 수식은 생략하겠습니다.)


아래는 wishart 분포의 모양입니다.

![wishart1](https://user-images.githubusercontent.com/48202736/106459372-1b136680-64d5-11eb-9c1e-dd81432ee24c.png)
{: style="width: 60%;" class="center"}

![wishart2](https://user-images.githubusercontent.com/48202736/106459378-1c449380-64d5-11eb-93d6-f1db1394458a.png)

![wishart3](https://user-images.githubusercontent.com/48202736/106459380-1d75c080-64d5-11eb-9c75-4fb0d25752a8.png)
{: style="width: 60%;" class="center"}











- <mark style='background-color: #fff5b1'> 수식적으로 접근하는 Gaussian Distribution </mark>

다음의 수식들은 PRML 책과 이 책에 대해  [깃허브 페이지](http://norman3.github.io/prml/docs/chapter02/3_1)를 참고했습니다.

아래의 수식은 일변량(Univariate Gaussian Distribution) 정규분포입니다.

$$N(x|\mu, \sigma^2) = \dfrac{1}{(2\pi\sigma^2)^{1/2}}\exp\left\{-\frac{1}{2\sigma^2}(x-\mu)^2\right\} \qquad{(2.42)}$$

그리고 아래의 수식은 다변량(Multivariate Gaussian Distribution) 정규분포입니다.

$$N({\bf x}|{\pmb \mu}, {\bf \Sigma}) = \dfrac{1}{(2\pi)^{D/2}|{\bf \Sigma}|^{1/2}}\exp\left\{-\frac{1}{2}({\bf x}-{\pmb \mu})^T{\bf \Sigma}^{-1}({\bf x}-{\pmb \mu})\right\} \qquad{(2.43)}$$





- <mark style='background-color: #dcffe4'> Mahalanobis distance and euclidean diatance </mark>

데이터의 변수가 2개 이상인 경우, 즉 다변량 정규분포를 생각할 때 특히 아래의 수식에서 지수상에 나타나는 이차식은 특히 중요합니다.

$$N({\bf x}|{\pmb \mu}, {\bf \Sigma}) = \dfrac{1}{(2\pi)^{D/2}|{\bf \Sigma}|^{1/2}}\exp\left\{-\frac{1}{2}({\bf x}-{\pmb \mu})^T{\bf \Sigma}^{-1}({\bf x}-{\pmb \mu})\right\} \qquad{(2.43)}$$

이 아래의 식을 바로 마할라노비스 거리(Mahalanobis distance)라고 합니다.

$$\Delta^2 = ({\bf x}-{\pmb \mu})^T{\bf \Sigma}^{-1}({\bf x}-{\pmb \mu}) \qquad{(2.44)}$$

이 식이 중요한 이유는 출력 분포를 가우시안 분포로 모델링하는 경우 (일반적으로 회귀문제) maximum log-likelihood문제를 풀게 되면 자연스럽게 이 이차식이 나오게 되고,
이 때 mean값에 대해 최적화를 할 경우 이 수식이 최대가 되는 값을 따라 mean을 정하기 때문입니다.

즉 다차원 상에 뿌려진 데이터 x가 평균과 얼마나 떨어져있느냐를 재는 거리 식을 최대화 하는 거라는 거죠.

근데 그 거리를 단순히 평균과의 거리를 재지 않고, 공변량을 반영해서 거리를 재겠다는 것이 핵심입니다.

![md1](https://user-images.githubusercontent.com/48202736/107332890-144daa80-6af8-11eb-8b08-c6af1e1b7452.jpg)

$$\Delta^2 = ({\bf x}-{\pmb \mu})^T{\bf \Sigma}^{-1}({\bf x}-{\pmb \mu}) \qquad{(2.44)}$$


위의 그림에서 공변량을 반영해서 잰 Mahalanobis distance와 Euclidean distance 길이를 비교한 결과가 다르게 나타나는걸 알 수 있습니다.

(여기서 수식에서 공분산 행렬이 항등 행렬 I가 될 경우, Mahalanobis distance =  Euclidean distance 가 됩니다.)



$$  ({\bf x}-{\pmb \mu})^T{\bf \Sigma}^{-1}({\bf x}-{\pmb \mu}) = \left[ \begin{matrix}(x_1 - \mu_1) & (x_2 - \mu_2)\end{matrix} \right]   \left[  \begin{matrix}   \sigma_11 & \sigma_12 \\ \sigma_21 & \sigma_22 \\  \end{matrix} \right]   \left[ \begin{matrix}(x_1 - \mu_1) \\ (x_2 - \mu_2)\end{matrix} \right]   $$

$$ if \space {\pmb \mu} = \left[ \begin{matrix} 2 \\ 3 \end{matrix} \right] \space and \space \pmb{\Sigma}^{-1} = \left[ \begin{matrix} 1 & 0 \\ 0 & 1 \\ \end{matrix} \right]  $$ 


$$  ({\bf x}-{\pmb \mu})^T{\bf \Sigma}^{-1}({\bf x}-{\pmb \mu}) = \left[ \begin{matrix}(x_1 - 2) & (x_2 - 3)\end{matrix} \right]   \left[  \begin{matrix}   1 & 0 \\ 0 & 1 \\  \end{matrix} \right]   \left[ \begin{matrix}(x_1 - 2) \\ (x_2 - 3)\end{matrix} \right]   $$

$$ = {(x-2)}^2 + {(x-3)}^2 = r^2 $$



우리는 모든 데이터 포인트에 대해서 이 값을 더한 걸 최소화해주는 분포를 골랴아 할 것입니다.

<img width="1040" alt="md" src="https://user-images.githubusercontent.com/48202736/107332867-0f88f680-6af8-11eb-9fdf-425430b1604b.png">

위의 예시는 두 클러스터에 대해서 MD와 ED를 잰 예시입니다.











- <mark style='background-color: #dcffe4'> EVD of Covariance Matrix </mark>

![evd](https://user-images.githubusercontent.com/48202736/107395495-3834de80-6b40-11eb-932d-a5f0c9471ee0.png)

$${\bf \Sigma}{\bf u}_i = \lambda_i {\bf u}_i \qquad{(2.45)}$$



$${\bf u}_i^T{\bf u}_j=I_{ij} \qquad{(2.46)}$$

$$I_{ij}=\left\{\begin{array}{lr}1 & if\;i=j\\0 & otherwise\end{array}\right. \qquad{(2.47)}$$


$${\bf \Sigma}=\sum_{i=1}^{D}{\lambda_i}{\bf u}_i{\bf u}_i^T \qquad{(2.48)}$$


$${\bf \Sigma}^{-1}=\sum_{i=1}^{D}\dfrac{1}{\lambda_i}{\bf u}_i{\bf u}_i^T \qquad{(2.49)}$$


$$\Delta^2 = \sum_{i=1}^{D}\frac{y_i^2}{\lambda_i} \qquad{(2.50)}$$

$$y_i={\bf u}_i^T({\bf x}-{\pmb \mu}) \qquad{(2.51)}$$


$${\bf y} = {\bf U}({\bf x}-{\pmb \mu}) \qquad{(2.52)}$$













- <mark style='background-color: #dcffe4'> Jsacobian Matrix </mark>
    
$$J_{ij}=\dfrac{\partial x_i}{\partial y_i}=U_{ji} \qquad{(2.53)}$$

- 야코비안이 좌표 축 변환을 만들어 낼 때 어떻게 변화하는지 좀 알아야 하는데, 여기서는 간단하게 다음의 성질만을 기술해본다.
    - 아주 간단하게만 이야기하자면 공간의 선형 변환시 발생되는 부피의 변화율을확률  식에 반영하자는 것.
    
$$\int_{\bf x} f({\bf x})d{\bf x} = \int_{\bf y} f({\bf y})|{\bf J}|d{\bf y}$$

이미 앞서서 \\( {\bf y} = {\bf U}({\bf x}-{\pmb \mu}) \\) 는 확인

$${\vert}{\bf J}{\vert}^2 = {\vert}{\bf U}^T{\vert}^2 = {\vert}{\bf U}^T{\vert}\;{\vert}{\bf U}{\vert} = {\vert}{\bf U}^T{\bf U}{\vert} = {\vert}{\bf I}{\vert} = 1 \qquad{(2.54)}$$

- \\( U \\) 는 직교 행렬 \\( \|J\|=1 \\) 

$$\left|\Sigma\right|^{\frac{1}{2}}=\prod_{j=1}^{D}\lambda_j^{\frac{1}{2}} \qquad{(2.55)}$$

- \\( x \\) 축에서 \\( y \\) 축으로 전환 \\( {\bf y} = {\bf U}({\bf x}-{\pmb \mu}) \\) 을 대입하고 기타 식들을 추가하면,

$$p({\bf y}) = p(x)|{\bf J}| = \prod_{j=1}^{D}\dfrac{1}{(2\pi\lambda_j)^{1/2}}\exp\left\{-\dfrac{y_j^2}{2\lambda_j}\right\} \qquad{(2.56)}$$

- 식을 잘 살펴보면 서로 독립적인 \\( D \\) 개의 정규 분포의 확률 값이 단순 곱으로 이루어져 있다는 것을 알 수 있다.
- 고유 벡터를 이용해서 축을 변환시켜 얻은 식은 결국 차원간 서로 독립적인 정규 분포를 만들어낸다.
- 이 식을 적분

$$\int p({\bf y})d{\bf y} = \prod_{j=1}^{D} \int_{-\infty}^{\infty}\dfrac{1}{(2\pi\lambda_j)^{1/2}}\exp \left\{-\dfrac{y_j^2}{2\lambda_j}\right\}dy_i=1 \qquad{(2.57)}$$

- 역시나 확률 값이므로 각각의 차원에 대해 전구간 적분하면 크기가 1이고, 이를 \\( D \\) 차원만큼 곱해도 여전히 결과는 1이다.

- 이제 가우시안 분포의 적률(moment)을 좀 살펴보도록 하자.
    - 참고로 적률(moment)은 고전 통계학에서 사용되었던 파라미터이다.

- \\( {\bf x} \\) 축에 대해 평균값을 살펴볼 예정인데 우선 식 전개를 편하게 하기 위해 \\( {\bf z} = ({\bf x}-{\pmb \mu}) \\) 를 놓고 식을 전개

$$E[{\bf x}] = \dfrac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\int\exp\left\{-\frac{1}{2}({\bf x}-{\pmb \mu})^T{\bf \Sigma}^{-1}({\bf x}-{\pmb \mu})\right\}{\bf x}d{\bf x}\\
= \dfrac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\int\exp\left\{-\frac{1}{2}({\bf z})^T{\bf \Sigma}^{-1}({\bf z})\right\}({\bf z}+{\pmb \mu})d{\bf z} \qquad{(2.58)}$$

- 여기에 \\( ({\bf z}+{\pmb \mu}) \\) 식이 추가되어 있으므로 \\( {\pmb \mu} \\) 만큼 평행이동한 함수이다.
- 따라서 중심이 \\( {\pmb \mu} \\) 이고 좌우 대칭인 정규 함수가 만들어진다. 따라서 평균은 다음과 같다.

$$E[{\bf x}]={\pmb \mu} \qquad{(2.59)}$$

- 이제 2차 적률(second order moments)에 대해 살펴보자.

$$E[{\bf x}{\bf x}^T]=\dfrac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\int\exp\left\{-\frac{1}{2}({\bf x}-{\pmb \mu})^T{\bf \Sigma}^{-1}({\bf x}-{\pmb \mu})\right\}{\bf x}{\bf x}^Td{\bf x}\\
= \dfrac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\int\exp\left\{-\frac{1}{2}({\bf z})^T{\bf \Sigma}^{-1}({\bf z})\right\}({\bf z}+{\pmb \mu})({\bf z}+{\pmb \mu})^Td{\bf z}$$

- 여기서 \\( ({\bf z}+{\pmb \mu})({\bf z}+{\pmb \mu})^T \\) 를 전개할 수 있다.
- 이를 전개한 수식에서 \\( {\pmb \mu}{\bf z}^T \\) 와 \\( {\bf z}{\pmb \mu}^T \\) 는 서로 대칭 관계이므로 제거된다.
- \\( {\bf u}{\bf u}^T \\) 는 수식에서 상수의 역할이므로 적분 바깥 쪽으로 나오게 된다. 
- 결국 우리가 집중해야 할 요소는 \\( {\bf z}{\bf z}^T \\) 이다.
- 참고로 \\( {\bf z} \\) 는 다음과 같이 고유벡터로 표현 가능하다.
$${\bf z}=\sum_{j=1}^{D}y_j{\bf u}_j \qquad{(2.60)}$$

- 여기서 \\( y_j={\bf u}_j^T{\bf z} \\) 이다.
- 따라서 식을 다음과 같이 전개 가능하다.

$$\dfrac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\int\exp\left\{-\frac{1}{2}({\bf z})^T{\bf \Sigma}^{-1}({\bf z})\right\}{\bf z}{\bf z}^Td{\bf z}$$

$$=\dfrac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\sum_{i=1}^{D}\sum_{j=1}^{D}{\bf u}_i{\bf u}_j^T\int \exp \left\{-\sum_{k=1}^{D}\frac{y_k^2}{2\lambda_k}\right\}y_iy_jd{\bf y}\\
=\sum_{i=1}^{D}{\bf u}_i{\bf u}_j^T\lambda_i=\Sigma \qquad{(2.61)}$$

- 원 식에 대입하면 다음과 같은 결과를 얻는다.

$$E[{\bf x}{\bf x}^T]={\pmb \mu}{\pmb \mu}^T + \Sigma \qquad{(2.62)}$$

- 뭐, 원래 알고 있던 2차 적률 값이다. (평균의 제곱과 분산의 합)
- 이제 공분산(covariance) 값도 한번 구해보자.
- \\( E[{\bf x}]={\pmb \mu} \\) 이므로 어차피 동일한 결과를 얻게 된다.

$$cov[{\bf x}]=E[({\bf x}-E[{\bf x}])({\bf x}-E[{\bf x}])^T] \qquad{(2.63)}$$

$$cov[{\bf x}]=\Sigma \qquad{(2.64)}$$















- <mark style='background-color: #dcffe4'> Limitation of Gaussian Distribution </mark>

- 제약(1) : 모수(parameter)의 개수

- 제약(2) : 분포의 모양이 단봉(unimodal)의 형태만 올 수 있음

- <mark style='background-color: #dcffe4'> Conditional Gaussian distributions </mark>

- <mark style='background-color: #dcffe4'> Marginal Gaussian distributions </mark>








- <mark style='background-color: #dcffe4'> Bayes’ theorem for Gaussian variables </mark>

- 이는 가우시안 선형 모델 (linear Gaussian model) 의 한 예이다.

$$p({\bf x}) = N({\bf x}\;|\;{\pmb \mu}, \Lambda^{-1}) \qquad{(2.99)}$$

$$p({\bf y}|{\bf x}) = N({\bf y}\;|\;{\bf A} {\bf x}+{\bf b} , L^{-1}) \qquad{(2.100)}$$


 - 베이즈 이론를 활용하여,
 - \\( p({\bf z}) = p({\bf x})p({\bf y}\|{\bf x}) \\) 인 식을 \\( p({\bf x}\|{\bf y})p({\bf y}) \\) 와 같은 식으로 전개함
 - 곱의 법칙에 따라서 \\( p({\bf z}) \\) 즉, \\( p({\bf x}, {\bf y}) \\) 를 구할 수 있고,
 - \\( p({\bf y}) \\) 와 \\( p({\bf x}\|{\bf y}) \\) 도 구할 수 있다.
 - 이러면현재 분포를 \\( p({\bf x}) \\) 와 \\( p({\bf y}\|{\bf x}) \\) 를 만족하도록 만들어놓고, 최종적으로 \\( p({\bf y}) \\) 등을 만들어낸다.

- 이후 과정은 증명 과정이다.

- 우선 \\( {\bf x} \\) 와 \\( {\bf y} \\) 의 결합 확률을 \\( {\bf z} \\) 로 정의하자.

$${\bf z} = \dbinom{ {\bf x} }{ {\bf y} } \qquad{(2.101)}$$

- 이제 결합 분포에 로그를 씌운다.

$$\ln{p({\bf z})} = \ln p({\bf x}) + \ln p({\bf y})\\
= -\frac{1}{2}({\bf x}-{\pmb \mu})^T\Lambda({\bf x}-{\pmb \mu}) -\frac{1}{2}({\bf y}-{\bf A}{\bf x}-{\bf b})^T{L}({\bf y}-{\bf A}{\bf x}-{\bf b})+const \qquad{(2.102)}$$

- 여기서 `const` 영역은 \\( {\bf x} \\) 나 \\( {\bf y} \\) 와는 상관없는 텀이다.
- 그 외의 텀은 \\( {\bf z} \\) 의 요소들에 대한 이차형식(`quadratic`)의 함수이다.
    - 앞서 이런 형태의 식에 대한 가우시안 분포 여부를 확인했었다.
    - 따라서 이 분포도 가우시안 분포가 된다는 것을 알 수 있다.
    
- 어쨌거나 위의 식을 모두 전개하여 이 중 이차항만을 추려보자.
    - 알다시피 공분산을 구하기 위해서이다.

$$-\frac{1}{2}{\bf x}^T(\Lambda + {\bf A}^T\Lambda{\bf A}){\bf x} - \frac{1}{2}{\bf y}^T{\bf L}{\bf y} + \frac{1}{2}{\bf x}^T{\bf A}{\bf L}{\bf y}\\
= -\frac{1}{2}\dbinom{ {\bf x} }{ {\bf y} }^T \left(\begin{array}{cc}\Lambda+{\bf A}^T{\bf L}{\bf A} & -{\bf A}^T{\bf L}\\-{\bf L}{\bf A} & {\bf L}\end{array} \right) \dbinom{ {\bf x} }{ {\bf y} } = -\frac{1}{2}{\bf z}^T{\bf R}{\bf z} \qquad{(2.103)}$$

- 신기하게도 \\( {\bf z} \\) 에 대한 이차형식(`quadratic`) 형태의 식이 전개되었다.
- 따라서 \\( R \\) 은 정확도 행렬이 된다. (공분산의 역행렬)

$${\bf R} = \left(\begin{array}{cc}\Lambda+{\bf A}^T{\bf L}{\bf A} & -{\bf A}^T{\bf L}\\-{\bf L}{\bf A} & {\bf L}\end{array}\right) \qquad{(2.104)}$$

- 역행렬을 만드는 식을 이용하여 공분산도 구할 수 있다.

$$cov[{\bf z}]={\bf R}^{-1} = \left(\begin{array}{cc}\Lambda^{-1} & \Lambda^{-1}{\bf A}^T \\ {\bf A}\Lambda^{-1} & {\bf L}^{-1}+{\bf A}\Lambda^{-1}{\bf A}^T  \end{array}\right) \qquad{(2.105)}$$

- 복잡해보이긴 해도 구할수 없는 식은 아니다.
- 이제 일차항을 묶어 얻어진 계수와, 앞서 구한 공분산을 이용하여 평균을 구할 수 있다.
- 일차항은 다음과 같다.

$${\bf x}^T\Lambda{\pmb \mu} - {\bf x}^T{\bf A}^T{\bf L}{\bf b} + {\bf y}^T{\bf L}{\bf b} = \dbinom{ {\bf x} }{ {\bf y} }^T\dbinom{\Lambda{\pmb \mu}-{\bf A}^T{\bf L}{\bf b}}{ {\bf L}{\bf b} } \qquad{(2.106)}$$

- 따라서 평균값은 다음과 같다.

$$E[{\bf z}] = {\bf R}^{-1}\dbinom{ {\bf x} }{ {\bf y} }^T\dbinom{\Lambda{\pmb \mu}-{\bf A}^T{\bf L}{\bf b}}{ {\bf L}{\bf b} } \qquad{(2.107)}$$

- \\( {\bf R}^{-1} \\) 을 대입하여 전개해보자. 최종 식으로 다음을 얻을 수 있다.

$$E[{\bf z}] = \dbinom{ {\pmb \mu} }{ {\bf A} {\pmb \mu} - {\bf b}} \qquad{(2.108)}$$

- 각 요소의 평균이 결국 \\( {\bf z} \\) 의 평균이 된다.
    - 직관적으로 매우 당연한 결과이지만 이를 얻어내기까지의 수식 계산이 쉽지 않네.

- 지금까지 결합 분포 \\( p({\bf x}, {\bf y}) \\) 를 살펴보았으므로 이제 주변 확률 분포(marginal distribution)를 살펴보도록 하자.
- 여기에서는 앞서 다루었던 결합 분포의 성질을 이용하게 된다.
- 얻은 결과는 다음과 같다. (식을 전개하기가 귀찮다.)

$$E[{\bf y}] = {\bf A}{\pmb \mu} + {\bf b} \qquad{(2.109)}$$

$$cov[{\bf y}] = {\bf L}^T + {\bf A}\Lambda^{-1}{\bf A}^T \qquad{(2.110)}$$

- 여기서 \\( {\bf A}={\bf I} \\) 인 경우 두 가우시안의 관계가 *convolution* 관계라고 한다.
    - 여기서 *convolution* 은 두 개의 가우시안 함수가 서로 오버랩되는 영역을 나타내는 식이라고 생각하면 된다.
    - 이렇게 오버랩되는 영역도 마찬가지로 가우시안 분포를 따르게 된다.
    - 이 때 생성되는 분포의 평균는 각각의 분포의 평균의 합의 평균이 되고 분산은 각각의 분포의 분산의 합의 평균이 된다.

- 이제 조건 분포 \\( p({\bf x}\|{\bf y}) \\) 에 대해 좀 알아보도록 하자.
- 이 때의 평균과 분산은 너무나도 쉽게 구해지는데, 왜냐하면 이미 앞에서 다 구했기 때문이다. (식 2.73, 2.15)

$$\Sigma_{a|b}=\Lambda_{aa}^{-1}$$

$${\pmb \mu}_{a|b}={\pmb \mu}_a - \Lambda_{aa}^{-1}\Lambda_{ab}({\bf x}_b-{\pmb \mu}_b)$$

- 이걸 \\( p({\bf x}\|{\bf y}) \\) 에 맞게 대입만 하면 된다.

$$E[{\bf x}|{\bf y}] = (\Lambda+{\bf A}^T{\bf L}{\bf A})^{-1}\{ {\bf A}^T{\bf L}({\bf y}-{\bf b})+\Lambda{\pmb \mu}\} \qquad{(2.111)}$$

$$cov[{\bf x}|{\bf y}] = (\Lambda+{\bf A}^T{\bf L}{\bf A})^{-1} \qquad{(2.112)}$$














- <mark style='background-color: #dcffe4'> Maximum Likelihood for the Gaussian </mark>

관찰 데이터 집합 \\( {\bf X}=({\bf x}\_1,...,{\bf x}\_n)^T \\) 가 주어졌을 때 데이터 \\( \{ {\bf x}\_n\} \\) 은 서로 독립적으로 발현된다고 생각하겠습니다(*i.i.d*). 
이 때, 각각의 관찰 데이터는 가우시안 분포를 따르게 되며 이를 가능도 함수로 이용할 때에는 보통 로그를 취해 사용하게 된다.

$$\ln p({\bf X}|{\pmb \mu}, \Sigma) = -\frac{ND}{2}\ln(2\pi) - \frac{N}{2}\ln|\Sigma|-\frac{1}{2}\sum_{n=1}^{N}({\bf x}_n-{\pmb \mu})^T\Sigma^{-1}({\bf x}_n-{\pmb \mu}) \qquad{(2.118)}$$

이 식은 사실 최종적으로는 아래의 두가지 값에만 영향을 받게 됩니다..

$$\sum_{n=1}^{N}{\bf x}_n \qquad{(2.119)}$$

$$\sum_{n=1}^{N}{\bf x}_n{\bf x}_n^T \qquad{(2.119)}$$

이를 충분통계량(*sufficient statistics*)라고 부릅니다. 이제 우리는 미분을 통해 0인 지점을 구하면 Maximum Likelihood Estimation의 해를 구할 수 있습니다.

$$\dfrac{\partial}{\partial {\pmb \mu}}\ln p({\bf X}|{\pmb \mu}, \Sigma) = \sum_{n=1}^{N}\Sigma^{-1}({\bf x}_n-{\pmb \mu}) = 0 \qquad{(2.120)}$$

$${\pmb \mu}_{ML} = \frac{1}{N}\sum_{n=1}^{N}{\bf x}_n \qquad{(2.121)}$$

공분산의 최대값은 다음과 같습니다.

$$\Sigma_{ML} = \frac{1}{N}\sum_{n=1}{N}({\bf x}-{\pmb \mu}_{ML})({\bf x}-{\pmb \mu}_{ML})^T \qquad{(2.122)}$$ 

이 식에서는 \\( {\pmb \mu}\_{ML} \\) 이 사용되기 때문에 먼저 \\( {\pmb \mu}\_{ML} \\) 을 구하고 \\( \Sigma\_{ML} \\) 을 구하면 됩니다.

만약 Maximum Likelihood 해의 기대값을 실제 분포하에서 계산하게 되면 다음의 결과를 얻게 된다고 합니다.

$$E[{\pmb \mu}_{ML}] = {\pmb \mu} \qquad{(2.123)}$$

$$E[\Sigma_{ML}] = \frac{N-1}{N}\Sigma \qquad{(2.124)}$$

여기서 평균의 최대 가능도 추정치의 기대값이 실제 평균과 동일함을 알 수 있는데, 공분산은 조금 다르게 됩니다. 이는 분산 값의 평균이 실제 분산보다 작은, 작 bias되어 있다는 소리입니다.

$$\tilde{\Sigma} = \frac{1}{N-1}\sum_{n=1}^{N}({\bf x}_n - {\pmb \mu}_{ML})({\bf x}_n - {\pmb \mu}_{ML})^T \qquad{(2.125)}$$

(사실 이 내용은 자유도(degree of freedom)와 관련이 깊은 내용이 있다고 합니다, 통계학 서적을 참고해도 되나 사실 이후 과정을 전개함에 있어 위의 식만 알아도 큰 무리가 없기 때문에, 굳이 찾아서 볼 필요까지는 없다고 합니다.)


(미분해서 해를 구하는 과정은 생략합니다.)








- <mark style='background-color: #dcffe4'> Sequential Estimation  </mark>

순차 추정의 방법은 관찰 데이터 집합이 매우 커서 한번에 계산이 불가능할 때, 그리고 해를 구한 뒤 샘플이 계속 들어올 경우(on-memory) 사용할 수 있는 방법입니다. 한번에 한 샘플을 연산하고 버리게 됩니다.

*MLE* 로 얻어진 \\( {\pmb \mu}_{ML} \\) 식을 업데이트 방식으로 바꾸어보겠습니다. \\( {\pmb \mu}_{ML} \\) 에서 마지막 샘플을 추출해보면 다음과 같습니다.
  
$${\pmb \mu}_{ML}^{(N)} = \frac{1}{N} \sum_{n=1}^{N}{\bf x}_n = \frac{1}{N}{\bf x}_N + \frac{1}{N}\sum_{n=1}^{N-1}{\bf x}_n\\
= \frac{1}{N}{\bf x}_N + \frac{N-1}{N}{\pmb \mu}_{ML}^{(N-1)}={\pmb \mu}_{ML}^{(N-1)}+\frac{1}{N}({\bf x}_N-{\pmb \mu}_{ML}^{(N-1)}) \qquad{(2.126)}$$

즉, \\( N-1 \\) 개의 데이터로부터 추정된 \\( {\pmb \mu}\_{ML}^{(N-1)} \\) 와 \\( N \\) 번째 관측된 데이터를 이용하여 \\( {\pmb \mu}\_{ML}^{(N)} \\) 을 구하게 됩니다.
여기서 \\( N \\) 의 값이 증가할수록 새로 관측되는 데이터의 기여도가 점점 작아지게 됩니다.
그리고 한번에 계산을 처리하는 배치 방식으로부터 식을 유도해 내었기 때문에 실제 결과는 동일하게 된다.
하지만 이런 방식은 매우 유용하지만 배치 방식의 식에서 업데이트 방식의 식을 항상 유도할 수 있는 것은 아니다.


(Robbins-Monro 생략)









- <mark style='background-color: #dcffe4'> Bayesian inference for the Gaussian </mark>

가우시안의 *MLE* 는 파라미터인 평균과 공분산에 대한 점추정(point estimation) 값이었습니다.

이번에는 베이지안 방법론에 대해 알아볼건데요, 현재 likelihood로 가우시안 분포를 가정하고 있기 때문에 우리가 베이지안 추론을 하게될 경우는 크게 아래의 세가지로 나뉘게 됩니다. 

- 1. 분산 값을 알고 있을 때 평균 값의 추론
- 2. 평균 값을 알고 있을 때 분산 값의 추론
- 3. 평균, 분산 둘 다 모를 때의 두 값에 대한 추론
    
우선은 쉽게 일변량 가우시안 분포부터 시작하도록 하겠습니다.

우선 가우시안 분포가 하나 주어져있고, 이 때의 분산 값은 이미 알고 있다고 생각해 봅시다. 
그리고 데이터는 \\( N \\) 번의 관찰 데이터 \\( {\bf x} = (x\_1,...,x\_N)^T \\) 가 주어졌습니다.

이제 가능도 함수를 구해볼까요?

$$p({\bf x}|\mu) = \prod_{n=1}^{N}p(x_n|\mu) = \dfrac{1}{(2\pi\sigma^2)^{N/2}}\exp\left\{-\frac{1}{2\sigma^2}\sum_{n=1}^{N}(x_n-\mu)^2\right\} \qquad{(2.137)}$$

이제 사후 분포를 계산하기 위해서 사전분포 \\( p(\mu) \\) 를 도입해보겠습니다.
우리는 이 함수를 \\( \mu \\) 의 사전 확률 함수로 공액(`conjugate`) 분포를 사용할 겁니다.
가우시안 분포로 예를 들어보겠습니다.

$$p(\mu) = N(\mu|\mu_0, \sigma_0^2) \qquad{(2.138)}$$

이제 우리는 likelihood와 prior의 곱으로 posterior를 얻을 수 있습니다.

$$p(\mu|{\bf x}) \propto p({\bf x}|\mu)p(\mu) \qquad{(2.139)}$$

$$p(\mu|{\bf x}) = N(\mu|\mu_N, \sigma_N^2) \qquad{(2.140)}$$

$$\mu_N = \frac{\sigma^2}{N\sigma_0^2+\sigma^2}\mu_0 + \frac{N\sigma_0^2}{N\sigma_0^2+\sigma^2}\mu_{ML} \qquad{(2.141)}$$

$$\frac{1}{\sigma_N^2}=\frac{1}{\sigma_0^2}+\frac{N}{\sigma^2} \qquad{(2.142)}$$

앞서 *MLE* 를 통한 평균 값의 추론 결과는 아래와 같았습니다.

$$\mu_{ML} = \frac{1}{N}\sum_{n=1}^{N}x_n \qquad{(2.143)}$$

베이지안 추론을 통해 얻어진 평균값과 분산에 대해 고찰하는 것은 매우 의미있는 일입니다.

**posterior의 mean**

- 우선 베이지안 추론을 통해 얻어진 사후 확률에 대한 평균값에 대해 생각해 보겠습니다.
    - 만약 \\( N \\) 이 \\( N=0 \\) 이라면 평균값은 그냥 \\( \mu_0 \\) 가 됩니다. (즉, 최초 설정한 평균이 된다.)
    - 만약 \\( N \\) 이 \\( N\rightarrow\infty \\) 라면 평균 값은 결국 *MLE* 의 결과와 같아지게 됩니다. ( \\( \mu_{N\rightarrow\infty}=\mu\_{ML} \\))

**posterior의 variance**

- 이제 베이지안 추론을 통해 얻어진 사후 확률의 분산값에 대해 생각해보겠습니다.
    - 만약 \\( N \\) 이 \\( N=0 \\) 이라면 분산도 평균과 마찬가지로 \\( \sigma_0^2 \\) 이 됩니다.
    - 만약 \\( N \\) 이 \\( N\rightarrow\infty \\) 라면 분산 값은 점점 0에 가까워집니다. ( \\( \sigma\_{N\rightarrow\infty}^2=\sigma^2/N \\))
        - 이 의미는 정확도(precision) 가 계속 증가한다는 의미이고,
        - 결국 분포의 모양이 점점 높은 피크를 가지게 된다는 의미가 됩니다.

![bayesian1](https://user-images.githubusercontent.com/48202736/107395491-379c4800-6b40-11eb-8bba-8208c31d7a17.png)



-----

- 우리는 이미 가우시안 분포에서 평균 값을 순차 추정의 식으로 변환하는 것을 살펴보았다.
    - 이 식에서는 입력 데이터 \\( x_N \\) 이 주어졌을 때 \\( N-1 \\) 번째 데이터 관찰된 데이터와의 조합으로 업데이트 식을 작성하였다.
    - 사실 베이지안 패러다임에서는 자연스럽게 이러한 순차 처리 방식을 적용할 수 있도록 해준다. (predictive model)
    - 이를 이용하여 가우시안 평균에 대한 추론 방법을 확인해 보도록 하자.
    - 식에서 우선 \\( x_N \\) 데이터만을 분리하는 과정이 필요하다.

$$p(\mu|{\bf x}) \propto \left[p(\mu)\prod_{n=1}^{N-1}p(x_n|\mu)\right]p(x_N|\mu) \qquad{(2.144)}$$

- 위의 식에서 네모난 영역으로 된 곳이 \\( N-1 \\) 번째 까지의 관찰 데이터에 관한 식이 된다.
    - 물론 위의 식을 그대로 사용하기 위해서는 모든 관찰 데이터는 독립적(i.i.d)으로 생성되어야 한다.
- 어쨌거나 베이지안 추론 방식에서는 구하고자 하는 평균의 값이 점추정 값이 아니라 사후 확률( \\(p(\mu\|{\bf x}) \\) ) 분포로 제공되게 되므로 위의 식을 그대로 업데이트 식으로 사용하면 된다.
    - 만약 평균 값을 점추정하고 싶은 경우 분포식의 평균 값을 사용하면 된다.
    - 현재는 가우시안 분포를 따르므로 최빈값(mode)를 구하거나 평균을 구하거나 얻어지는 값은 동일하다.
    - 공분산의 값은 고정된 값이라 가정하여 식에서 생략했다.

- 다음으로 공분산 추론하는 방법을 알아보도록 한다. 
    - 평균을 구할 때 공분산 값이 고정되어있다고 가정한 것과 마찬가지로 공분산을 추론할 때에는 고정된 평균값이 주어졌다고 가정한다.
    - 계산의 편리성을 위해 다시 한번 공액 사전 분포 (conjugate prior distribution)를 사용한다.
    - 실제 계산에서는 공분산 값을 직접 추론하는 것보다 공분산의 역수, 즉 정확도(precision) \\( \lambda \equiv 1/\sigma^2 \\) 를 구하는게 낫다.

$$p({\bf x}|\lambda) = \prod_{n=1}^{N} N(x_n|\mu, \lambda^{-1}) \propto \lambda^{N/2} \exp\left\{ -\frac{\lambda}{2} \sum_{n=1}^{N}(x_n-\mu)^2\right\} \qquad{(2.145)}$$


- 정확도에 대한 사후 확률 분포에서 사용되는 공액 사전 분포는 **감마** (*gamma*) 분포이다.
- 이에 대한 식은 다음과 같다.

$$Gam(\lambda|a,b)=\frac{1}{\Gamma(a)}b^a\lambda^{a-1}\exp(-b\lambda) \qquad{(2.146)}$$

- 여기서 \\( \Gamma(a) \\) 는 감마 함수로 다음과 같이 정의되어 있다.

$$\Gamma(x) = \int_{0}^{\infty}u^{x-1}e^{-u}du$$

- 감마 분포의 평균과 분산은 다음과 같다.

$$E[\lambda] = \frac{a}{b} \qquad{(2.147)}$$

$$var[\lambda] = \frac{a}{b^2} \qquad{(2.148)}$$

- \\( a \\) 와 \\( b \\) 의 여러 값에 대한 감마 분포를 그림으로 나타내면 다음과 같다.



- \\( a>0 \\) 인 경우 감마 분포는 유한 적분(finite integral)이며 \\( a \ge 1 \\) 인 경우 분포 자체는 유한값이다.
    - 교재에는 이렇게 간단하게 적혀있으나 이게 무슨 말인지 모르겠다.
        - 실제로 해당 기능을 구현할 때 수렴 여부로 결과 값을 만들 수 있는지 없는지 체크할 때 쓴다는 건 어디서 들었다.
    - 우선 \\( a \ge 1 \\) 여야 \\( \infty \\) 로 수렴하는 값이 없다는 것은 알 수 있다.
    - \\( \Gamma(a) \\) 함수는 \\( a>0 \\) 이어야 유한 적분(finite integral)을 만족한다.
        - 이 부분이 조금 이해가 안된다. 수학적 배경이 있는 듯.
        - 대략적으로 적분 가능 구간이 존재한다는 것은 알 수 있다. (a의 값에 따라 달라진다.)
        - 이후에 이에 대한 전개가 없으므로 그냥 넘어가기로 하자.

-----

- 이제 사전 분포를 \\( Gam(\lambda\|a\_0, b\_0) \\) 로 고려하여 전개한다.
- 그러면 최종적으로 사후 확률을 다음과 같이 얻을 수 있다.

$$p({\bf x}|\lambda) \propto \lambda^{a_0-1}\lambda^{N/2} \exp \left\{-b_0\lambda-\frac{\lambda}{2}\sum_{n=1}^{N}(x_n-\mu)^2\right\} \qquad{(2.149)}$$

- 공액(conjugate)적인 속성에 의해 위의 분포도 감마 분포를 따르게 된다.
    - 이 때의 \\( a \\) 와 \\( b \\) 값을 구해보도록 하자.

$$a_N = a_0 + \frac{N}{2} \qquad{(2.150)}$$

$$b_N = b_0 + \frac{1}{2}\sum_{n=1}^{N}(x_n-\mu)^2 = b_0 + \frac{N}{2}\sigma_{ML}^2 \qquad{(2.151)}$$

- 위의 사후 확률 분포 식을 유심히 보면 정규화를 위한 상수가 생략되어 있는 것을 알 수 있는데, 필요하다면 \\( \Gamma(a_N) \\) 을 이용하여 계산할 수 있다.
- \\( a_N \\) 을 구하는 식으로부터 \\( N \\) 개의 관찰 데이터가 \\( a_N \\) 값에 미치는 영향을 확인해볼 수 있다.
    - \\( a_N \\) 의 식을 보면 \\( N/2 \\) 만큼 값이 보정되고 있다.
    - 만약 \\( N=2a_0 \\) 라면 어떻게 될까? 이러면 \\( a_N = a_0 + \frac{2a_0}{2} = 2a_0 \\) 가 된다. 
        - 즉, 초기 값에서 \\( a_0 \\) 만큼 증가함을 알 수 있음.
    - \\( b_N \\) 도 마찬가지인데 \\( N=2a_0 \\) 라면 사전 확률의 분산 값은 \\( 2b_0/(2a_0)=b_0/a_0 \\) 이고 이를 대입하면 \\( b_N=2b_0 \\) 가 된다.
        - 즉, 초기 값에서 \\( b_0 \\) 만큼 증가함을 알 수 있음
    - 결국 effective number 는 \\( N=2a_0 \\) 지점임
        - 여기서 effective number는 관찰 데이터의 영향력이 지정된 사전 확률의 영향력을 넘어서는 지점에서의 관찰 데이터의 수라고 생각하면 된다.

- 참고로 지금까지 분산을 구하기 위해 정확도(precision)를 이용하여 식을 전개하였는데, 반대로 공분산을 이용하여 식을 전개할수도 있다.
    - 이 때에는 공액 분포로 감마 분포가 아니라 역감마 분포 (*inverse gamma distribution*)를 사용한다.
    
-----

- 이제 평균과 공분산(실제로 구하는 것은 정확도)을 둘 다 모른다고 가정할 때의 파라미터 추정 방식을 살펴보도록 하자.
- 이 때의 공액 사전 분포를 찾기 위해 가능도 함수에서 \\( \mu \\) 과 \\( \lambda \\) 의 의존성을 확인해야 한다.

$$p({\bf x}|\mu, \lambda) = \prod_{n=1}^{N}\left(\frac{\lambda}{2\pi}\right)^{1/2} \exp\left\{-\frac{\lambda}{2}(x_n-\mu)^2\right\}\\
\propto \left[\lambda^{1/2}\exp\left(-\frac{\lambda\mu^2}{2}\right)\right]^N\exp\left\{\lambda\mu\sum_{n=1}^{N}x_n-\frac{\lambda}{2}\sum_{n=1}^{N}x_n^2\right\} \qquad{(2.152)}$$

- 위 식은 가능도 함수로 이로 부터 \\( \mu \\) 와 \\( \lambda \\) 값을 추론해야 한다.
- 식을 완전히 분해할 수는 없고, 식의 전개를 통해 어느 정도 분해를 한다.
- 이제 사전 확률(prior distribution)을 좀 살펴보도록 하자.
    - 여기서 관심을 가져야 하는 모수는 2개. 즉 평균과 분산을 동시에 랜덤 변수로 고려해야 한다.
    - 따라서 사용되는 사전 확률은 \\( p(\mu, \lambda) \\) 가 된다.
    - 원래 공액적 관계를 만들어내기 위해서는 사전 확률의 분포와 사후 확률의 분포를 같도록 만들어야 한다.
    - 여기서는 가능도 함수에 내포된 \\( \mu \\) 와 \\( \lambda \\) 의 함수적인 의존성을 그대로 유지한 사전 분포를 생각해본다.
        - 그냥 가능도 함수 꼴로 사전 분포를 만들어본다는 이야기.
        - 근데 이게 가능한건가 하는 생각이 들기는 하지만 어쨌거나 현재 가우시안 분포를 다루고 있기 때문에,
        - 사전 분포를 가우시안이나 감마 분포의 형태로 취급하는 것도 이상한 것은 아님 (원래 공액 관계의 분포임)
    
$$p(\mu, \lambda) \propto \left[\lambda^{1/2}\exp\left(-\frac{\lambda\mu^2}{2}\right)\right]^\beta\exp\{c\lambda\mu-d\lambda\}\\
= \exp\left\{-\frac{\beta\lambda}{2}(\mu-c/\beta)^2\right\}\lambda^{\beta/2}\exp\left\{-(d-\frac{c^2}{2\beta})\lambda\right\} \qquad{(2.153)}$$

- 가능도 함수 꼴로 만들되 자잘한 상수 값은 항상 동일할 수 없기 때문에 추가적인 상수를 도입한다. ( \\( c \\) , \\( d \\) , \\( \beta \\) )
- 결합 확률 식에 의해 \\( p(\mu, \lambda) = p(\mu\|\lambda)p(\lambda) \\) 가 성립하므로 식을 다음과 같이 기술할 수도 있다.

$$p(\mu, \lambda) = N(\mu|\mu_0, (\beta\lambda)^{-1})Gam(\lambda|a,b) \qquad{(2.154)}$$

- 위의 식을 보아하니 가정한 분포 꼴로 대충 떨어지는 것 같다.
- 각각을 대입하여 전개하면 다음과 같은 식을 얻게 된다.

$$\mu_0 = \frac{c}{\beta}$$

$$a=(1+\beta)/2$$

$$b=d-\frac{c^2}{2\beta}$$

- 이 분포는 *Normal-Gamma* 분포의 형태 꼴이다.
- 이를 도식화하면 다음과 같다.


- 위의 그림은 \\( \mu_0=0 \\) , \\( \beta=2 \\) , \\( a=5 \\) , \\( b=6 \\)  일 때의 컨투어이다.

-----

- 만약 1차원이 아니라 \\( D \\) 차원일 때에는 어떻게 처리해야 하나?
    - 사실 별로 어렵지는 않다.
    - 다변량 가우시안 분포 \\( N({\bf x}\|{\pmb \mu}, \Lambda^{-1}) \\) 을 도입하여 전개하면 된다.

- 이 때 사용되는 사전 확률 분포는 다음과 같다.
    
$$W(\Lambda|{\bf W}, v) = B|\Lambda|^{(v-D-1/2)}\exp\left(-\frac{1}{2}Tr({\bf W}^{-1}\Lambda\right) \qquad{(2.155)}$$

- 이런 분포를 위샤트 분포 (*Wishart distribution*)라고 한다.

- 여기서 \\( v \\) 는 자유도(degrees of freedom)라고 하고, \\( {\bf W} \\) 는 \\( D \times D \\) 크기의 행렬이 된다.
- 정규화 상수인 \\( B \\) 는 다음과 같이 정의된다.

$$B({\bf W}, v) = |{\bf W}|^{-v/2}\left(2^{vD/2}\pi^{D(D-1)/4}\prod_{i=1}^{D}\left(\frac{v+1-i}{2}\right)\right)^{-1} \qquad{(2.156)}$$

- 1차식에서와 마찬가지로 정확도 행렬이 아닌 공분산 행렬로 이를 전개 가능하다. 
    - 이 때에는 역 위샤트 분포 (*inverse Wishart distribution*)를 사용해야 한다.

- 앞서 설명한 바와 같이 공액 사전 분포를 다음과 같이 기술할 수도 있다.

$$p({\pmb \mu}, \Lambda|\mu_0, \beta, {\bf W}, v) = N({\pmb \mu}|{\pmb \mu}_0, (\beta\Lambda)^{-1})W(\Lambda|{\bf W}, v)$$

- 이를 *normal-Wishart* 분포 또는 *Gaussian-Wishart* 분포라고 부른다.






- <mark style='background-color: #dcffe4'> Student’s t-distribution </mark>

- <mark style='background-color: #dcffe4'> Periodic variables </mark>






- <mark style='background-color: #dcffe4'> Mixtures of Gaussians </mark>

![mog1](https://user-images.githubusercontent.com/48202736/107395496-38cd7500-6b40-11eb-9ffd-453e52503ee8.png)

![mog2](https://user-images.githubusercontent.com/48202736/107395499-39660b80-6b40-11eb-918d-f8b6dd83df6b.png)

![mog3](https://user-images.githubusercontent.com/48202736/107395501-39660b80-6b40-11eb-8987-52c811df712f.png)




- <mark style='background-color: #dcffe4'> GMM vs NN </mark>

출처 : [link](https://stats.stackexchange.com/questions/463706/are-neural-networks-mixture-models)

![mog](https://user-images.githubusercontent.com/48202736/107328636-5d9afb80-6af2-11eb-9554-f1bdea0b03ed.gif)

![neuralnet](https://user-images.githubusercontent.com/48202736/107328642-5ffd5580-6af2-11eb-8217-c7eb90d08bd2.gif)
