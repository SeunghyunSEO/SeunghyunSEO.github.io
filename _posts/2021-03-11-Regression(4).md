---
title: (미완)Regression (4/6) - Kernelization and Gaussian processes

categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true

comments: true
---


---
< 목차 >
{: class="table-of-content"}
* TOC
{:toc}
---

이번 글에서 다루고 싶은 내용은 아래의 그림에서

![reg4_1](/assets/images/regression/reg4_1.png)
*Fig.*

비선형성과 베이지안 방법론을 적용하고 여기에 `커널 트릭 (Kernel Trick)` 까지 적용해보는 겁니다.

## <mark style='background-color: #fff5b1'> Recap : Bayeisan Non-Linear Regression </mark>

아래의 그림처럼 커널을 사용하고 $$z = \phi(x)$$

![reg4_2](/assets/images/regression/reg4_2.png)

여기에 베이지안 방법론까지 적용하면 아래의 수식을 통해 새로운 데이터셋 $$x^{\ast}$$가 들어왔을때의 분포를 예측할 수 있었습니다.

$$
Pr(w^{\ast} \vert z^{\ast}, X, W) = Norm_w[ \frac{\sigma_p^2}{\sigma^2} z^{\ast T} Z w - \frac{\sigma_p^2}{\sigma^2} z^{\ast T} Z (Z^TZ + \frac{\sigma^2}{\sigma_p^2} I)^{-1} Z^TZw, \\
\sigma_p^2 z^{\ast T} z^{\ast} - \sigma_p^2 z^{\ast T} Z (Z^TZ + \frac{\sigma^2}{\sigma_p^2} I)^{-1} Z^T z^{\ast} + \sigma^2 ]
$$


## <mark style='background-color: #fff5b1'> Kernel Trick </mark>

여기서 중요한 점은 아래의 수식에 데이터 그 자체는 필요하지 않고 `데이터들 사이의 관계` 식인 $$z_iT^ z_j$$, 즉 내적한 결과만 들어가 있다는 걸 알 수 있습니다.  

$$
Pr(w^{\ast} \vert z^{\ast}, X, W) = Norm_w[ \frac{\sigma_p^2}{\sigma^2} z^{\ast T} Z w - \frac{\sigma_p^2}{\sigma^2} z^{\ast T} Z (Z^TZ + \frac{\sigma^2}{\sigma_p^2} I)^{-1} Z^TZw, \\
\sigma_p^2 z^{\ast T} z^{\ast} - \sigma_p^2 z^{\ast T} Z (Z^TZ + \frac{\sigma^2}{\sigma_p^2} I)^{-1} Z^T z^{\ast} + \sigma^2 ]
$$

그러니까 $$x$$ 를 커널에 태워서 $$z$$들로 만들고 그 데이터들 100개면 100개끼리 `dot product`한 결과로만 표현된다는 거죠.
여기서 커널 트릭의 핵심 아이디어는 바로 위의 과정을 한큐에 처리할 수 있는 함수를 정의하자는 겁니다.

우리가 이러한 함수를 굉장히 잘 고르면 우리는 $$z=f[x]$$와 동일한 역할을 하는 함수를 얻을 수 있고, 
이 때의 장점은 어떤 $$x$$가 어떠한 기저함수를 타고 $$z$$ 로 매핑이 되는지 명시적으로 (explicit) 하게 정하지 않아도 된다는 것이 되며, 심지어 저차원의 데이터 $$x$$가 우리가 정의한 커널 함수를 통해 `암시적 (implicit)` 으로 알게된 어떠한 기저함수를 통해 매우 노거나 무한차원의 $$z$$로 매핑이 될 수도 있다는 겁니다.


이럴 


## <mark style='background-color: #fff5b1'> Gaussian Proccess (GP) Regression </mark>

### <mark style='background-color: #dcffe4'> Example : GP with RBF Kernel  </mark>

![reg4_3](/assets/images/regression/reg4_3.png)
*Fig.*



## <mark style='background-color: #fff5b1'> References </mark>

1. [Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.](http://www.computervisionmodels.com/)

2. [Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)

3. [Features and Basis Functions - Cs Princeton](https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/basis-functions.pdf)
