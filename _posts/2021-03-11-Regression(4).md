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

여기서 


## <mark style='background-color: #fff5b1'> Gaussian Proccess (GP) Regression </mark>

### <mark style='background-color: #dcffe4'> Example : GP with RBF Kernel  </mark>

![reg4_3](/assets/images/regression/reg4_3.png)
*Fig.*



## <mark style='background-color: #fff5b1'> References </mark>

1. [Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.](http://www.computervisionmodels.com/)

2. [Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)

3. [Features and Basis Functions - Cs Princeton](https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/basis-functions.pdf)
