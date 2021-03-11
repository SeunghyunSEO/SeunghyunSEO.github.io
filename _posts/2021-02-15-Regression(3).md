---
title: (미완)Regression (3/4) - Non-linear regression, Kernelization and Gaussian processes

categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true

comments: true
---

이전까지 우리는 선형 회귀에 대해서 MLE, MAP 등에 대해 알아봤습니다. ML, MAP로 해를 구하는 것이 약간의 차이가 있지만, 
결국 이들은 우리가 구하고자 하는 파라메터와 입력 데이터간 선형 결합되어 있는 관계식에서 파라메터를 추정하는 것이라는 점은 같았습니다.

하지만 이러한 방식은 한계가 존재합니다. 가령 우리가 가진 데이터를 도저히 직선(Linear)으로 표현하기 힘들다면 어떨까요? (여기서 Linear는 매개변수가 아닌 독립변수인 경우를 말합니다.)
곡선(Non-Linear)으로 만들어야 하지 않을까요??

우리가 이번에 알아보게될 내용은 구하고자 하는 파라메터들 간의 선형 결합은 유지하면서 비선형의 함수를 통해서 non-linearity를 추가해 회귀 곡선의 표현력을 높혀보자는 내용이 될 것입니다.

---
< 목차 >
{: class="table-of-content"}
* TOC
{:toc}
---

## <mark style='background-color: #fff5b1'> Non-Linear Regression </mark>

![reg1](https://user-images.githubusercontent.com/48202736/107945467-03f36f00-6fd3-11eb-86ec-1a68cda77511.png)
*Fig. 회귀 문제의 다양한 Variation*

위의 그림을 보시면 우리가 이번에 다루게 될 내용에 대해서 감이 오실 것 같습니다, 글로 써보면 아래와 같겠군요.

> 1. 비선형성을 추가해서 좀더 표현력을 높힌 회귀 함수 <br>
> 2. 베이지안 방법론과 비선형성 두 가지를 결합한 강력한 회귀 함수 <br>

우선은 비선형 회귀 함수에 대해서 생각을 해 보도록 하겠습니다.

우리가 일반적으로 알고있는 선형 회귀 문제에 대한 수식은 아래와 같습니다.

$$
Pr(w_i \vert x_i, \theta) = Norm_{w_i}[\theta^T x_i, \sigma^2]
$$

여기서 비선형성을 주기위해서 우리는 입력값을 어떠한 비 선형 함수에 통과시켜 볼 수 있습니다.

$$
Pr(w_i \vert x_i, \theta) = Norm_{w_i}[\theta^T x_i, \sigma^2]
$$

$$
where \space z_i = f[x_i]
$$

즉 다시말해서 원래의 입력값 $$x$$를 $$z$$로 바꾼 뒤 이에 대해서 선형 회귀 문제를 푸는 것이죠.
여기서 $$x$$를 $$z$$로 매핑해주는 함수가 중요한데, 우리는 바로 이 기저 함수(Basis Function) 이라고 합니다.




## <mark style='background-color: #fff5b1'> Basis Functions </mark>

기저 함수에는 다양한 종류들이 존재할 수 있습니다. 

<img width="1154" alt="스크린샷 2021-02-15 오후 9 54 57" src="https://user-images.githubusercontent.com/48202736/107949532-c2fe5900-6fd8-11eb-9117-40161f31dc8c.png">
*Fig. 기저 함수의 예시들. 왼쪽부터 다항 기저 함수이며 가운데는 가우시안 모양의 기저 함수, 맨 오른쪽은 시그모이드의 기저함수이다.*

위에 그림에 나온 기저 함수들 외에도 푸리에 기저 함수, 코사인 기저 함수등이 사용될 수도 있습니다.
이제 각각의 기저 함수가 사용될 경우 어떠한 비선형 곡선을 만들어 낼 수 있는지에 대해서 하나씩 살펴보도록 하겠습니다. 




### <mark style='background-color: #dcffe4'> Polynomial Regression </mark>

우선 다항 회귀(polynomial regression) 입니다.

이는 입력 변수 $$x_i$$에 대해서 이에 대한 제곱항, 세제곱항등의 가중치 합(weighted sum)으로 곡선을 표현하는 방법입니다.

<img width="1380" alt="basis2" src="https://user-images.githubusercontent.com/48202736/107968811-2c3e9600-6ff2-11eb-9388-d9dc0794fdad.png">
*Fig. 다항 기저 함수의 몇가지 예시(좌)와 이에 랜덤하게 가중치를 곱한 결과(우)*

위의 그림의 가중치가 곱해진 다항 기저 함수를 이제 합해주기만 하면 우리는 새로운 곡선을 만들어 낼 수 있는데요,
이를 수식적으로 나타내면 아래와 같습니다.

$$
Pr(w_i \vert x_i, \theta) = Norm_{w_i}[\theta_0 + \theta_1 x_i + \theta_2 x_i^2 + \theta_3 x_i^3, \sigma^2 ]
$$

이를 간단하게 표현하면

$$
Pr(w_i \vert x_i, \theta) = Norm_{w_i}[\theta^T z_i, \sigma^2]
$$

$$
where, \space z_i = \left[ \begin{matrix} 1 \\ x_i \\ x_i^2 \\ x_i^3 \end{matrix} \right]
$$

이 됩니다.

```
여기서 회귀 문제를 풀기 위해서 우리가 여태까지 해왔던 대로 출력 함수를 가우시안 분포로 가정했으며, 다항 차수는 현재의 예제에서는 3차 이지만 차수를 더 늘릴 수 있습니다. 
다만 그러할 경우 일반적으로 회귀 곡선이 주어진 데이터에만 너무 잘 피팅되는, 이른바 오버피팅이 일어날 수 있습니다.
```

우리는 위의 식에서 어떠한 가중치를 곱해줄것인가?, 즉 각각의 기저함수와 곱해질 파라메터만을 ML 혹은 MAP로 학습하면 되는것입니다. (물론 fixed-variance 문제가 아니라면, variance도 구해야겠네요)




### <mark style='background-color: #dcffe4'> Radial Basis Functions </mark>

그 다음으로 알아볼 것은 방사 기저 함수(Radial Basis Function, RBF) 혹은 가우시안 기저 함수(Gaussian Function)라고 불리는 기저 함수입니다.

<img width="1384" alt="basis5" src="https://user-images.githubusercontent.com/48202736/107968816-2e085980-6ff2-11eb-9be0-28795e808329.png">
*Fig. 방사(가우시안) 기저 함수의 몇가지 예시(좌)와 이에 랜덤하게 가중치를 곱한 결과(우)*

각각의 기저 함수는 아래와 같은 꼴이며

$$
z_i = exp[\frac{-(x_i-\alpha_i)}{\lambda}]
$$

마찬가지로 우리는 각 기저함수에 어떤 가중치를 줄 것인지만 학습을 통해 정하면 됩니다.

![reg2](https://user-images.githubusercontent.com/48202736/107945481-081f8c80-6fd3-11eb-94c4-71fdea34641d.png)
*Fig. 각각의 기저함수와 그에 해당하는 학습된 가중치를 곱해 만들어 낸 최종 곡선의 모양은 (d)와 같다.*




### <mark style='background-color: #dcffe4'> Arc Tan Functions </mark>

아까 위에서는 시그모이드 기저 함수에 대해서 설명했었는데 어차피 tanh 함수는 시그모이드와 $$tanh(a)=2\sigma(2a)-1$$의 관계를 가지고 있기 때문에 그게 그거이지만, 
자료가 tanh 기저함수에 대한 그림 밖에 없기 때문에 tanh 함수로 생각해보도록 하겠습니다 :)

<img width="1383" alt="basis6" src="https://user-images.githubusercontent.com/48202736/107968819-2ea0f000-6ff2-11eb-9922-57917d17c638.png">
*Fig. tanh 기저 함수의 몇가지 예시(좌)와 이에 랜덤하게 가중치를 곱한 결과(우)*

여태까지 계속 해왔던 얘기 이기 때문에 그림으로 이해하고 넘어가시면 될 것 같습니다.

![reg3](https://user-images.githubusercontent.com/48202736/107945484-08b82300-6fd3-11eb-9229-944ad2186d69.png)
*Fig. 각각의 기저함수와 그에 해당하는 학습된 가중치를 곱해 만들어 낸 최종 곡선의 모양은 (d)와 같다.*




### <mark style='background-color: #dcffe4'> Other Basis Functions </mark>

코사인 기저 함수와 퓨리에 기저 함수 등이 있으며 이러한 기저 함수를 골라 사용하는것의 장점이 있겠으나 이 글에서는 다루지 않고 넘어가도록 하겠습니다.

<img width="1388" alt="basis3" src="https://user-images.githubusercontent.com/48202736/107968813-2cd72c80-6ff2-11eb-8df6-54ce6a70593c.png">
*Fig. 코사인 기저 함수의 몇가지 예시(좌)와 이에 랜덤하게 가중치를 곱한 결과(우)*

<img width="1386" alt="basis4" src="https://user-images.githubusercontent.com/48202736/107968814-2d6fc300-6ff2-11eb-89be-bcda8b2a1e57.png">
*Fig. 퓨리에 기저 함수의 몇가지 예시(좌)와 이에 랜덤하게 가중치를 곱한 결과(우)*





## <mark style='background-color: #fff5b1'> ML Solution for Non-Linear Regression </mark>

자 이제 우리가 최종적으로 구하고자 하는 Non-Linear Regression의 해를 구해보겠습니다.

$$
Pr(w_i \vert x_i,\theta) = Norm_{w_i}[\theta^T z_i, \sigma^2]
$$

$$
where, \space z_i = \left[ \begin{matrix} 1 \\ x_i \\ x_i^2 \\ x_i^3 \end{matrix} \right]
$$

위의 수식에서 우리는 기저 함수들간의 가중치 합으로 데이터에 맞는 곡선을 구하기 위해서, 가중치를 구하면 됩니다.
가우시안 분포를 가정한 경우, 우리는 간단하게 log-likelihood의 미분한 값이 0이 되는 값을 계산해 간단하게 닫힌 해(closed-form solution)을 구할 수 있었는데요.

수식으로 나타내면 아래와 같게 됩니다.

$$
\bigtriangledown ln Pr(w_i \vert x_i) = \theta \sum_{i=1}^{N} \{ w_i - \phi^T f[x_i] \} {f[x_i]}^T = 0
$$

위의 수식의 해는 아래와 같게 되고,

$$ \hat{\phi} = (ZZ^T)^{-1}Zw $$
  
$$ \hat{\phi} = \frac{(w-Z^T\phi)^T(w-Z^T \phi)}{I} $$


만약 $$z_i=f[x_i]$$가 $$x_i$$가 되는 경우, 즉 기저 함수를 안 쓰는것과 같다고 생각하면, 이는 우리가 일반적으로 알고 있는 선형 회귀 문제와 같게 되고 이 때의 해는 아래와 같게 됩니다.

$$ \hat{\phi} = (XX^T)^{-1}Xw $$

$$ \hat{\phi} = \frac{(w-X^T\phi)^T(w-X^T \phi)}{I} $$

여기서 $$Z$$는 데이터의 개수가 $$N$$개 이며 입력 차원이 $$M$$일 경우 $$N \times M$$ 함수가 되며,

$$
Z = 
\left[ \begin{matrix} 
f_0[x_1] && f_1[x_1] && \cdots  && f_{M-1}[x_1] \\ 
f_0[x_2] && f_1[x_2] && \cdots  && f_{M-1}[x_2] \\ 
\vdots  && \vdots  && \ddots  && \vdots  \\ 
f_0[x_N] && f_1[x_N] && \cdots  && f_{M-1}[x_N] 
\end{matrix} \right]
$$

위의 행렬을 바로 설계 행렬(design matrix)이라고 부릅니다.

여기서 우리가 구한 식 

$$ \hat{\phi} = (ZZ^T)^{-1}Zw $$

의 $$ (ZZ^T)^{-1}Z $$ 를 무어-펜로즈 수도 인버스(Moor-Penrose Pseudo Inverse)라고 하며

$$ Z^{-1}=(ZZ^T)^{-1}Z $$ 

이 됩니다.

그러므로 $$ \hat{\phi} = (Z)^{-1}w $$ 로 간단하게 해를 구할 수 있게 됩니다.






## <mark style='background-color: #fff5b1'> Bayesian Approach </mark>

자 여기서 끝이 아닙니다. 우리가 Maximum likelihood solution을 구했으니 당연히 베이지안 방법으로도 문제를 풀어보고 싶겠죠? 

![reg1](https://user-images.githubusercontent.com/48202736/107945467-03f36f00-6fd3-11eb-86ec-1a68cda77511.png)
*Fig. 회귀 문제의 다양한 Variation*

우리는 각 기저함수의 가중치들의 하나의 해를 구하는 '점 추정' 방식이 아니라 여러 가중치들에 대해서도 고려해 보고 싶은 것이죠.

![reg4](https://user-images.githubusercontent.com/48202736/107945486-0950b980-6fd3-11eb-917c-87da25117dd2.png)
*Fig. Bayesian Approach for Regression using RBF*

베이지안 추론을 하기 위한 수식을 쓰면 아래와 같습니다.

$$
Pr(w^{\ast} \vert z^{\ast}, X, W) = Norm_w[ \frac{\sigma_p^2}{\sigma^2} z^{\ast T} Z w - \frac{\sigma_p^2}{\sigma^2} z^{\ast T} Z (Z^TZ + \frac{\sigma^2}{\sigma_p^2} I)^{-1} Z^TZw, \space \sigma_p^2 z^{\ast T} z^{\ast} - \sigma_p^2 z^{\ast T} Z (Z^TZ + \frac{\sigma^2}{\sigma_p^2} I)^{-1} Z^T z^{\ast} + \sigma^2 ]
$$

(위의 수식에서 $$\sigma^2$$ 는 marginal likelihood로 학습하면 됩니다.)





## <mark style='background-color: #fff5b1'> References </mark>

1. [Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.](http://www.computervisionmodels.com/)

2. [Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)

3. [Features and Basis Functions - Cs Princeton](https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/basis-functions.pdf)
