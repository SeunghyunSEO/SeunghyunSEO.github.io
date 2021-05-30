---
title: Regression (2/6) - Bayesian Linear Regression
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

## <mark style='background-color: #fff5b1'> ML solution for Modeling Gaussian Dist over Output, W </mark>

우리는 이전에 `회귀 (Regression)` 문제, 그 중에서도 파라메터들에 대한 선형 결합으로 선이 표현되는 `선형 회귀 (Linear Regression)` 문제를 푸는 방법에 대해 알아봤습니다.
곡선을 찾아내기 위해서 가장 먼저 출력($$w$$)에 대한 분포를 가우시안 분포로 정의하고 `likelihood`인 $$Pr(y \mid x,\theta)$$ 를 최대화 하는 `Maximum likelihood Estimation, MLE` 방법으로 데이터에 알맞는 (fitting) 최적의 선을 찾아냈죠.


그리고 우리는 더 나아가 $$\theta$$에 대한 `prior`를 하나 더 정의해서 likelihood와 곱해서 구한 `posterior`, $$Pr(\theta \mid x,y)$$ 를 최대화 하는 `Maximum A Posterior, MAP` 방법을 사용했습니다.

![reg2_1](/assets/images/regression/reg2_1.png){: width="60%"}
*Fig. 일반적인 점 추정 방식의 회귀 곡선*

위의 그림은 MAP로 최적의 파라메터를 구했을 때의 그림입니다.


그치만 사실 우리가 찾은 선은 뭔가 불편한데요,
그것은 모든 x 에 대해 y 분포가 제 각기 다른데도 불구하고 (즉 샘플링 된 데이터의 밀도가 다름), 우리가 찾은 직선은 전구간에 걸쳐 다 똑같은 굵기라는 것입니다. 

즉 `전 구간에 있어 동일한 자신감 (confidence)`를 가지고 있는데요 (다르게 말하면 데이터가 없는 부분에서 over-confident 하다는 문제를 보인다는 것), 이는 우리가 결국 곡선을 구성하는 파라메터들에 대한 단 하나의 해를 가지고 곡선을 예측했기 때문입니다. 


이를 해결하기 위해서 어떻게할까요? 
그 해답은 바로 `Bayesian Approach`를 회귀 문제에도 적용하는 것인데요.

베이지안 방법론은 간단하게 말해서 posterior 분포를 추정하고 이에 대한 `파라메터를 전부 고려하여 (weighted-sum)` 출력 분포를 리턴하는 겁니다.
이런식으로 데이터의 밀도 (분포)에 따라서 confidence를 달리하고 더욱 그럴듯한 회귀 결과를 내는게 가능해지는데요,

![reg2_2](/assets/images/regression/reg2_2.png){: width="80%"}
*Fig. 데이터 밀도에 따라서 confident가 달라짐, 즉 불확실성(uncertainty)를 나타내는 곡선을 얻어냄*

이는 모든 구간에 동일한 confidence를 주지 않는 방법은 다르게 말하면 우리의 목표는 데이터 밀도가 적거나 데이터가 없는 부분에 대해서는 우리가 추정한 곡선이 잘 모르겠다는 의미로 큰 variance를 갖게 하는, 즉 `불확실성 (uncertainty)`를 나타내는 것이라고 할 수 있습니다.


MAP나 MLE가 결국에는 학습 데이터($$X,Y$$)를 보고 딱 하나의 해를 구하는, 즉 `점 추정 (point estimation)`을 하는 방법이기 때문에 우리는 어떤 새로운 테스트 데이터 $$x^{\ast}$$가 들어왔을 때 우리가 학습 데이터로부터 구한 하나의 최적의 파라메터를 이용해서 단순히 결과를 산출해내는 것이지만

$$
\theta^{\ast} \leftarrow X,Y
y^{\ast} = p(y^{\ast} \vert x^{\ast}, \theta^{\ast})
$$

베이지안 방법론은 어떤 테스트 데이터가 마찬가지로 들어왔을 때 데이터로 부터 얻어낸 파라메터의 분포를 기반으로 그 분포로 부터 나올 수 있는 모든 파라메터들을 전부 고려해서 아래와 같이 $$y^{\ast}$$ 를 예측해냅니다.

$$
y^{\ast} = p(y^{\ast} \vert x^{\ast}, X,Y) = \int p(y^{\ast} \vert x^{\ast}, \theta ) p(\theta \vert X,Y)
$$


![reg2_3](/assets/images/regression/reg2_3.png)
*Fig. uncertainty는 데이터 개수에 종속적임( 분홍색 음영이 큰 것은 그 데이터 포인트x에서 불확실성이 크다는 것)*











## <mark style='background-color: #fff5b1'> Bayesian Regression </mark>

![reg2_4](/assets/images/regression/reg2_4.png)

(이번 포스트에서는 Bayesian Linear Regression만 다루고 나머지는 다음에 다루도록 하겠습니다.)


이전에 선형 회귀 문제에서 파라메터를 구하던 것과 비슷하게 접근해 보도록 하겠습니다.

> 1. $$likelihood : p(x\mid\theta)$$ <br>
> 2. $$posterior \propto likelihood \times prior : p(\theta \mid x) \propto p(x \mid \theta)p(\theta)$$ <br> 

`likelihood` 와 `prior`를 모두 가우시안 분포로 정의하겠습니다. 
(왜냐면 `posterior`를 구해서 적분에 이용하게 될 텐데, 이는 두 분포의 곱이기 때문입니다. 그리고 두 가우시안 분포의 곱은 가우시안 분포기 때문이죠.)

<center>$$ likelihood : Pr(w|X) = Norm_w[X^T\phi,\sigma^2I] $$</center>
<center>$$ prior : Pr(\phi) = Norm_\phi[0,\sigma_p^2I] $$</center>

여기서 헷갈리지 말아야 할 점은 $$prior$$에 존재하는 분산은 $$\sigma_p^2$$라는 것입니다.
위의 사후 확률을 구하는 관계식을 이용해서 $$posterior$$를 구하면 다음과 같습니다.

<center>$$ posterior : Pr(\phi|X,w) = Norm_\phi[\frac{1}{\sigma^2} A^{-1}Xw, A^{-1}] $$</center>
<center>$$ where \space A = \frac{1}{\sigma^2} XX^T + \frac{1}{\sigma_p^2}I $$</center>



![reg2_5](/assets/images/regression/reg2_5.png)

위의 그림의 왼쪽은 원래 추정하고자 했던 $$\phi$$의 사전 확률인 $$prior$$가 가우시안 분포를 나타내고, 
오른쪽은 $$\phi$$의 분포를 나타냅니다. (variance는 나중에 다룰 예정)





### <mark style='background-color: #dcffe4'> Inference </mark>

Bayesian Deep Learning 분야에서 유명한 Yarin Gal의 thesis를 보면 Bayesian Modeling에서의 Inference를 아래와 같이 정의하고 있으니 읽어 보시고 일반적인 딥러닝 모델들에서 얘기하는 Inference와의 차이가 무엇인지를 생각해보면 좋을 것 같습니다.

```
Note that “inference” in Bayesian modelling has a different meaning to that in deep learning. In Bayesian modelling “inference” is the process of integration over model parameters. This means that “approximate inference” can involve optimisation at training time (approximating this integral). This is
in contrast to deep learning literature where “inference” often means model evaluation at test time alone.
```

다시 돌아가서, 우리가 학습 데이터 $$X,W$$ pair, 즉 입력 데이터 $$X$$와 이에 해당하는 정답 $$W$$를 가지고 있다고 합시다. 
그리고 우리는 정의한 모델을 통해 어떤 학습 데이터에 존재하지 않는 $$x^{\ast}$$에 대응하는 $$w^{\ast}$$를 찾고 싶습니다.


'$$x^{\ast}$$에 대응하는 $$w^{\ast}$$' 이는 수식적으로 아래와 같이 표현할 수 있습니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,W) $$</center>

이는 `marginalization` 테크닉을 통해 아래와 같이 나눌 수 있는데요,

<center>$$ Pr(w^{\ast}|x^{\ast},X,W) = \int Pr(w^{\ast}|x^{\ast},\phi) Pr(\phi|X,W) d\phi $$</center>


우리는 위의 수식에 있는 `posterior`, $$Pr(w \mid x,\phi)$$와 $$Pr(\phi \mid X,W)$$에 대해 이미 정의를 했습니다. 
이를 사용해서 계속해서 전개하면 아래와 같은 수식을 얻을 수 있습니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,w) = \int Norm_{w^{\ast}}[\phi^T x^{\ast},\sigma^2] Norm_{\phi}[\frac{1}{\sigma^2} A^{-1}Xw, A^{-1}] d\phi $$</center>
<center>$$ = Norm_{w^{\ast}}[\frac{1}{\sigma^2}x^{\ast T}A^{-1}Xw,x^{\ast T}A^{-1}x^{\ast} + \sigma^2] $$</center>
<center>$$ where \space A = \frac{1}{\sigma^2} XX^T + \frac{1}{\sigma_p^2}I $$</center>

이렇게 구한 식은 자세히 보시면 파라메터 $$\phi$$가 없습니다.
이는 우리가 데이터 집합 $$(X,W)$$ 와 적절히 가정한 prior,likelihood분포가 있다면 어떤 테스트 데이터 x가 들어왔을때 그때의 output w의 분포를 파라메터 $$\phi$$를 정확히 하나 콕 찝어서 추정하지 않고 나타낼 수 있다는 겁니다.
여기서 우리는 마지막 솔루션이 $$\sigma$$에만 종속되어있다는 것을 알 수 있는데요, 
일반적으로 이 값은 ` 정해져있거나 (fixed variance)` 그렇지 않을 경우 `marginal likelihood를 최적화` 하는 방법으로 (maximum likelihood) 구해놓은 뒤에 사용합니다. 
(구해놓으면 그 뒤로는 바로바로 사용 가능 하다는 소리)



아래의 수식에 대해서 그림으로 한번 더 생각해보자면,

<center>$$ Pr(w^{\ast}|x^{\ast},X,W) = \int Pr(w^{\ast}|x^{\ast},\phi) Pr(\phi|X,W) d\phi $$</center>


![reg2_6](/assets/images/regression/reg2_6.png)

a)는 추정하고자 하는 파라메터 $$\phi_0,\phi_1$$의 분포를 나타내는 것입니다.
원래 `MAP는 여기서 최대가 되는 값 하나만`을 학습을 통해 구했으나, 이제는 b)처럼 가능한 파라메터 $$\phi_{1}$$, $$\phi_{2}$$, $$\phi_{3}$$ ... 에 대해서 모두 생각을 해보자는 것이죠.


b)는 파라메터 $$\phi_0,\phi_1$$가 어떤 값이냐에 따라서 선형 회귀의 직선이 어떻게 표현되는지를 나타냅니다. 즉 a에서 분포를 다 고려해 본다는 것은 이렇게 다 찍어본다는 것이고 그때마다 다른 회귀 곡선들을 `weighted-sum`한다는 것과 같습니다.


c)는 말 그대로 위의 식 처럼 가능한 파라메터 $$\phi$$에 대해서 모두 적분한 결과입니다.  


마지막으로 우리가 구한 사후 분포의 mean에 관한 수식을 전부 풀어쓰면 아래와 같습니다.

$$ 
Pr(w^{\ast} \vert x^{\ast}, X, W) = Norm_w[ \frac{\sigma_p^2}{\sigma^2} x^{\ast T} X w - \frac{\sigma_p^2}{\sigma^2} x^{\ast T} X (X^TX + \frac{\sigma^2}{\sigma_p^2} I)^{-1} X^TXw,  \\
\space \sigma_p^2 x^{\ast T} x^{\ast} - \sigma_p^2 x^{\ast T} X (X^TX + \frac{\sigma^2}{\sigma_p^2} I)^{-1} X^T x^{\ast} + \sigma^2 ] 
$$








## <mark style='background-color: #fff5b1'> 데이터 수에 따른 MLE, MAP 그리고 Bayesian Approach </mark>

우리가 앞서 MLE와 MAP는 '점 추정'을 한다고 했었죠.
반면에 베이지안 관점에서는 테스트 데이터에 대한 정답을 추론할 때 마다 적분을 하기 위한 점이 아닌 posterior분포를 다 사용하는, '분포 추정'을 한다고 할 수 있었습니다.
이에 대해 조금만 더 생각해볼까요?


우리가 가지고 있는 데이터가 적다고 가정해보겠습니다.
MLE는 prior가 없기 때문에 MAP에 비해서 좋지 않은 결과를 낼 수도 있습니다.
하지만 MAP 또한 데이터가 적기 때문에 `posterior가 넓게 분포`하게 되어, 별로 높지 않은 자신감 (확률)로 뽑은 posterior가 최대가 되는 해인 MAP 솔루션은 베이지안 방법보다 별로 좋지 않을 수 있습니다 


![reg2_7](/assets/images/regression/reg2_7.png)
*Fig. 일반적인 선형회귀의 ML 솔루션은 MSE Loss의 최소값*

![reg2_8](/assets/images/regression/reg2_8.png)

하지만 데이터가 많아지다못해 무한대에 가까워지면 이는 점점 posterior 분포는 어느 한 점을 나타내는 것과도 같 `delta function`에 가깝게 되고. 
이 때의 인퍼런스를한 결과는 점 추정을 하는 MLE가 MAP가 엄청 유사해지고, 이 posterior에서 파라메터들을 샘플링해서 더해야하는 베이지안 방법론 또한 유사해집니다.

![reg2_9](/assets/images/regression/reg2_9.png)
![reg2_10](/assets/images/regression/reg2_10.png)
![reg2_11](/assets/images/regression/reg2_11.png)
*Fig. 데이터가 많아질수록 posterior가 delta function 가까워지는 모습*






## <mark style='background-color: #fff5b1'> Limitation </mark>

베이지안 방법으로 문제를 푸는 것은 상당히 괜찮은 접근으로 보입니다.
하지만 위에서 유도한 식과 달리 실제로는 $$posterior$$ 를 제대로 구하기 쉽지 않습니다. (위에서는 $$posterior$$ 에 대한 관계식 정도만 언급했음)

> 1. $$likelihood : p(x\mid\theta)$$ <br>
> 2. $$posterior \propto likelihood \times prior : p(\theta \mid x) \propto p(x \mid \theta)p(\theta)$$ <br> 

위와 같이 $$posterior$$ 를 간단한 관계식으로 표현했지만, 사실 $$posterior$$를 구하기 위한 Bayes' Rule은 조금 더 복잡하기 때문입니다.







### <mark style='background-color: #dcffe4'> Bayes' Rule </mark>

<center>$$posterior :  p(\theta \mid X,W) = \frac{p(W \mid X, \theta)p(\theta)}{p(W \mid X)}$$</center>

<center>$$p(W \mid X) = \int p(W|X,\theta)p(\theta)d\theta$$</center>

위에서 보시는 거와 같이 사실은 베이즈 룰에서는 분모, normalizer term 혹은 model evidence 라고 하는 적분 term이 중요합니다.

이 적분을 계산하는 것은 `marginalising the likelihood over ω` 혹은 `marginal likelihood`라고도 합니다.


우리가 예시로 든 간단한 Bayesian Linear Regression 문제에서는 likelihood와 prior를 둘 다 `conjugate` 관계인 가우시안 분포로 적당히 가정했기 때문에 적분을 계산하는게 상대적으로 쉬웠습니다.
하지만 조금만 복잡해져도 (예를들어 basis function이 고정되어 있지 않은 Basis Function Regression이라던가, Neural Network 라던가...) 위의 적분은 계산하기가 쉽지 않습니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,w) = \int Pr(w^{\ast}|x^{\ast},\phi) Pr(\phi|X,w) d\phi $$</center>

즉, 위의 식에서 $$Pr(\phi \mid X,w)$$를 제대로 구할 수 없다는 것이죠.


이런 경우를 `The true posterior cannot usually be evaluated analytically.` 혹은 `the true posterior is intractable` 하다고 합니다.
이를 해결하기 위해서는 $$true \space posterior$$와 유사하지만 계산하기 `쉬운 분포로 근사(approximate)`를 해서 문제를 풀게 됩니다.
이렇게 근사하는 방법들에는 `Laplace Approximation`이나 `Variational Inference` 같은 방법들이 등이 있습니다.


그리고 이렇게 멋지게 posterior 분포를 쉬운 분포로 근사해내도 우리가 결국 원하는 아래의 적분식을 구하는 것 또한 쉽지 않습니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,W) = \int Pr(w^{\ast}|x^{\ast},\phi) Pr(\phi|X,W) d\phi $$</center>


그렇기 때문에 이렇게 베이지안 방법론을 간단한 회귀 문제가 아닌 비선형성이 포함된 현대의 딥러닝에 적용하는기 위한 Practical한 방법론들이 제안되어 왔는데요.

![reg2_12](/assets/images/regression/reg2_12.jpeg)
*Fig. 신경망 (Neural Network NN) 에 분포 추정 방식을 적용한, 이른 바 Bayesian Neural Network (BNN)*
![reg2_13](/assets/images/regression/reg2_13.jpeg)
*Fig. 가우시안 프로세스에 베이지안 방법론을 적용한 것은, 현대 딥러닝의 가장 심플하면서도 강력한 정규화 방법론 중 하나인 Dropout을 하는것과 같다는 것을 수식적으로 증명한 Yarin Gal*

Regression 시리즈는 아니지만 나중에 기회가 되면 Bayesian Deep Learning에 대해서도 다뤄보도록 하겠습니다.







## <mark style='background-color: #fff5b1'> References </mark>

1. [Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.](http://www.computervisionmodels.com/)

2. [Gal, Yarin. "Uncertainty in deep learning." University of Cambridge 1, no. 3 (2016): 4.](https://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html)

3. [What My Deep Model Doesn't Know... from Yarin Gal](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)

4. [NYU Bayesian Deep Learning : Tutorials](https://wjmaddox.github.io/assets/BNN_tutorial_CILVR.pdf)
