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

우리는 이전에 회귀 문제, 그 중에서도 선형 회귀 문제를 푸는 방법에 대해 알아봤습니다.

가장 먼저 출력($$w$$)에 대한 분포를 가우시안 분포로 정의하고 $$likelihood$$인 $$Pr(y \mid x,\theta)$$ 를 최대화 하는 Maximum likelihood 방법이나,

$$\theta$$에 대한 $$prior$$를 하나 더 정의해서 $$likelihood$$와 곱해서 구한 $$posterior$$, $$Pr(\theta \mid x,y)$$ 를 최대화 하는 Maximum A Posterior 방법을 사용했습니다.

![reg2_1](/assets/images/regression/reg2_1.png){: width="60%"}
*Fig. 일반적인 점 추정 방식의 회귀 곡선*

위의 그림은 MAP로 최적의 파라메터를 구했을 때의 그림입니다.


그치만 사실 뭔가 불편합니다.


뭐가 불편하냐면 그것은 모든 실제로 x 에 대해 y 분포가 제 각기 다른데도 불구하고 (즉 샘플링 된 데이터의 밀도가 다름), 우리가 찾은 직선은 전구간에 걸쳐 다 똑같은 굵기라는 것입니다. 


즉 전 구간에 있어 동일한 confidence를 가지고 있다, 즉 데이터가 없는 부분에서 over-confident 하다는 문제를 보인다는 것입니다.


이를 해결하기 위해서 어떻게할까요? 당장 생각할 수 있는 방법은 $$posterior$$ 가장 큰 값 하나만 구하는 MAP를 사용하지 말고, 
한발 더 나아가 가능한 모든 파라메터에 대해 적분하는 Bayesian 방법을 사용하는 것입니다.

모든 구간에 동일한 confidence를 주지 않는 방법은 다르게 말하면 우리의 목표는 데이터 밀도가 적거나 데이터가 없는 부분에 대해서는 우리가 추정한 곡선이 잘 모르겠다는 의미로 큰 variance를 갖게 하는, 즉 uncertainty를 나타내는 것 입니다.

![reg2_2](/assets/images/regression/reg2_2.png)
*Fig. 데이터 밀도에 따라서 confident가 달라짐, 즉 불확실성(uncertainty)를 나타내는 곡선을 얻어냄*

![reg2_3](/assets/images/regression/reg2_3.png)
*Fig. uncertainty는 데이터 개수에 종속적임( 분홍색 음영이 큰 것은 그 데이터 포인트x에서 불확실성이 크다는 것)*






## <mark style='background-color: #fff5b1'> Bayesian Regression </mark>

![reg2_4](/assets/images/regression/reg2_4.png)

오늘은 Bayesian Linear Regression만 다루고 나머지는 다음에 다루도록 하겠습니다.

우리가 잘 아는 관계식이 하나 있습니다.

> 1. $$likelihood : p(x\mid\theta)$$ <br>
> 2. $$posterior \propto likelihood \times prior : p(\theta \mid x) \propto p(x \mid \theta)p(\theta)$$ <br> 

이전에 선형 회귀 문제에서 파라메터를 구하던 것과 비슷하게 접근해봅시다.

$$likelihood$$ 와 $$prior$$를 모두 가우시안 분포로 정의하겠습니다. 
(왜냐면 $$posterior$$를 구해서 적분을 하게 될 텐데, $$posterior$$는 두 분포 $$likelihood$$ 와 $$prior$$의 곱이기 때문입니다. 그리고 두 가우시안 분포의 곱은 가우시안 분포기 때문이죠.)

<center>$$ likelihood : Pr(w|X) = Norm_w[X^T\phi,\sigma^2I] $$</center>

<center>$$ prior : Pr(\phi) = Norm_\phi[0,\sigma_p^2I] $$</center>

여기서 헷갈리지 말아야 할 점은 $$prior$$에 존재하는 분산은 $$\sigma_p^2$$라는 것입니다.

위의 사후 확률을 구하는 관계식을 이용해서 $$posterior$$를 구하면 다음과 같습니다.

<center>$$ posterior : Pr(\theta|X,w) = Norm_\phi[\frac{1}{\sigma^2} A^{-1}Xw, A^{-1}] $$</center>
<center>$$ where \space A = \frac{1}{\sigma^2} XX^T + \frac{1}{\sigma_p^2}I $$</center>



![reg2_5](/assets/images/regression/reg2_5.png)

위의 그림의 왼쪽은 원래 추정하고자 했던 $$\phi$$의 사전 확률인 $$prior$$가 가우시안 분포를 나타내고, 
오른쪽은 $$\phi$$의 분포를 나타냅니다. (variance는 나중에 다룰 예정)





### <mark style='background-color: #dcffe4'> Inference </mark>

Bayesian Deep Learning 분야에서 유명한 Yarin Gal의 thesis를 보면 Bayesian Modeling에서의 Inference를 아래와 같이 정의하고 있습니다.

그러니 읽어보신 뒤, Bayesian inference를 일반적인 딥 러닝에서의 추론과 햇갈리지 마시고 흐름을 따라가시면 좋을 것 같습니다.

```
Note that “inference” in Bayesian modelling has a different meaning to that in deep learning. In Bayesian modelling “inference” is the process of integration over model parameters. This means that “approximate inference” can involve optimisation at training time (approximating this integral). This is
in contrast to deep learning literature where “inference” often means model evaluation at test time alone.
```

자 우리가 학습 데이터 $$X,W$$ pair, 즉 입력 데이터 $$X$$와 이에 해당하는 정답 $$W$$를 가지고 있다고 합시다. 


그리고 우리는 정의한 모델을 통해 어떤 학습 데이터에 존재하지 않는 $$x^{\ast}$$에 대응하는 $$w^{\ast}$$를 찾고 싶습니다.


'$$x^{\ast}$$에 대응하는 $$w^{\ast}$$' 이는 수식적으로 아래와 같이 표현할 수 있습니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,W) $$</center>

이는 marginalization 테크닉을 통해 아래와 같이 나눌 수 있습니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,W) = \int Pr(w^{\ast}|x^{\ast},\phi) Pr(\phi|X,W) d\phi $$</center>

```
우리는 위의 수식에 대해서 한번 제대로 생각해볼 필요가 있습니다. 위의 오른쪽 적분 식이 의미하는 바는 뭘까요?
왼쪽의 p(y*|x*,\phi)는 우리가 정의한 모델 (여기서는 가우시안 분포의 파라메터, 더 나아가서는 뉴럴 네트워크의 파라메터들)을 통해 예측한 결과 분포 이고
오른쪽의 p(\phi|X,W)는 데이터를 통해 찾아낸, 데이터와 걸맞는 모델 파라메터의 분포 입니다.

원래 같았으면 예를들어 전체 데이터에 대해서 정의한 likelihood p(y|x,\phi) 분포에서 이것의 값을 최대로 하는 \phi 딱 '한개의 점'만 구하면 됐는데

이 두개를 곱한것을 가능한 모델 파라메터에 대해서 모두 적분한다... 

우선 조금 더 진행해보겠습니다.
```

우리는 이미 $$Pr(w \mid x,\phi)$$와 $$Pr(\phi \mid X,W)$$에 대해 정의를 했습니다. 

계속해서 전개해보겠습니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,w) = \int Norm_{w^{\ast}}[\phi^T x^{\ast},\sigma^2] Norm_{\phi}[\frac{1}{\sigma^2} A^{-1}Xw, A^{-1}] d\phi $$</center>

<center>$$ = Norm_{w^{\ast}}[\frac{1}{\sigma^2}x^{\ast T}A^{-1}Xw,x^{\ast T}A^{-1}x^{\ast} + \sigma^2] $$</center>

<center>$$ where \space A = \frac{1}{\sigma^2} XX^T + \frac{1}{\sigma_p^2}I $$</center>

이렇게 구한 식은 우리가 데이터 집합 $$(X,W)$$ 와 적절히 가정한 prior,likelihood분포가 있다면 어떤 테스트 데이터 x가 들어왔을때 그때의 output w의 분포를 파라메터를 추정하지 않고 나타낼 수 있다는 겁니다.

여기서 우리는 마지막 솔루션이 $$\sigma$$에만 종속되어있다는 것을 알 수 있는데요, 일반적으로 이 값들은 정해져있거나 (fixed variance) 그렇지 않을 경우 marginal likelihood를 최적화 하는 방법으로 (maximum likelihood) 구해놓은 뒤에 사용합니다. (구해놓으면 그 뒤로는 바로바로 사용 가능 하다는 소리)

![reg2_6](/assets/images/regression/reg2_6.png)

a)는 추정하고자 하는 파라메터 $$\phi_0,\phi_1$$의 분포를 나타내는 것입니다.
원래 MAP는 여기서 최대가 되는 값 하나만을 학습을 통해 구했으나, 이제는 b)처럼 가능한 파라메터 $$\phi_{1}$$, $$\phi_{2}$$, $$\phi_{3}$$ ... 에 대해서 모두 생각을 해보자는 것이죠.


b)는 파라메터 $$\phi_0,\phi_1$$가 어떤 값이냐에 따라서 선형 회귀의 직선이 어떻게 표현되는지를 나타냅니다.


c)는 말 그대로 위의 식 처럼 가능한 파라메터 $$\phi$$에 대해서 모두 적분한 결과입니다.  

마지막으로 우리가 구한 사후 분포의 mean에 관한 수식을 전부 풀어쓰면 아래와 같습니다.

$$ Pr(w^{\ast} \vert x^{\ast}, X, W) = Norm_w[ \frac{\sigma_p^2}{\sigma^2} x^{\ast T} X w - \frac{\sigma_p^2}{\sigma^2} x^{\ast T} X (X^TX + \frac{\sigma^2}{\sigma_p^2} I)^{-1} X^TXw, \space \sigma_p^2 x^{\ast T} x^{\ast} - \sigma_p^2 x^{\ast T} X (X^TX + \frac{\sigma^2}{\sigma_p^2} I)^{-1} X^T x^{\ast} + \sigma^2 ] $$








## <mark style='background-color: #fff5b1'> 점 추정? 분포 추정? </mark>

ML 과 MAP는 각각 $$likelihood$$와 $$posterior(likelihood \times prior)$$ 분포를 구한뒤 분포의 가장 큰 값일 때의 파라메터를 찾는 것입니다.

즉 이를 '점 추정' 한다고 할 수 있습니다.

반면에 베이지안 관점에서는 테스트 데이터에 대한 정답을 추론할 때 마다 적분을 하기 위한 점이 아닌 posterior분포를 다 사용하는, '분포 추정'을 한다고 할 수 있습니다.

하지만 만약 우리가 가지고 있는 데이터가 많아지다못해 무한대에 가까워지면 이는 점점 posterior 분포를 어느 한 점을 나타내는 delta function에 가깝게 만듭니다. 
이 때의 인퍼런스를한 결과는 점 추정을 하는 ML, MAP와 동일해집니다. (최대값 찾는거랑 똑같으니)

![reg2_7](/assets/images/regression/reg2_7.png)
*Fig. 일반적인 선형회귀의 ML 솔루션은 MSE Loss의 최소값*

![reg2_8](/assets/images/regression/reg2_8.png)
![reg2_9](/assets/images/regression/reg2_9.png)
![reg2_10](/assets/images/regression/reg2_10.png)
![reg2_11](/assets/images/regression/reg2_11.png)
*Fig. 데이터가 많아질수록 posterior가 delta function 가까워지는 모습*








## <mark style='background-color: #fff5b1'> Why Integral over all parameter? </mark>

위에서부터 글을 읽으시면서 "근데 왜 marginalization을 해야 하는거지? 왜 적분을 해야하는거지?"라는 생각을 하시는 분이 계실 것 같습니다.


그것은 이렇게 해석할 수 있을 것 같습니다.


본래 우리가 알고 싶은 것은 $$Pr(Y \mid X)$$ 분포입니다. 


그래서 완벽한 분포를 알고 있다면 여기에 결과를 알고싶은 데이터 포인트 $$x^{\ast}$$를 집어넣으면 당연히 모든 데이터에 대한 분포를 설명하는 $$Pr(Y \mid X)$$를 가지고 있기 때문에
우리는 $$Pr(y^{\ast} \mid x^{\ast} X)$$ 라는 정답을 알 수 있습니다.


하지만 우리는 주어진 문제에 대해 모든 데이터셋을 가지고 있지 않고 학습 데이터만을 가지고 있죠.


우리는 문제를 완벽하게 설명하는 함수가 무엇인지는 알 수 없지만 이와 유사한 근사 함수라도 알고 싶은겁니다. 
그래서 $$\theta$$ 라는 파라메터를 도입해 $$Pr(Y \mid X, \theta)$$ 라는 분포를 만들어 내고, 가장 그럴듯하게 데이터를 설명하는 분포의 파라메터 $$\theta$$를 찾아내는 겁니다..


하지만 $$\theta$$를 학습으로 찾아낸다 한들 임의로 정한 $$\theta$$ 분포를 가져다 쓰는게 마음에 들지 않아 이것마저 없애버리고 싶은겁니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,w) = \int Pr(w^{\ast}|x^{\ast},\phi) Pr(\phi|X,w) d\phi $$</center>

그래서 우리가 구하고 싶은 $$Pr(w^{\ast} \mid x^{\ast},X,w)$$ 를 위와 같은 식으로 나타내고 파라메터에 대해서 적분을 해서 없애버리는 겁니다.







## <mark style='background-color: #fff5b1'> Limitation </mark>

베이지안 방법으로 문제를 푸는 것은 상당히 괜찮은 접근으로 보입니다.


하지만 위에서 유도한 식과 달리 실제로는 $$posterior$$ 를 제대로 구하기 쉽지 않습니다. (위에서는 $$posterior$$ 에 대한 관계식 정도만 언급했음)

> 1. $$likelihood : p(x\mid\theta)$$ <br>
> 2. $$posterior \propto likelihood \times prior : p(\theta \mid x) \propto p(x \mid \theta)p(\theta)$$ <br> 

위와 같이 $$posterior$$ 를 간단한 관계식으로 표현했지만, 사실 $$posterior$$를 구하기 위한 Bayes' Rule은 조금 더 복잡합니다.







### <mark style='background-color: #dcffe4'> Bayes' Rule </mark>

<center>$$posterior :  p(\theta \mid X,W) = \frac{p(W \mid X, \theta)p(\theta)}{p(W \mid X)}$$</center>

<center>$$p(W \mid X) = \int p(W|X,\theta)p(\theta)d\theta$$</center>

위에서 보시는 거와 같이 사실은 베이즈 룰에서는 분모, normalizer term 혹은 model evidence 라고 하는 적분 term이 중요합니다.

이 적분을 계산하는 것은 'marginalising the likelihood over ω' 혹은 'marginal likelihood'라고도 합니다.


우리가 예시로 든 간단한 Bayesian Linear Regression 문제에서는 $$likelihood$$와 $$prior$$를 둘 다 conjugate 관계인 가우시안 분포로 적당히 가정했기 때문에 적분을 계산하는게 상대적으로 쉽습니다.


하지만 조금만 복잡해져도 (예를들어 basis function이 고정되어 있지 않은 Basis Function Regression이라던가, Neural Network 라던가...) 위의 적분은 계산하기가 쉽지 않습니다.

<center>$$ Pr(w^{\ast}|x^{\ast},X,w) = \int Pr(w^{\ast}|x^{\ast},\phi) Pr(\phi|X,w) d\phi $$</center>

즉, 위의 식에서 $$Pr(\phi \mid X,w)$$를 제대로 구할 수 없다는 것이죠.


이런 경우를 'The true posterior cannot usually be evaluated analytically.' 혹은 'the true posterior is intractable' 하다고 합니다.

이를 해결하기 위해서는 $$true \space posterior$$와 유사하지만 계산하기 쉬운 분포로 근사(approximate)를 해서 문제를 풀게 됩니다.

이렇게 근사하는 방법들에는 Laplace Approximation, Variational Inference 등이 있습니다.


하지만 이렇게 베이지안 방법론을 간단한 회귀 문제가 아닌 비선형성이 포함된 현대의 딥러닝에 적용하는것은 문제가 있는데요, 이를 Practical하게 해결한 방법론들이 많이 존재합니다.

![reg2_12](/assets/images/regression/reg2_12.png)
*Fig. 신경망 (Neural Network NN) 에 분포 추정 방식을 적용한, 이른 바 Bayesian Neural Network (BNN)*
![reg2_13](/assets/images/regression/reg2_13.png)
*Fig. 가우시안 프로세스에 베이지안 방법론을 적용한 것은, 현대 딥러닝의 가장 심플하면서도 강력한 정규화 방법론 중 하나인 Dropout을 하는것과 같다는 것을 수식적으로 증명한 Yarin Gal*

 Regression 시리즈는 아니지만 나중에 기회가 되면 Bayesian Deep Learning에 대해서도 다뤄보도록 하겠습니다.







## <mark style='background-color: #fff5b1'> References </mark>

1. [Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.](http://www.computervisionmodels.com/)

2. [Gal, Yarin. "Uncertainty in deep learning." University of Cambridge 1, no. 3 (2016): 4.](https://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html)

3. [What My Deep Model Doesn't Know... from Yarin Gal](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)

4. [NYU Bayesian Deep Learning : Tutorials](https://wjmaddox.github.io/assets/BNN_tutorial_CILVR.pdf)
