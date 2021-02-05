---
title: Generative vs Discriminative Models
categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> Learning and Inference </mark>

간단하게 말해서 대부분의 머신러닝 알고리즘의 해를 구하는 매커니즘은 다음과 같다고 할 수 있을 것 같습니다.

> 1. 수학적으로 입력 x와 출력 y의 관계를 설명할 model을 설정한다.
> 2. 학습 데이터 $$x_i,y_i$$들을 통해서 model의 파라메터를 학습한다. (Learning Algorithm)
> 3. 학습이 끝난 후 모델을 이용해 주어진 테스트 입력 x에 대한 $$Pr(w \mid x)$$ 를 구한다. (Inference Algorithm)

어떤 방식(ML, MAP, Bayesian)으로 어떤 분포를 학습하던 (x에대한 분포 $$Pr(x \mid y)$$ 이던 y에 대한 분포 $$Pr(y \mid x)$$이던) 목적은 곧 '어떻게 $$Pr(w \mid x)$$를 모델링하는가?' 입니다.

- <mark style='background-color: #fff5b1'> Generative vs Discriminative Models </mark>

둘의 차이에 대해서 얘기하기 전에 notation을 다시 정하도록 하겠습니다.

> <mark style='background-color: #dcffe4'> Notation </mark> <br>
> $$ x $$ : input state, 데이터 입력값 <br>
> $$ w $$ : world state, x에 대응하는 값 <br>
> $$ \theta $$ : parameter, 우리가 알고싶은, 추정하려는 값 <br>

Generative Model은 $$Pr(x \mid w)$$ 를 모델링하고


Discriminative Model은 $$Pr(w \mid x)$$ 를 모델링 하는 것이 목적입니다.


하지만 위에서도 말했듯 결국 추론시에 필요한 것은 '$$Pr(w \mid x)$$'가 됩니다.

- <mark style='background-color: #fff5b1'> Generative Model </mark>

과연 '어떻게 $$Pr(w \mid x)$$를 모델링하는가?' 그에 대한 첫번째 방법으로 $$Pr(x \mid y)$$를 모델링 하는 방법이 있습니다.

이러한 모델을 Generative Model, 생성 모델이라고 하는데, 이런 경우 먼저

> 1. $$x$$에 대해 적절한 prior를 고른다. <br>
> 2. $$Pr(x \mid w,\theta)$$를 정한다. <br>

```
주의!, 여기서 Prior는 학습하고자하는 파라메터에 대한 prior가 아닙니다, x의 분포입니다. 용어가 좀 혼용돼 햇갈리실 
```


그리고 파라메터를 학습하고 추론하는것은 아래와 같습니다.

> 1. Learning algorithm : 입력 $$x$$에 대해 적절한 prior $$Pr(x)$$를 고르고, $$x,w$$에 대해서 $$Pr(x \mid w)$$를 학습한다.<br>
> 2. Inference algorithm : 출력 $$w$$에 대한 $$Pr(w)$$를 정의하고 Bayes' Rule을 통해 $$Pr(w \mid x)$$를 계산한다. <br>

<center>$$ Bayes' \space Rule : Pr(w \mid x) = \frac{Pr(x \mid w)Pr(w)}{\int{Pr(x \mid w)}{Pr(w)}dw} $$</center>

아마 여기서 의문점이 하나 생기실 겁니다. (안생기셨나요? 저는 생겼습니다...)


어차피 학습 다 끝나고 우리에겐 새로운 입력 데이터 x 를 모델에 넣어서 결과를 내줄 $$Pr(y \mid x)$$가 필요한데 왜 굳이 $$Pr(x \mid y)$$를 모델링 해야 할까?


이에 대한 답은 아래에 예시를 들면서 설명을 해보도록 하도록 하고 우선은 넘어가도록 하겠습니다.

- <mark style='background-color: #fff5b1'> Discriminative Model </mark>

Discriminative Model, 판별 모델의 경우

> 1. $$w$$에 대해 적절한 prior를 고른다. <br>
> 2. $$Pr(w \mid x,\theta)$$를 정한다. <br>

학습과 추론은

> 1. Learning algorithm : 입력 $$w$$에 대해 적절한 prior $$Pr(w)$$를 고르고, $$x,w$$에 대해서 $$Pr(w \mid x)$$를 학습한다.<br>
> 2. Inference algorithm : 학습 자체를 $$Pr(w \mid x)$$에 대한 분포에 대해 했기 때문에 바로 테스트 데이터 x를 넣는다.

여기서는 $$Pr(x \mid y)$$를 바로 구하게 됐으니 마음이 편안합니다.


하지만 사실 생성 모델, 판별 모델 전부 말로는 와닿지 않습니다. 그래서 몇가지 예시(회귀,분류 문제)를 들어 두 모델의 차이를 이해해 보도록 하겠습니다. 

- <mark style='background-color: #fff5b1'> Example 1 : Regression </mark>

- <mark style='background-color: #dcffe4'> Regression : Discriminative </mark>

Discriminative Regression Model의 경우를 살펴봅시다.

> 1. $$w$$에 대해 적절한 prior로 가우시안 분포를 고른다. <br>
> 2. $$Pr(w \mid x,\theta)$$를 $$ Pr(w \mid x,\theta) = Norm_w[\theta^T x, \sigma^2] $$라고 정의한다. <br>
> 3. 학습할 파라메터는 $$\theta, \sigma^2$$ 이다.

(이런 모델은 Linear Regression이라고 부른다.)

![dis_reg1](https://user-images.githubusercontent.com/48202736/106454584-4e9ec280-64ce-11eb-9b83-2dea0deed256.png)
{: style="width: 60%;" class="center"}
 
이런경우 앞서 배운것과 마찬가지로 ML, MAP, Bayesian Aprroach 등으로 파라메터를 학습하면 됩니다.


그리고 애초에 $$Pr(w \mid x,\theta)$$에 대해 가우시안 분포라고 정의하고, 그 파라메터를 학습한것이기 때문에


Inference를 할 때도 x 데이터를 넣고 y의 자표를 읽으면 끝입니다. (아래 그림 참조)

![dis_reg2](https://user-images.githubusercontent.com/48202736/106454596-552d3a00-64ce-11eb-8cd0-91f999bbd3bb.png)
{: style="width: 60%;" class="center"}

- <mark style='background-color: #dcffe4'> Regression : Generative </mark>


Generative Regression Model의 경우를 살펴봅시다.

> 1. $$x$$에 대해 적절한 prior로 가우시안 분포를 고른다. <br>
> 2. $$Pr(x \mid w,\theta)$$를 $$ Pr(x \mid w,\theta) = Norm_x[\theta^T w, \sigma^2] $$라고 정의한다. <br>
> 3. 학습할 파라메터는 $$\theta, \sigma^2$$ 이다.

![gen_reg1](https://user-images.githubusercontent.com/48202736/106454606-5cecde80-64ce-11eb-9ddb-24942a10c01a.png)
{: style="width: 60%;" class="center"}

판별 모델과 거의 유사하지만, 우리가 학습하고자 정의한 분포가 $$Pr(x \mid w,\theta)$$ 라는게 중요합니다.


w(혹은 편하게 y) 값이 1.72일 때 그 때의 학습 데이터 x가 어떻게 퍼져있는지 그 분포를 학습하는 것이라고 생각할 수 있습니다.


(후에 분류 문제에서도 생성,판별 모델에 대해서 생각해 볼 건데, 그 경우를 예로 들어보자면, 학습 데이터에 '고양이' 라고 레이블 된 사진들의 분포가 어떻게 생겼는지를 (뚱뚱한 고양이일 확률은 몇, 얼룩 고양이는 몇 등등) 학습한다고 생각하면 될 것 같습니다.) 

![gen_reg2](https://user-images.githubusercontent.com/48202736/106454609-5eb6a200-64ce-11eb-8000-52cc6e2466e5.png)
{: style="width: 60%;" class="center"}

하지만 우리가 앞서 말한 것처럼 어떤 모델을 사용하던, 두 모델 모두 관심사는 '어떻게 $$Pr(w \mid x)$$를 모델링하는가?' 입니다.


즉 우리는 Pr(x \mid w)에 대해서 학습했기 때문에 학습이 다 끝나고 입력 데이터 x를 넣고 결과를 확인하는 추론시에는 Pr(w \mid x)에 대해 알아야 한다는 거죠.


그렇게 하기 위해서 Bayes' Rule을 사용한다고 했는데 베이즈 룰은 다음과 같습니다.

<center>$$ Bayes' \space Rule : Pr(w \mid x) = \frac{Pr(x \mid w)Pr(w)}{\int{Pr(x \mid w)}{Pr(w)}dw} $$</center>

![gen_reg3](https://user-images.githubusercontent.com/48202736/106454612-5f4f3880-64ce-11eb-897e-cb3cbb0602b7.png)

학습을 통해 x의 분포인 $$Pr(x \mid w)$$는 이미 구했고, $$Pr(w)$$ 는 학습 데이터로부터 바로 구할 수 있습니다. (위의 그림처럼)


베이즈 룰을 사용하기 위해 이 두 분포를 곱하면 공식의 분자인 $$Pr(x \mid w)Pr(w)$$, 즉 학습 데이터 x,y의 결합분포 $$Pr(w,x)$$가 됩니다.

($$ Pr(x,y) = Pr(x \mid y)Pr(y) = Pr(y \mid x)Pr(x) $$ 이기 때문)

![joint1](https://user-images.githubusercontent.com/48202736/106454699-81e15180-64ce-11eb-9d5a-1546c71d246a.png)
{: style="width: 60%;" class="center"}

![gen_reg4](https://user-images.githubusercontent.com/48202736/106454619-624a2900-64ce-11eb-99d5-fbd9a3e7e5b1.png)
{: style="width: 60%;" class="center"}

결과적으로 우리는 $$Pr(x \mid w)$$를 학습했지만 베이즈 룰을 통해 $$Pr(w \mid x)$$를 구할 수 있습니다.

![gen_reg5](https://user-images.githubusercontent.com/48202736/106454622-637b5600-64ce-11eb-8ea5-742d2205fe1e.png)


생성 모델이라고 이름이 붙혀진 것은 우리가 $$Pr(x \mid w)$$를 학습하기 때문에 입력 데이터의 분포를 알아냄으로써 분포로 부터 학습 데이터에 없는 $$x$$ 를 샘플링 할 경우 유의미한 데이터를 얻어낼 수 있다 (아무 분포에서 막 샘플링한게 아니니까) 라고 하는데 사실 아직도 감이 잘 안옵니다... 


이제 분류 문제에 대해서도 똑같이 적용해보도록 하겠습니다.


- <mark style='background-color: #fff5b1'> Example 2 : Classification </mark>

2개의 클래스를 판별하는 이진 분류 모델에 대해서 생각해보도록 하겠습니다.

- <mark style='background-color: #dcffe4'> Classification : Discriminative </mark>

Discriminative Classification Model의 경우를 살펴봅시다.

> 1. $$w$$에 대해 적절한 prior로 베르누이 분포를 고른다. <br>
> 2. $$Pr(w \mid x)$$를 $$ Pr(w \mid x) = Bern_w[\frac{1}{1+exp[-\theta_0 -\theta_1 x]}] $$라고 정의한다. <br>
> 3. 학습할 파라메터는 $$\theta_0,\theta_1$$ 이다.

![dis_class1](https://user-images.githubusercontent.com/48202736/106454645-6c6c2780-64ce-11eb-8a8a-e8981f35bb92.png)
{: style="width: 40%;" class="center"}

마찬가지로 ML, MAP, Bayesian 어떤 방법으로도 파라메터를 찾아낼 수 있습니다.

![dis_class2](https://user-images.githubusercontent.com/48202736/106454648-6d9d5480-64ce-11eb-84ef-435eaf89e822.png)

Regression 때와 마찬가지로 $$Pr(w \mid x)$$ 에 대한 분포를 정의하고 그 분포의 파라메터를 학습했으니, 바로 입력 데이터 x를 넣어 추론할 수 있습니다.

- <mark style='background-color: #dcffe4'> Classification : Generative </mark>

Generative Classification Model의 경우를 살펴봅시다.


이번에도 마찬가지로 $$Pr(x \mid w)$$를 학습하겠죠?

> 1. $$x$$에 대해 적절한 prior로 가우시안 분포를 고른다. <br>
> 2. $$Pr(x \mid w)$$를 $$ Pr(x \mid w) = Norm_x[\mu_w, \sigma_w^2] $$라고 정의한다. <br>
> 3. 학습할 파라메터는 $$\mu_0,\mu_1,\sigma_0^2,\sigma_1^2$$ 입니다.

![gen_class1](https://user-images.githubusercontent.com/48202736/106454667-742bcc00-64ce-11eb-844b-4d44d8170c2e.png)
{: style="width: 60%;" class="center"}

현재 이진 분류 문제를 예시로 들었으니, 우리가 찾아야 할 것은 continuous한 입력에 대해 가우시안 분포로 모델링 했기 때문에 각 클래스 당 mean,variance 1개 씩 
총 $$\mu_0,\mu_1,\sigma_0^2,\sigma_1^2$$ 4개가 됩니다.

![gen_class2](https://user-images.githubusercontent.com/48202736/106454677-768e2600-64ce-11eb-82f0-006524ba9f6f.png)
{: style="width: 60%;" class="center"}

이제 학습 데이터에서 바로 정답 클래스의 분포 $$Pr(w)$$를 구할 수 있습니다. 이진 분류 문제이기 때문에 정답 분포는 $$[0,1,0,1,1,1,0,...,0]$$ 이런식으로 이산적이겠죠? 베르누이 분포로 모델링 할 수 있습니다.

![gen_class3](https://user-images.githubusercontent.com/48202736/106454681-77bf5300-64ce-11eb-8245-aff4d1dd1160.png)
{: style="width: 40%;" class="center"}

그리고 저희가 원하는것은 마찬가지로 $$Pr(w \mid x)$$ 이기 때문에 마찬가지로 베이즈 룰을 이용해 구해보면 다음과 같은 결과를 얻을 수 있습니다.

![gen_class4](https://user-images.githubusercontent.com/48202736/106454682-78f08000-64ce-11eb-9f4c-f4bd67027b2a.png)

$$Pr(x \mid w)$$를 학습한 뒤 구한 $$Pr(w \mid x)$$ 분포와 (좌) vs 다이렉트로 $$Pr(w \mid x)$$를 학습한 분포가 조금 차이가 있어 다.(우)

![gen_class4 5](https://user-images.githubusercontent.com/48202736/106454685-7a21ad00-64ce-11eb-9095-5c201c9edb86.png)
{: style="width: 70%;" class="center"}


어떤 차이가 있는지 감이 오셨나요...?


이제 우리는 어떤 경우에 Generative (좌) vs Discriminative (우) 를 선택할지를 생각해 봐야 합니다.

![gen_class6](https://user-images.githubusercontent.com/48202736/106454687-7b52da00-64ce-11eb-97a5-1b7bb971a032.png)
{: style="width: 90%;" class="center"}

- <mark style='background-color: #fff5b1'> Intuition...? </mark>

그전에 혹시 글의 서두에서 언급했던 궁금증이 생각 나시나요?

> 생성 모델 : 어차피 학습 다 끝나고 우리에겐 새로운 입력 데이터 x 를 모델에 넣어서 결과를 내줄 $$Pr(y \mid x)$$가 필요한데 왜 굳이 $$Pr(x \mid y)$$를 모델링 해야 할까?

우리가 지금까지 두 가지 타입의 모델에 대해서 살펴봤기 때문에 이제는 이 질문에 대해서 다시 생각해 볼만 합니다.


$$Pr(x \mid y)$$를 모델링한다는 것은 예를들어 분류 문제를 생각해보면, 각 분류 클래스(개,고양이 두개 라고 칩시다) 에 대한 분포가 어떻다를 모델링하는 겁니다.


즉 우리는 $$Pr(x \mid y)$$를 모델링 함으로써 학습 데이터 (예를들어 고양이 사진 500장, 개 500장)를 통해 고양이이 라는 클래스의 이미지들이 가지는 분포가 어떤지를 알 수 있게 된다는 겁니다.


반면에 다이렉트로 $$Pr(y \mid x)$$ 를 학습한 것은 단순히 클래스 두 개를 분류할 Decision Boundary를 학습하는 것일 뿐입니다.

![image](https://user-images.githubusercontent.com/48202736/105727674-ab4e2a80-5f6e-11eb-863a-ce606a8e29b4.png)
{: style="width: 60%;" class="center"}

(이미지 출처 : [link](https://medium.com/@jordi299/about-generative-and-discriminative-models-d8958b67ad32))

![gen_class2](https://user-images.githubusercontent.com/48202736/106454677-768e2600-64ce-11eb-82f0-006524ba9f6f.png)
{: style="width: 60%;" class="center"}

(위의 두 그림은 입력 데이터 x가 2차원, 1차원일 때의 예시입니다.)

첫번째 그림에서는 고양이를 나타내는 분포와 개를 나타내는 분포가 한 장에 표시되어있지만 어쨌든 고양이만 생각해보겠습니다.

우리는 고양이의 경우에 해당하는 이미지들의 분포를 학습을 통해 알아냈습니다. 이 분포가 일정 수준 이상이라고 생각할 때, 이 분포로부터 샘플링으로 이미지를 한장 뽑아보면 어떻게 될까요?

개 사진이 나올까요? 아니면 이도저도 아닌 이미지가 나올까요? 아마 그래도 학습 데이터와 유사한 혹은 어쩌면 학습 데이터에서 찾아볼 수 없었던 뚱뚱한? 고양이가 나올 수도 있습니다.


생성 모델의 이점은 (입력) 데이터의 진짜 분포를 학습을 통해 찾아내고 이를 통해 데이터를 만들어 낼 수 있다는 데 있습니다. 


사실 모델의 이름이 Generative, 생성인 이유가 여기에 있습니다.


```
* 추가적으로 1차원 연속 데이터 x에 대한 예시가 있습니다.
```

![An-illustration-of-the-difference-between-the-generative-and-discriminative-models-in](https://user-images.githubusercontent.com/48202736/105729032-2532e380-5f70-11eb-9b86-37afaf0c2602.png)

(이미지 출처 : [link](https://www.researchgate.net/figure/An-illustration-of-the-difference-between-the-generative-and-discriminative-models-in_fig9_319093376))

(비슷한 내용이라 더 자세하게 쓰지는 않겠습니다. 다만 위의 예시에서는 Generative Model $$Pr(x \mid y)$$를 학습하고 베이즈 룰을 통해 $$Pr(y \mid x)$$를 만들어내지 않고도 단순히 두 클래스의 분포에 테스트 입력 데이터를 넣었을 때 더 높은 확률을 나타내는 클래스로 와인을 분류하는 방법을 사용합니다.)

- <mark style='background-color: #fff5b1'> Generative Model vs Discriminative Model Pros and Cons </mark>

- <mark style='background-color: #dcffe4'> Generative Model </mark>
  - <mark style='background-color: #ffdce0'> Pros </mark>
    - 1.각 클래스가 어떤 분포로 부터 근거했는지 근본적인 부분에 대해서 생각할 수 있다.
    - 2.찾은 분포를 통해 새로운 데이터를 생성 할 수 있다.
    - 3.베이지안 정리를 통해 p(x)의 주변 밀도까지 구할 수 있다. ($$\rightarrow$$ 이를 바탕으로 발생 확률이 낮은 새 데이터 포인트들을 미리 발견할 수 있다. 이런 데이터들은 낮은 정확도를 뱉을 것이기 때문에 이상점 검출 등에 사용 될 수 있다.)
    - 4.Unsupervised Learning에 적합하다.
  - <mark style='background-color: #ffdce0'> Cons </mark>
    - 1.보통 입력 데이터의 차원이 크기 때문에 그것을 모델링하기란 쉽지 않다. (모델 파라메터가 엄청 많아지고 계산량이 많아짐)
    - 2.일정 수준 이상의 제대로 된 분포를 찾기 위해서는 학습 데이터가 많이 필요하다.
    - 3.사후 확률을 계산하는데 영향을 미치지 않는 추가적인 정보가 많이 포함되어 있을 수도 있다.
  - Examples
    - Naive Bayes
    - Gaussian mixture model
    - Hidden Markov Models (HMM)
- <mark style='background-color: #dcffe4'> Discriminative Model </mark>
  - <mark style='background-color: #ffdce0'> Pros </mark>
    - 1.계산량이 적다.
    - 2.필요한 데이터도 적다. (물론 데이터가 많으면 좋겠지만 생성모델처럼 데이터 요구량이 엄청 많지는 않다.)
  - <mark style='background-color: #ffdce0'> Cons </mark>
    - 1.해석하기 어렵다.
    - 2.단순히 분류를 할 뿐 (decision boundary를 만들어낼 뿐) 데이터를 만들어 낼 수는 없다.
    - 3.Unsupervised Learning에 부적합하다.
  - Examples
    - Logistic regression
    - SVM
    - Neural Networks
    
어떤 모델이 더 우수하다고는 할 수 없습니다. (마치 ML과 Bayesian Approach중 무엇이 우수한지 알 수 없듯, 주장만 있을 뿐...) 


대부분의 경우에 Discriminative Model이 계산량이 적고 모델링 하기 쉽기 때문에 인기가 많지만 경우에 따라 Generative Model을 사용해야 할 때가 있을 수 있습니다.


그리고 현재도 인기가 많은 2015년 Ian Goodfellow가 제시한 GAN은 이렇게 학습하기 어렵고 큰 모델을 어떻게 학습할 것인가를 잘 풀어낸 하나의 예시라고 할 수 있습니다.


다음에 기회가 되면 많은 생성 모델에 대해서 집중적으로 다뤄보도록 하겠습니다! 

  

- <mark style='background-color: #fff5b1'> References </mark>

1. [Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.](http://www.computervisionmodels.com/)

2. [Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
