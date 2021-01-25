---
title: Generative vs Discriminative Models
categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> Learning and Inference </mark>

머신러닝의 매커니즘은 다음과 같.

> 1. 수학적으로 입력 x와 출력 y의 관계를 설명할 model을 설정한다.
> 2. 학습 데이터 $$x_i,y_i$$들을 통해서 model의 파라메터를 학습한다. (Learning Algorithm)
> 3. 학습이 끝난 후 모델을 이용해 주어진 테스트 입력 x에 대한 $$Pr(w|x)$$ 를 구한다. (Inference Algorithm)

- <mark style='background-color: #fff5b1'> Generative vs Discriminative Models </mark>

둘의 차이에 대해서 얘기하기 전에 notation을 다시 정하도록 하겠습니다.

> <mark style='background-color: #dcffe4'> Notation </mark> <br>
> $$ x $$ : input state, 데이터 입력값 <br>
> $$ w $$ : world state, x에 대응하는 값 <br>
> $$ \theta $$ : parameter, 우리가 알고싶은, 추정하려는 값 <br>

Generative Model은 $$Pr(x|w)$$ 를 모델링하고


Discriminative Model은 $$Pr(w|x)$$ 를 모델링 합니다.

- <mark style='background-color: #fff5b1'> Generative Model </mark>

두 모델 모두 관심사는 '어떻게 $$Pr(w|x)$$를 모델링하는가?' 입니다.

Generative Model, 생성 모델의 경우

> 1. $$x$$에 대해 적절한 prior를 고른다. <br>
> 2. $$Pr(x|w,\theta)$$를 정한다. <br>

학습과 추론은

> 1. Learning algorithm : 입력 $$x$$에 대해 적절한 prior $$Pr(x)$$를 고르고, $$x,w$$에 대해서 $$Pr(x|w)$$를 학습한다.<br>
> 2. Inference algorithm : 출력 $$w$$에 대한 $$Pr(w)$$를 정의하고 Bayes' Rule을 통해 $$Pr(w|x)$$를 계산한다. <br>

<center>$$ Bayes' \space Rule : Pr(w|x) = \frac{Pr(x|w)Pr(w)}{\integral{Pr(x|w)}{Pr(w)}dw} $$</center>

- <mark style='background-color: #fff5b1'> Discriminative Model </mark>

Discriminative Model, 판별 모델의 경우

> 1. $$w$$에 대해 적절한 prior를 고른다. <br>
> 2. $$Pr(w|x,\theta)$$를 정한다. <br>

학습과 추론은

> 1. Learning algorithm : 입력 $$w$$에 대해 적절한 prior $$Pr(w)$$를 고르고, $$x,w$$에 대해서 $$Pr(w|x)$$를 학습한다.<br>
> 2. Inference algorithm : 학습 자체를 $$Pr(w|x)$$에 대한 분포에 대해 했기 때문에 바로 테스트 데이터 x를 넣는다.


사실 생성 모델, 판별 모델 전부 말로는 와닿지 않아서 예제를 보는게 나을거 같습니다.

- <mark style='background-color: #fff5b1'> Example 1 : Regression </mark>

- <mark style='background-color: #dcffe4'> Regression : Discriminative </mark>

Discriminative Regression Model의 경우를 살펴봅시다.

> 1. $$w$$에 대해 적절한 prior로 가우시안 분포를 고른다. <br>
> 2. $$Pr(w|x,\theta)$$를 $$ Pr(w|x,\theta) = Norm_w[\theta^T x, \sigma^2] $$라고 정의한다. <br>
> 3. 학습할 파라메터는 $$\theta, \sigma^2$$ 이다.

(이런 모델은 Linear Regression이라고 부른다.)

![image](https://user-images.githubusercontent.com/48202736/105446589-c80c0900-5cb5-11eb-9671-fbaf131d6f33.png)

이런경우 앞서 배운것과 마찬가지로 ML, MAP, Bayesian Aprroach 등으로 파라메터를 학습하면 됩니다.


그리고 애초에 $$Pr(w|x,\theta)$$에 대해 가우시안 분포라고 정의하고, 그 파라메터를 학습한것이기 때문에


Inference를 할 때도 x 데이터를 넣고 y의 자표를 읽으면 끝입니다. (아래 그림 참조)

![image](https://user-images.githubusercontent.com/48202736/105446592-c93d3600-5cb5-11eb-8143-9c3ff6dbf0b4.png)

- <mark style='background-color: #dcffe4'> Regression : Generative </mark>


Generative Regression Model의 경우를 살펴봅시다.

> 1. $$x$$에 대해 적절한 prior로 가우시안 분포를 고른다. <br>
> 2. $$Pr(x|w,\theta)$$를 $$ Pr(x|w,\theta) = Norm_x[\theta^T w, \sigma^2] $$라고 정의한다. <br>
> 3. 학습할 파라메터는 $$\theta, \sigma^2$$ 이다.

![image](https://user-images.githubusercontent.com/48202736/105446601-d0fcda80-5cb5-11eb-9689-0c31ed7bde82.png)

판별 모델과 거의 유사하지만, 우리가 학습하고자 정의한 분포가 $$Pr(x|w,\theta)$$ 라는게 중요합니다.


w(혹은 편하게 y) 값이 1.72일 때 그 때의 학습 데이터 x가 어떻게 퍼져있는지 그 분포를 학습하는 것이라고 생각할 수 있습니다.


(후에 분류 문제에서도 생성,판별 모델에 대해서 생각해 볼 건데, 그 경우를 예로 들어보자면, 학습 데이터에 '고양이' 라고 레이블 된 사진들의 분포가 어떻게 생겼는지를 (뚱뚱한 고양이일 확률은 몇, 얼룩 고양이는 몇 등등) 학습한다고 생각하면 될 것 같습니다.) 

![image](https://user-images.githubusercontent.com/48202736/105446605-d35f3480-5cb5-11eb-86bb-8a33300dbd91.png)

하지만 우리가 앞서 말한 것처럼 어떤 모델을 사용하던, 두 모델 모두 관심사는 '어떻게 $$Pr(w|x)$$를 모델링하는가?' 입니다.


즉 우리는 Pr(x|w)에 대해서 학습했기 때문에 학습이 다 끝나고 입력 데이터 x를 넣고 결과를 확인하는 추론시에는 Pr(w|x)에 대해 알아야 한다는 거죠.


그렇게 하기 위해서 Bayes' Rule을 사용한다고 했는데 베이즈 룰은 다음과 같습니다.

<center>$$ Bayes' \space Rule : Pr(w|x) = \frac{Pr(x|w)Pr(w)}{\integral{Pr(x|w)}{Pr(w)}dw} $$</center>

![image](https://user-images.githubusercontent.com/48202736/105446609-d5c18e80-5cb5-11eb-8f82-fd620423d775.png)

학습을 통해 x의 분포인 $$Pr(x|w)$$는 이미 구했고, $$Pr(w)$$ 는 학습 데이터로부터 바로 구할 수 있습니다. (위의 그림처럼)


베이즈 룰을 사용하기 위해 이 두 분포를 곱하면 공식의 분자인 $$Pr(x|w)Pr(w)$$, 즉 학습 데이터 x,y의 결합분포 $$Pr(w,x)$$가 됩니다.

($$Pr(x,y) = Pr(x|y)Pr(y)$$ 이기 때문)

![image](https://user-images.githubusercontent.com/48202736/105656572-e405e980-5f05-11eb-9bae-1dee177ed5c7.png)

![image](https://user-images.githubusercontent.com/48202736/105446616-d8bc7f00-5cb5-11eb-8c3f-a1b1777f5df1.png)



![image](https://user-images.githubusercontent.com/48202736/105446621-db1ed900-5cb5-11eb-96ba-780fb7cfa3c7.png)

- <mark style='background-color: #fff5b1'> Example 2 : Classification </mark>

- <mark style='background-color: #dcffe4'> Classification : Discriminative </mark>

![image](https://user-images.githubusercontent.com/48202736/105446663-eeca3f80-5cb5-11eb-8ac0-c34bc071995b.png)
![image](https://user-images.githubusercontent.com/48202736/105446667-f12c9980-5cb5-11eb-8c6b-14a664b2b5d4.png)

- <mark style='background-color: #dcffe4'> Classification : Generative </mark>


![image](https://user-images.githubusercontent.com/48202736/105446672-f38ef380-5cb5-11eb-9b8b-db2238d1cb6d.png)
![image](https://user-images.githubusercontent.com/48202736/105446675-f558b700-5cb5-11eb-8bd8-5e8ad17e4c43.png)

![image](https://user-images.githubusercontent.com/48202736/105446675-f558b700-5cb5-11eb-8bd8-5e8ad17e4c43.png)
![image](https://user-images.githubusercontent.com/48202736/105446703-01dd0f80-5cb6-11eb-9b33-ed3e6e74c6a8.png)
![image](https://user-images.githubusercontent.com/48202736/105446708-06a1c380-5cb6-11eb-8d13-c549738a7943.png)


- <mark style='background-color: #fff5b1'> References </mark>

1. [Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.](http://www.computervisionmodels.com/)

2. [Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
