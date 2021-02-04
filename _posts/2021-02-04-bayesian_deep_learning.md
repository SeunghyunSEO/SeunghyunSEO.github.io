---
title: Bayesian Deep Learning
categories: DeepLearning
tag: [tmp]

toc: true
toc_sticky: true
---


- <mark style='background-color: #fff5b1'> Bayesian Approach </mark>

머신러닝에서 Bayesian Approach는 무엇을 의미할까요?

일반적인 머신러닝, 딥러닝에서는 우리가 추정하고자하는 출력 분포의 파라메터를 '점 추정(A Point Estimation)' 하게 됩니다.

이는 likelihood만 가지고 Maximum likelihood를 하던 아니면 prior를 곱해 posterior를 구한 뒤 계산하는 Maximum A Posterior를 하던 마찬가지입니다.

둘 다 어떤 분포에서 (likelihood, posterior) 가장 큰 값을 나타내는 점 하나만을 최적화나 미분을 통해 구해버리니까요.

(예를 들어 선형회귀에서 우리가 찾고 싶은 게 mean을 나타내는 직선(gradient)라면 $$\hat{y}=f(x,\theta)$$ 이기 때문에 점 추정 결과는 $$\theta_1=0.16$$, $$\theta_2=1.2$$ 이렇게 딱 떨어진다는 겁니다.

하지만 이렇게 해서는 'over confident' 하다는 문제가 발생할 수 있습니다.

그림으로 다시 볼까요

- MLE for Linear Regression

아래는 Maximum lieklihood 방법을 통해 likelihood를 크게 만드는 파라메터 하나만을 '점 추정'한 선형회귀의 결과입니다. 

![mle1](https://user-images.githubusercontent.com/48202736/106867120-91e87380-6710-11eb-8a5e-4b989f8f3f9e.png)

- Bayesian Approach for Linear Regression

아래는 Bayesian 방법을 통해 Inference한 방법입니다. 

![bayesian1](https://user-images.githubusercontent.com/48202736/106867124-9319a080-6710-11eb-9503-1acc27d78ddf.png)

둘의 차이는 아래의 이미지를 보면 회귀 곡선이 데이터 포인트가 별로 없는 부분에 대해서는 variance가 엄청 크게 나타난다는 겁니다. 즉 (y값이 1일확률~3일 확률이 거의 같을 정도로 햇갈려 하는, 모델이 잘 모르겠다고 말하는것과 같죠) 


분류 문제를 볼까요? 여기서는 조금 더 직관적인 그림이 있는 것 같아 책의 그림이 아닌 다른 그림을 가져왔습니다.

![BNN](https://user-images.githubusercontent.com/48202736/106858019-5c3d8d80-6704-11eb-8a3a-82846ae6ddf7.png)
(이미지 출처 : [link](https://medium.com/aitrics/drug-discovery-2-%EA%B0%80%EC%83%81%ED%83%90%EC%83%89%EC%9D%84-%EC%9C%84%ED%95%9C-%EC%8B%A0%EB%A2%B0%ED%95%A0-%EC%88%98-%EC%9E%88%EB%8A%94-%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-7a4b4eb63106))

마찬가지로 MAP (ML과 다른점은 prior가 곱해진 점) 의 가장 큰 값만 취하는 점추정의 경우(위) decision boundary만 구할 수 있었기 때문에 학습이 끝난 뒤 별 모양의 데이터가 들어가면

모델이 자신있게 'class1'이라고 분류하는 모습을 볼 수 있습니다.

하지만 아래의 그림을 보면 파라메터의 posterior 분포를 적분하는, 그러니까 모든 파라메터에 대해서 선을 다 그어보는(?) bayesian이기에 별 모양의 데이터가 들어갔을 때 class1 : 0.6, class2 : 0.4 같은 결과를 보이는걸 알 수 있습니다.


(아래 그림은 아쉬워서? 이해를 돕기위해? 추가했습니다...)

- MLE for Classification

![mle_cls1](https://user-images.githubusercontent.com/48202736/106867129-944acd80-6710-11eb-92b1-4c6aa7ee132f.png)

- Bayesian Approach for Classification

![bayesian_cls1](https://user-images.githubusercontent.com/48202736/106867133-957bfa80-6710-11eb-8183-7f2ae9e97522.png)

이렇듯 베이지안 방법은 가능한 파라메터에 대해서 적분을 해야한다는 문제와 그리고 경우에 따라 (아니 대부분의 경우) posterior 계산이 어렵다는 문제가 있지만 설득력이 있는 결과를 제시합니다.

- <mark style='background-color: #fff5b1'> Neural Network </mark>

우리가 알고싶은 것은 Bayesian Neural Network (BNN) 나아가 Bayesian Deep Learning 이기 때문에, 이번에는 Neural Network에 대해서 알아보도록 하겠습니다.

- <mark style='background-color: #fff5b1'> Bayesian Neural Network (BNN) </mark>


- <mark style='background-color: #fff5b1'> Variational Inference </mark>


- <mark style='background-color: #fff5b1'> Bayesian Deep Learning </mark>


- <mark style='background-color: #fff5b1'> Model Uncertainty </mark>


- <mark style='background-color: #fff5b1'> References </mark>

1. [[Drug Discovery] #2 가상탐색을 위한 신뢰할 수 있는 인공지능](https://medium.com/aitrics/drug-discovery-2-%EA%B0%80%EC%83%81%ED%83%90%EC%83%89%EC%9D%84-%EC%9C%84%ED%95%9C-%EC%8B%A0%EB%A2%B0%ED%95%A0-%EC%88%98-%EC%9E%88%EB%8A%94-%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-7a4b4eb63106)

2. [Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." In international conference on machine learning, pp. 1050-1059. PMLR, 2016.](http://proceedings.mlr.press/v48/gal16.pdf)

3. [(Yarin Gal)THESIS: UNCERTAINTY IN DEEP LEARNINGLink to this paper](https://www.cs.ox.ac.uk/people/yarin.gal/website/thesis/thesis.pdf)

4. [Weight Uncertainty in Neural Networks](https://arxiv.org/pdf/1505.05424)

4. [NYU Bayesian Deep Learning : Tutorials](https://wjmaddox.github.io/assets/BNN_tutorial_CILVR.pdf)

5. [Yarin Gal's Blog](https://www.cs.ox.ac.uk/people/yarin.gal/website/publications.html)

6. [Bayesian Deep Learning NIPS Workshop](http://bayesiandeeplearning.org/)
