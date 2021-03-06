---
title: (OpenAI Spinning Up Series) - Introduction
categories: ReinforcementLearning
tag: [RL]

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> Why OpenAI Spinnin Up? </mark>

사실 예전부터 강화학습을 공부해보고 싶었습니다. 


2014,15년부터 이미 강화학습을 고독하게 독파 분들의 말씀을 들어보면 'latte는 제대로 된 자료 거의 없었어... david silver, sutton 책 보는게 다인데 이해도 잘 ' 라고 합니다. 


그에 반해 지금은 사실 영어 자료도 더 다양해지고(silver, sutton책 받고 Berkley RAIL 의 CS285, stanford의 CS231,224 같은 강화학습버전 CS234 등...) 그것들을 번역한 자료 도 많아졌습니다. (서튼 책의 번역본, silver의 강의를 쉽게 풀어서 다시 설명해주시는 NCsoft의 노승은님의 youtube channel 등, 물론 다른 책도 많습니다.)

<br>

하지만 이런것들은 대부분 제한적으로 RL 을 커버하거나(특히 Deep RL부분이 너무 적음) 초심자 입장에서 너무 헤비하거나(CS285가 이 경우인데 최신 학기 수업까지 업로드 되고 있어 굉장히 자료도 좋고 강의를 하는 Sergey Levine이 세계적인 RL 석학이라 너무 좋지만 초심자 입장에선 조금 버겁다고 느꼈습니다) 하는 문제가 있다고 생각이 들었습니다. (물론 딥러닝을 시작하려 해도 머신러닝의 개념이 필요하듯 RL의 개념을 모르고 지나갈 순 없지만...)

<br>

OpenAI가 Spinning up 자료를 만든것은 위에서 말한것과 비슷하게 심층 강화학습(Deep RL)을 배우기 위한 자료가 마땅치 않기 때문이라고 얘기하고 있습니다. 


아래는 "Why OpenAI Built This"의 일부입니다.  

```
"At OpenAI, we believe that deep learning generally—and deep reinforcement learning specifically—will play central roles in the development of powerful AI technology. ... we encourage everyone who asks this question to study these fields.

However, while there are many resources to help people quickly ramp up on deep learning, deep reinforcement learning is more challenging to break into. ... Beyond that, they need both a high-level view of the field—an awareness of what topics are studied in it, ...

There is not yet a standard deep RL textbook, ...

And learning to implement deep RL algorithms is typically painful, ..."
```

OpenAI가 이 자료에서 커버하는 내용들은 다음과 같습니다. 


<mark style='background-color: #ffdce0'> 1. a short introduction to RL terminology, kinds of algorithms, and basic theory, </mark>

<mark style='background-color: #ffdce0'> 2. an essay about how to grow into an RL research role, </mark>

<mark style='background-color: #ffdce0'> 3. a curated list of important papers organized by topic, </mark>

<mark style='background-color: #ffdce0'> 4. a well-documented code repo of short, standalone implementations of key algorithms, </mark>

<mark style='background-color: #ffdce0'> 5. and a few exercises to serve as warm-ups. </mark>


(부제 자체가 Intro to Deep RL이라 그런지 RL은 핵심적인 terminology와 역사적인 알고리즘에 대해서만 다루고, 그 뒤 봐야할 Deep RL 페이퍼리스트도 정리해주고, 본인들 코드도 살짝씩 보여주면서 넘어가기 때매 이 부분이 맘에들었습니다.)


- <mark style='background-color: #fff5b1'> Spinnin Up Package </mark>

[Spinnin Up Package](https://spinningup.openai.com/en/latest/user/installation.html) 링크에 들어가면 각 OS마다 Deep RL 알고리즘을 사용할 수 있는 패키지를 설치할 수 있고 그 안에 뭐가 들어가있는지를 알 수 있습니다.


내용물은 다음과 같습니다.

- Algorithms
  - What’s Included
  - Why These Algorithms?
    - The On-Policy Algorithms
    - The Off-Policy Algorithms
  - Code Format
    - The Algorithm Function: PyTorch Version
    - The Algorithm Function: Tensorflow Version
    - The Core File

- <mark style='background-color: #dcffe4'> What’s Included </mark>

Spinning up 패키지에는 다음 알고리즘이 들어가 있다고 .


<mark style='background-color: #ffdce0'> - Vanilla Policy Gradient (VPG) </mark>

<mark style='background-color: #ffdce0'> - Trust Region Policy Optimization (TRPO) </mark>

<mark style='background-color: #ffdce0'> - Proximal Policy Optimization (PPO) </mark>

<mark style='background-color: #ffdce0'> - Deep Deterministic Policy Gradient (DDPG) </mark>

<mark style='background-color: #ffdce0'> - Twin Delayed DDPG (TD3) </mark>

<mark style='background-color: #ffdce0'> - Soft Actor-Critic (SAC) </mark>


각각의 알고리즘은 간단한 MLP actor-critic, fully-observed, non-image-based environment에서 수행되었다고 합니다.
그리고 알고리즘마다 Pytorch, Tensorflow 두가지 버전이 있어서 골라 쓰면 된다고 합니다. 


- <mark style='background-color: #dcffe4'> Why These Algorithms? </mark>

6개 정도의 Deep RL 알고리즘을 골라서 설명한다는 것 같은데 왜 그런지는 링크에서 확인하시면 좋을 것 같습니다.

[Why These Algorithms?](https://spinningup.openai.com/en/latest/user/algorithms.html)

- <mark style='background-color: #dcffe4'> Code Format </mark>

저는 파이토치 유저이기 때문에 파이토치의 코드 포맷만 한번 보겠습니다.

```

# 랜덤 시드(얼핏 들은 바로는 중요하다고 합니다.) 설정 및 실험환경 설정
1.Logger setup
2.Random seed setting
3.Environment instantiation

# 알고리즘(네트워크) 설정 
4.Constructing the actor-critic PyTorch module via the actor_critic function passed to the algorithm function as an argument

# temporal
5.Instantiating the experience buffer

# 학습을 위한 Loss function, optimizer, logger 등 설정
6.Setting up callable loss functions that also provide diagnostics specific to the algorithm
7.Making PyTorch optimizers
8.Setting up model saving through the logger
9.Setting up an update function that runs one epoch of optimization or one step of descent

# run epoch
10.Running the main loop of the algorithm:
  a.Run the agent in the environment
  b.Periodically update the parameters of the agent according to the main equations of the algorithm
  c.Log key performance metrics and save agent

```

전반적인 코드 포맷이 딥러닝 네트워크를 구성해서 학습할 때와 크게 다른 것 같지 않습니다.

여기 까지 해서 왜 OpenAI Spinning Up 을 통해 Deep RL을 살펴보게 되었는지와 앞으로 익히게 될 내용이 무엇인가에 대해 알아봤습니다.

다음 부터는 본격적으로 RL의 기본적인 개념 살짝과 바로 Deep RL의 알고리즘을 약간의 수식과 코드와 함께 살펴보도록 하겠습니다. 
