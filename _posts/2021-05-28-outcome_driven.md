---
title: (yet) Outcome-Driven Reinforcement Learning via Variational Inference
categories: Reinforcement_Learning_and_Deep_Reinforcement_Learning
tag: [DeepLearning]

toc: true
toc_sticky: true
---

본 논문은 강화학습계의 석학인 Sergey Levin과 베이지안 딥러닝 분야의 석학인 Yarin Gal이 같이 쓴 논문이다.

논문의 요지는 강화학습이 optimal policy를 trial and error를 통해서 얻는 과정이지만, 그러기 위해서는 reward를 어떻게 디자인하는가? 등에 대한 design decision이 너무 많이 들어간다는 겁니다.
그래서 본 논문에서는 강화학습에 대한 새로운 관점을 제시하는데, 이는 reward를 최대화 하는것 보다 desired outcome을 달성하는 action을 추론하는 문제가 될 것이라고 합니다.  
이러한 outcome-directed inference problem을 풀기 위해서 우리가 환경과의 상호작용으로 부터 directly 배울 수 있는 well-shaped한 reward function을 얻기 위해서 novel한 variational inference formulation을 제안했다고 합니다. 그리고 이러한 variational objective 로부터 연구진은 기존의 Bellman backup operator를 연상시키는 새로운 probabilistic Bellman backup operator 를 제안했고 이를 off-policy algorithm을 develop하는 데 써서 goal-directed tasks 들을 풀었다고 합니다. We empirically demonstrate that this method eliminates the need to design reward functions and leads to effective goal-directed behaviors.

- <mark style='background-color: #fff5b1'> tmp </mark>

[outcome_driven1.png](/assets/images/outcome_driven/outcome_driven1.png)
*Fig.*
[outcome_driven2.png](/assets/images/outcome_driven/outcome_driven2.png)
*Fig.*
[outcome_driven3.png](/assets/images/outcome_driven/outcome_driven3.png)
*Fig.*
[outcome_driven4.png](/assets/images/outcome_driven/outcome_driven4.png)
*Fig.*
[outcome_driven5.png](/assets/images/outcome_driven/outcome_driven5.png)
*Fig.*
[outcome_driven6.png](/assets/images/outcome_driven/outcome_driven6.png)
*Fig.*
[outcome_driven7.png](/assets/images/outcome_driven/outcome_driven7.png)
*Fig.*


- <mark style='background-color: #fff5b1'> Refernece </mark>
