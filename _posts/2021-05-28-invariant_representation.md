---
title: (yet) Learning Invariant Representations for Reinforcement Learning without Reconstruction
categories: Reinforcement_Learning_and_Deep_Reinforcement_Learning
tag: [DeepLearning]

toc: true
toc_sticky: true
---

본 논문은 강화학습을 위해서 Representation을 학습할 때 task와 상관없는 요소들 (task-irrelevant details) 은 빼고 학습을 하자는 내용을 담고 있습니다.
즉 자율주행을 자동차를 학습하는 과정에서 카메라로 부터 들어오는 정보에서 구름이나 멀리 떨어져있는 나무,집 같은 것들은 주행을 하는 데 아무런 영향을 끼치지 않는 요소라는 것입니다. 
이를 위해서 제안한 방법론은 이미지를 받아서 차원을 줄였다가 다시 복원 (reconstruction)시키는 AutoEncoding 같은 방식으로 학습하지 않고 state-space 상에서의 거리들과 latent-space상에서의 bisimulation distance를 일치시키는 방식으로 인코더를 학습시킨다고 합니다. 

## <mark style='background-color: #fff5b1'> Problem Definition and Contribution Points </mark>

![invariant_representation1](/assets/images/invariant_representation/invariant_representation1.png)
*Fig.*

위의 그림이 시사하는 바는, Caption에도 나와있듯, 차의 종류나 구름의 유무 같은 상관없는 것들은 배제하고 task에 중요한 도로정보같은 요소가 같으면 같은 latent space로 매핑 시키는 것이 이 논문의 목적임을 의미합니다.


## <mark style='background-color: #fff5b1'> Related Work </mark>

![invariant_representation2](/assets/images/invariant_representation/invariant_representation2.png)
*Fig.*


## <mark style='background-color: #fff5b1'> Preliminaries </mark>

![invariant_representation3](/assets/images/invariant_representation/invariant_representation3.png)
*Fig.*
![invariant_representation4](/assets/images/invariant_representation/invariant_representation4.png)
*Fig.*
![invariant_representation5](/assets/images/invariant_representation/invariant_representation5.png)
*Fig.*
![invariant_representation6](/assets/images/invariant_representation/invariant_representation6.png)
*Fig.*
![invariant_representation7](/assets/images/invariant_representation/invariant_representation7.png)
*Fig.*
![invariant_representation8](/assets/images/invariant_representation/invariant_representation8.png)
*Fig.*
![invariant_representation9](/assets/images/invariant_representation/invariant_representation9.png)
*Fig.*
![invariant_representation10](/assets/images/invariant_representation/invariant_representation10.png)
*Fig.*
![invariant_representation11](/assets/images/invariant_representation/invariant_representation11.png)
*Fig.*

- <mark style='background-color: #fff5b1'> Refernece </mark>
