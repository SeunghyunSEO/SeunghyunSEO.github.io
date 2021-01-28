---
title: Algorithms - Vanilla Policy Gradient
categories: OpenAI_Spinning_UP
tag: [RL]

toc: true
toc_sticky: true
---

 - <mark style='background-color: #dcffe4'> Key idea </mark>

```
The key idea underlying policy gradients is to push up the probabilities of actions that lead to higher return, 
and push down the probabilities of actions that lead to lower return, until you arrive at the optimal policy.
```

정책 경사 알고리즘 (Policy Gradient Algorithm)의 핵심적인 아이디어는 위와 같습니다

'너가 최종적으로 얻게 될 보상(return)에 긍정적인 영향을 끼치는 action의 확률은 높히고, 그 반대는 낮추게끔 파라메터를 업데이트하자' 입니다.

 - <mark style='background-color: #dcffe4'> Quick Facts </mark>

앞으로는 Vanilla Policy Gradient 를 앞글자만 따서 VPG라고 하도록 하겠습니다. 
(Vanilla는 수수한, 꾸미지 않은, 실제적인라는 뜻을 가지고 있습니다. 아마 PG는 이후 다양한 variation이 존재해서 그런 것 같습니다.)  


(아래는 굳이 번역하지 않도록 하겠습니다.)

> 1.VPG is an on-policy algorithm. <br>
> 2.VPG can be used for environments with either discrete or continuous action spaces. <br>
> 3.The Spinning Up implementation of VPG supports parallelization with MPI. <br>

- <mark style='background-color: #dcffe4'> Key Equations </mark>

자 이제 VPG 핵심적인 equation들에 대해 짧게 알아보도록 하겠습니다.

> <mark style='background-color: #dcffe4'> Notation </mark> <br>
> 1.$$ s $$ : 현재 상황, state ex) 바둑판 현재 상황 <br>
> 2.$$ a $$ : 취할 수 있는 행동, action ex) 에이전트가 취할 수 있는 행동 ex)바둑에서는 전체 바둑판 중에 둘 수 있는 곳 전부, 미로찾기에서는 예를들어 상,하,좌,우 (두 경우 모두 discrete한 경우지만 continuous도 가능?) <br>
> 3.$$ \pi_{\theta} $$ : 정책, policy ex) 바둑에서 다음 수를 바로 이 정책을 따라 둠 (파라메터 $$\theta$$로 정의되어 있음)  <br>
> $$ \pi_{\theta}(a \mid s) $$ : 일반적으로 정책은 현재 상황이 주어졌을 때~ 그 때 둘 수 있는 액션이 확률 분포로 주어짐 ex) 미로찾기 상:0.7, 하:0.2, 좌:0.05, 우:0.05  <br>

$$\pi_{\theta}$$를 파라메터 $$\theta$$로 정의된 정책(policy)이라고 하겠습니다. $$J(\theta)$$를 finite-horizon undiscounted return of the policy라고 하겠습니다.

<center>$$ J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}{
       \sum_{t=0}^{T} \log \pi_{\theta}(a_t|s_t) A^{\pi_{\theta}}(s_t,a_t)
       }$$</center>

이를 파라메터 $$\theta$$에 대해서 미분하면 아래와 같아집니다. 

<center>$$\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}{
       \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) A^{\pi_{\theta}}(s_t,a_t)
       }$$</center>

여기서 $$\tau$$ 는 trajectory 이고 $$A^{\pi_{\theta}}$$ 현재 policy 에 대한 Advantage function 입니다.


PG 알고리즘은 policy performance에 대한 확률적 경사 상승법 (stochastic gradient ascent, SGD) 에 의해 policy 파라메터를 업데이트합니다.

<center>$$ \theta_{k+1} = \theta_k + \alpha \nabla_{\theta} J(\pi_{\theta_k}) $$</center>

Policy gradient implementations typically compute advantage function estimates based on the infinite-horizon discounted return, despite otherwise using the finite-horizon undiscounted policy gradient formula.



