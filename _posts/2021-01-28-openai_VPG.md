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

(아래는 굳이 번역하지 않도록 하겠습니다.)


앞으로는 Vanilla Policy Gradient 를 앞글자만 따서 VPG라고 하도록 하겠습니다. 
(Vanilla는 수수한, 꾸미지 않은, 실제적인라는 뜻을 가지고 있습니다. 아마 PG는 이후 다양한 variation이 존재해서 그런 것 같습니다.)  

> 1.VPG is an on-policy algorithm. <br>
> 2.VPG can be used for environments with either discrete or continuous action spaces. <br>
> 3.The Spinning Up implementation of VPG supports parallelization with MPI. <br>

 - <mark style='background-color: #dcffe4'> Key Equations </mark>
 
 자 이제 VPG 핵심적인 equation들에 대해 짧게 알아보도록 하겠습니다.
 
 <center>$$\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}{
        \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) A^{\pi_{\theta}}(s_t,a_t)
        }$$</center>
    
    
 <center>$$ \theta_{k+1} = \theta_k + \alpha \nabla_{\theta} J(\pi_{\theta_k}) $$</center>
        
