---
title: (미완) Reinforcement Learning(RL) Overview
categories: Reinforcement_Learning_and_Deep_Reinforcement_Learning
tag: [RL]

toc: true
toc_sticky: true
---

$$\rightarrow$$ [[Post written by lilian Weng](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#common-approaches)]


위의 포스트는 OpenAI의 머신 러닝 연구자인 [lilian Weng](https://scholar.google.com/citations?user=dCa-pW8AAAAJ)이 작성한 강화학습에 대한 전반적인 개념에 대한 글이며
본 포스트는 위의 글을 상당 부분 참고하여 만들어 졌습니다..


![image](https://user-images.githubusercontent.com/48202736/106112753-fa73a580-6190-11eb-961b-d598ee0902a3.png)
{: style="width: 40%;" class="center"}
*Fig. OpenAI의 Lilian Weng*


릴리안의 포스트를 번역하는것은 아래와 같이 그녀에게 이메일로 알려주면 된다고 하는데, 이 글은 아직 비공식 글이기 때문에 우선 번역을 ...해보도록 하겠습니다. (아무도 안볼듯 ㅎ)

```
Q: Can I translate your posts to another language?

A: Yes and my pleasure! But please email me in advance and please keep the original post link on top (rather than in tiny font at the end, and yes I do see that case and that makes me feel sad).
```

하지만 이 포스트는 제가 릴리안의 글을 똑같은 포맷으로 그대로 가져와 번역만 한 것이 아닙니다.<br>
읽으면서 더 자료가 필요하다 싶은 부분은 자료를 추가하고, 아닌 부분은 줄이는 등 수정을 할 것이기 때문에 뭔가 읽다가 불편하시다면 릴리안의 오리지널 포스트를 읽으시길 추천드립니다.


그러면 시작하도록 하겠습니다 :)

## <mark style='background-color: #fff5b1'> What is Reinforcement Learning? </mark>

자, 어떤 알려지지 않은 환경에 ```agent```(로봇이라고 생각하면 될듯)가 놓여져 있다고 생각합시다, 그리고 이 agent는 환경과 상호작용을 통해서 어떠한 ```rewards```(보상)을 얻을 수 있다고 합시다. <br>

![image](https://user-images.githubusercontent.com/48202736/106164050-ecdc1100-61cc-11eb-86c5-cffd1d3166fb.png)
*(이미지 출처 : [link](https://studywolf.wordpress.com/2015/03/29/reinforcement-learning-part-3-egocentric-learning/))*

(위의 예시를 보시면 로봇 쥐가 agent이고, 현재 미로와 interaction을 한다는 것은 쥐가 움직이면(action) 상태(state)가 가 변하게 되는 것 등을 말하며, 
얻게 될 reward에 대해 얘기해 보자면, 치즈를 먹으면 +10점, 벽에 부딪히면 -1점, 그 외에는 0점 등으로 정의를 할 수 있을 것 같습니다.) <br>

Agent는 결과적으로 누적 보상을 최대화 하는방향으로 액션을 취해야 할 것입니다. <br>
실제, real world에서 위에서 얘기한 것들은 가령 게임에서 고득점을 얻거나 상대방을 이기는 것 등을 목적으로 하는 로봇들을 예시로 들 수 있을겁니다.      


![image](https://user-images.githubusercontent.com/48202736/106113304-afa65d80-6191-11eb-8486-3b9f8fba7cb7.png)
{: style="width: 70%;" class="center"}
*Fig. 1. 그림이 전달하는 바는 명확합니다, 우리가 현재 상황에서 어떤 액션을 취하면 환경에 그만한 변화가 생긱고, 그렇기 때문에 생기는 reward를 agent가 받게 되는거죠.*

강화학습 (Reinforcement Learning, RL)의 목적은  시행 착오(trial and error, 이는 모든 머신러닝의 기본적인 매커니즘이죠)를 통해서 '이런 상황에서 이런 행동을 했더니 이렇던데?' 같은 feedback 통해 가장 좋은 전략(strategy)을 학습하는 겁니다. <br>
Optimal한 전략을 가지고 있으면, agent는 future rewards를 최대화 하는 방향으로 환경에 적응할 수 있을 겁니다. <br>

## <mark style='background-color: #fff5b1'> Machine Learning </mark>

머신러닝과 강화학습은 무슨 관련이 있을까?

<img width="800" alt="스크린샷 2021-01-29 오후 3 09 47" src="https://user-images.githubusercontent.com/48202736/106238352-12583180-6244-11eb-915a-3da566ce7411.png">
{: style="width: 70%"}
*(이미지 출처 : David Silver의 강의)*

강화학습은 머신러닝의 한 갈래입니다. 머신러닝의 매커니즘은 어떻게보면 굉장히 단순하다고 할 수 있는데요, 우리가 어떤 이미지분류나 음성인식 같은 문제를 풀고 싶다고 생각해 봅시다.
우리가 원하는것은 데이터를 넣어서 그에 맞는 답을 출력해주는 (음성인식이면 텍스트를, 이미지 분류면 그 이미지가 뭔지에 대한 결과를) 어떤 완벽한 함수(oracle function)를 구하는겁니다. <br>

하지만 우리는 그런건 당연히 단박에, 제대로 구할 수 없습니다. <br>

대신에 우리는 그 함수를 수많은 파라메터 $$\theta$$로 모델링 한 다음에 수많은 데이터를 통해서 파라메터를 업데이트 해서, 우리가 가지고 있는 함수를 oracle function에 최대한 비슷하게 만드는 겁니다.
결국 우리의 목적은 최대한 oracle function과 근사한 approximate function을 구하는거죠. <br>

어떻게 이 함수를 구하느냐 ? 어떻게 학습을 할거냐? 어떻게 데이터를 입력으로 주고 출력으로 줘서 모델링 할거냐? 에 따라서 지도 학습 (Supervised Learning), 비지도 학습(Unsupervised Learning), 강화 학습(Reinforcement Learning) 으로 나눌 수 있는데 우리가 지금 배우고자 하는게 강화학습이 되는겁니다. <br>

> 1. Supervised Learning : $$y=f(x)$$ <br>
> 2. Unsupervised Learning : $$x~p(x)$$ or $$x=f(x)$$ <br>
> 3. Reinforcement Learning : Finda a policy, $$p(a \vert s)$$ which maximizes the expected sum of future rewards.

일반적으로 음성인식, 이미지 분류, 기계번역 같은 문제는 위에서 말한 것 처럼 지도 학습으로 문제를 풀고 있습니다. 가령 음성을 x라고 해서 f에 넣으면 그에 해당하는 텍스트 y를 뽑아주는거죠.
이런 f를 학습하기 위해서 지도학습에서는 '자 이 음성(x)일때 정답은 이 텍스트(y)야' 하고 $$X,Y$$를 pair로 정답으로 주어줍니다. <br>

하지만 강화학습은 그렇게 학습되지 않죠. 예를들어 바둑을 두는 agent를 학습 할 때 '자 미로에서 이런 칸에서는 무조건 앞으로가야돼' 라고 정답을 알려주는게 아니고 '미로를 탈출해서 보상을 받아라, 시간이 오래걸리면 패널티를 받는다' 같은 식으로 최종 보상을 가장 크게 받는 방식으로 알아서 '자 미로에서 이런 칸에서는 이렇게 가는 게 낫다' 라는 정책을 학습하게 설계해서 학습합니다. <br>

이 점을 잘 기억해두시길 바랍니다.

```
물론 여러개를 섞어서 학습하는 방식을 모델링 할 수도 있습니다. 
예를들어 우리가 잘 알고있는 AlphaGo는 매 순간(바둑판의 상태) 어떤식으로 바둑을 둬야할지에 대한 정답을 알려줍니다(다른 프로기사들은 이럴 경우 이렇게 두더라).
그런식으로 Supervised Learning을 통해 네트워크를 충분히 학습한 네트워크를 또 강화학습을 통해 학습하고 ... 

자세한 건 알파고 논문을 나중에 따로 리뷰하면서 설명해보도록 하겠습니다.
```

### <mark style='background-color:black; color:white'>(2/11)</mark> <mark style='background-color: #dcffe4'> In the beginning was Andrey Markov </mark>

![image](https://user-images.githubusercontent.com/48202736/106242109-8564a680-624a-11eb-9718-19f82f009492.png)
*Fig. Andrey Markov 선생님*

#### <mark style='background-color: #ffdce0'> Markov Chains </mark>

#### <mark style='background-color: #ffdce0'> Markov Decision Processes </mark>

강화학습을 공부할 때 다른 어떤것들보다 중요한 개념이 있는데 바로 ```Markov Decision Processes``` (MDPs) 입니다. 대부분의 RL문제들은 MDP를 따릅니다.
MDP의 모든 상태들은 "Markov" 특성을 갖는데, 이는 현재 상태가 t일때 미래에 영향을 주는 것은 오직 t의 상태 뿐이라는 겁니다. (t-1, t-2, .... 1은 영향 안줌) <br>
식으로 나타내면 아래와 같이 쓸 수 있습니다:

$$
\mathbb{P}[ S_{t+1} \vert S_t ] = \mathbb{P} [S_{t+1} \vert S_1, \dots, S_t]
$$

다시 말하면, 과거의 상태가 어떻든간에 미래를 결정하는데 영향을 주는건 현재뿐이라는 겁니다.


![image](https://user-images.githubusercontent.com/48202736/106113925-57bc2680-6192-11eb-9427-f78d9dc6407f.png)
{: style="width: 100%;" class="center"}
*Fig. 3. The agent-environment interaction in a Markov decision process. (Image source: Sec. 3.1 Sutton & Barto (2017).)*


MDP는 5가지 요소 $$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$$ 로 구성되어있습니다:

- $$\mathcal{S}$$ - a set of states;
- $$\mathcal{A}$$ - a set of actions;
- $$P$$ - transition probability function;
- $$R$$ - reward function;
- $$\gamma$$ - discounting factor for future rewards.
In an unknown environment, we do not have perfect knowledge about $$P$$ and $$R$$.


![image](https://user-images.githubusercontent.com/48202736/106113937-5b4fad80-6192-11eb-9a7c-c6d905ff284a.png)
{: class="center"}
*Fig. 4. A fun example of Markov decision process: a typical work day. (Image source: [randomant.net/reinforcement-learning-concepts](https://randomant.net/reinforcement-learning-concepts/))*




### <mark style='background-color:black; color:white'>(1/11)</mark> <mark style='background-color: #dcffe4'> Key Concepts </mark>

자 이제 RL에서 쓰이는 중요한 concept, notation에 대해 알아보도록 합시다 :)


우선 agent가 행동하는(acting)곳은 바로 ```environment```(환경)입니다. <br>
agent가 한 특정 행동에 대해서 어떻게 환경이 반응 하는지는 ```model``` 로부터 정의되는데요, 이 모델은 우리가 알 수도 있고 모를 수도 있습니다. <br>
agent는 환경의 수많은 ```states```(상태) ($$s \in \mathcal{S}$$)들 중 하나에 있을겁니다. (격자 구조 미로라면 미로의 한 칸, 바둑을 두는 상황이라면 바둑이 놓여진 어느 한 상황) <br>
그리고 그 주어진 상태에서 취할 수 있는 수많은 ```actions```(행동) ($$a \in \mathcal{A}$$) 중 하나를 취할겁니다. 그러면 당연히 상태가 변하겠죠? (미로에서 움직였으면 상태가 변하는거고, 현재 상황의 바둑판에 돌을 두면 그 다음 상태가 되는겁니다.)<br>
근데 agent가 어느 state에 도착할지는 ```transition probabilities between states``` ($$P$$)에 의해서 결정됩니다. <br>

예를들어 2x2 Grid World 에는 총 4개의 state가 존재할 것이고 각 각의 상태로 transition할 확률이 아래처럼 존재할 수 있다는 겁니다.

||$$s_{11}$$|$$s_{12}$$|$$s_{21}$$|$$s_{22}$$|
|--|--|--|--|--|
|$$s_{11}$$|0.1|0.2|0.4|0.3|
|$$s_{12}$$|0.25|0.3|0.1|0.35|
|$$s_{21}$$|0.3|0.2|0.1|0.4|
|$$s_{22}$$|0.35|0.1|0.25|0.3|

그리고 일단 액션을 취하면 envirionment가 '니가 취한 행동의 대가야'라며 ```reward``` ($$r \in \mathcal{R}$$)를 피드백으로 줍니다.<br>

#### <mark style='background-color: #ffdce0'> Model-based vs Model-free </mark>

![reinforcement_learning_model_free_monte_carlo_three_episodes_fast](https://user-images.githubusercontent.com/48202736/106167816-f7000e80-61d0-11eb-86ef-49c4a802326b.gif)
(Model-free 알고리즘의 예시, 출처 : [link](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html))

model이라는 것은 reward function과 transition probabilities를 정의하게 됩니다. 하지만 이전에 언급한 것 처럼 우리는 모델이 어떻게 작용을 하는지 알 수도 있고 모를 수도 있습니다. 
그래서 우리는 모델을 아는 것, 모르는 것두 가지 상황에 대해 생각해 봐야 합니다.

- Know the model: (환경, 어떻게 상호작용하는지 등)모든 정보를 다 알고 미리 앞서 생각해 planning할 수 있다는 것입니다; **<span style="color:#e01f1f">model-based RL</span>** 라고도 합니다. 우리가 환경을 정확히 알고 있을 때에는 (어떻게 작용하는지 명확히 암) 우리는 [Dynamic Programming](https://en.wikipedia.org/wiki/Dynamic_programming) (DP)을 통해서 최적 해(optimal solution)을 구해낼 수 있습니다. 

- Does not know the model: 부족한 정보를 가지고 학습하는겁니다; **<span style="color:#e01f1f">do model-free RL</span>** 한다고 합니다. 혹은 try to learn the model explicitly as part of the algorithm. 앞으로 우리가 얘기할 것들은 대부분 Model-free 상황일겁니다. <br>

아래의 그림은 현재 잘 알려진 (Deep) Reinforcement Learning 알고리즘들을 Model-based, Model-free에 따라 세분화 해서 나눈 그림입니다. (아직은 잘 모르겠네요 ㅎ)

![image](https://user-images.githubusercontent.com/48202736/106139389-b5f80200-61b0-11eb-80c2-bedcd6b4dd8f.png)
(이미지 출처 : [OpenAI Spinning UP](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html))

#### <mark style='background-color: #ffdce0'> Policy and Value functions in RL </mark>

이번에는 RL에서 정말 중요한 핵심 개념인 Policy와 Value function들에 대해서 알아보도록 하겠습니다.<br><br>

```policy```(정책) $$\pi(s)$$는 agent가 어떤 상황에 처했을 때 어떤 행동을 하는게 최적인지를 **<span style="color:#e01f1f">'Total Rewards'(겜 끝날때 까지  총 보상)를 최대화 한다는 목적으로</span>** 알려주는(확률분포로 알려줍니다) 함수입니다.<br>
각각의 state는 가치 함수라고 하는 ```value function``` $$V(s)$$ 와 깊은 연관이 있는데, 이 가치 함수는 '현재 상태' 에서 '우리가 현재 해당 policy를 따라 행동을 함으로써 미래에 얻을 수 있는 ```future rewards```의 기대값을 예측하죠. 다시 말하면, 가치함수는 **<span style="color:#e01f1f">'현재 상태가 얼마나 좋은가?'</span>** 를 정량화 해주는 겁니다.

![image](https://user-images.githubusercontent.com/48202736/106158313-f2cef380-61c6-11eb-958c-dd9aad64a88d.png)

위의 그림의 예시를 보면 (아직 우리가 배우지 않은) Optimal Value Function을 구했을때의 Value Function을 볼 수 있는데요, 말 그대로 현재 상태가 얼마나 좋은가? 가 Grid World의 발판에 숫자로 쫙 쓰여져 있습니다(b). 그리고 이때의 policy가 곧 Optimal Policy 라고 할 수 있습니다(c).

정말 다 떼놓고 간단하게 말하면, 강화학습에서 우리가 학습하고자 하는 것은 이 두가지 ```policy function```, ```value function``` 라고 할 수 있습니다. <br>
이 둘을 어떻게 학습하느냐에 따라서 Value-Based 냐 Policy-Based냐가 나눠지는 것이죠.<br>

![image](https://user-images.githubusercontent.com/48202736/106113912-52f77280-6192-11eb-879d-973a2364c249.png)
{: style="width: 100%;" class="center"}
*Fig. 2. Summary of approaches in RL based on whether we want to model the value, policy, or the environment. (Image source: reproduced from David Silver's RL course [lecture 1](https://youtu.be/2pWv7GOvuf0).)*


에이전트와 환경 사이의 상호작용은 일련의 action과 그 때의 obeserved rewards를 내포하고 있습니다 ($$t=1, 2, \dots, T$$). <br>
에이전트가 스토리를 진행시키면서 환경에 대한 정보를 축적하고, optimal policy를 학습하면서? best policy를 효율적으로 학습하기위해서 다음 액션으로 무엇을 취할지를 결정합니다.<br><br>

자 state, action 그리고 매 시간(time step) $$t$$ 마다 받는 reward를 각각 $$S_t$$, $$A_t$$, and $$R_t$$ 라고 합시다.

> state, 상태 : $$S_t$$ <br> 
> action, 행동 : $$A_t$$ <br>
> reward, 보상 : $$R_t$$ <br>

마지막으로 에이전트가 '$$S_1$$ 에서 $$A_1$$ 행동을 했더니 $$R_2$$ 를 받았고, $$S_2$$로 이동이 됐고, 또 $$A_2$$를 하고....' 이런 게임이 끝날 때 까지 일련의 상호작용을  ```episode``` (또는 "trial" 이나 "trajectory" 라고도 합니다.) 라고 하며, 게임이 끝날 때의 terminal state를 대문자 T를 써서 $$S_T$$ 라고 합니다:

$$
S_1, A_1, R_2, S_2, A_2, \dots, S_T
$$


아래의 용어들은 앞으로 보게될 수많은 RL 알고리즘들에서 마주치게 될테니 익숙해지시길 바랍니다 : <br>
(아래는 제가 강화학습에 대한 지식이 얕아서 적당히 의역하기 힘들어 원문 그대로 두도록 하겠습니다)

```
- Model-based: Rely on the model of the environment; either the model is known or the algorithm learns it explicitly.

- Model-free: No dependency on the model during learning.

- On-policy: Use the deterministic outcomes or samples from the target policy to train the algorithm.

- Off-policy: Training on a distribution of transitions or episodes produced by a different behavior policy rather than that produced by the target policy.
```

#### <mark style='background-color: #ffdce0'> Model: Transition and Reward </mark>

강화학습에서 model은 환경을 설명하는 a descriptor(서술어?) 라고 할 수 있습니다. 모델을 가지고 있다는 것은, 우리가 환경과 어떻게 상호작용을 할지, 그리고 어떤 상태에서 어떤 행동을 했을때 에이전트에게 어떤 피드백을 결과로 줄지를 학습하거나, 추론할 수 있다는 것을 의미합니다. <br>
model은 두가지 주된 요소를 가지고 있습니다, 바로 transition probability function $$P$$ 그리고 reward function $$R$$ 입니다. <br>

자 이제 우리가, 어떤 state s에 있다고 생각해 봅시다.
우리는 그 다음 state s'에 도달하기 위해서 action을 취하고 그 결과로 reward r 을 얻을 겁니다.
이것이 바로 **transition** step 이고, (s, a, s', r) 라는 튜플로 표현이 됩니다. <br>

Transition function P는 reward r을 얻으면서 액션을 취한 뒤 내가 s 에서 s'으로 이동할 확률을 기록합니다. <br>

우리는 $$\mathbb{P}$$를 사용해 확률을 나타냅니다. <br>

$$
P(s', r \vert s, a)  = \mathbb{P} [S_{t+1} = s', R_{t+1} = r \vert S_t = s, A_t = a]
$$

$$\ast$$ 위의식은 말로하면 이렇습니다. 현재 상태 s에서 a라는 액션을 취했을 때 보상 r을 받고 s'로 이동할 확률을 나타내는 식입니다.
즉 확률이니까 s에서 a라는 액션을 취해도 s'로 못갈수도 있습니다.

그러므로 state-transition function (상태 천이 행렬)은 $$P(s', r \vert s, a)$$의 함수로 정의될 수 있습니다:

$$
P_{ss'}^a = P(s' \vert s, a)  = \mathbb{P} [S_{t+1} = s' \vert S_t = s, A_t = a] = \sum_{r \in \mathcal{R}} P(s', r \vert s, a)
$$

Reward function R은 액션 하나가 가져올 그 다음 reward를 예측하는 함수입니다:

$$
R(s, a) = \mathbb{E} [R_{t+1} \vert S_t = s, A_t = a] = \sum_{r\in\mathcal{R}} r \sum_{s' \in \mathcal{S}} P(s', r \vert s, a)
$$


#### <mark style='background-color: #ffdce0'> Policy </mark>


정책이라고 하는 Policy는, agent가 state s 에서 취할 수 있는 액션에 대한 함수 $$\pi$$ 입니다. 
이건 deterministic할 수도 있고, stochastic할 수도 있습니다:
- Deterministic: $$\pi(s) = a$$.
- Stochastic: $$\pi(a \vert s) = \mathbb{P}_\pi [A=a \vert S=s]$$.

(deterministic 하다는 것은 하나의 점을 뽑는다고 볼 수 있고, 
stochastic하다는 것은 예를들어 $$\pi(a \vert s) = [상=0.12,하=0.48,좌=0.3,우=0.2]$$ 처럼 결과를 뱉는, 즉 분포를 뽑는다고 할 수 있을 것 같습니다.)


#### <mark style='background-color: #ffdce0'> Value Function </mark>

Value function은 현재 상태가 얼마나 좋냐를 정량적으로 나타내는 함수입니다. (인풋 : state , 출력 : how good?=0.7) (위에서도 언급했지만 다르게 말하면 이는 '현재 상태' 에서 '우리가 현재 해당 policy를 따라 행동을 함으로써 미래에 얻을 수 있는 ```future rewards```의 기대값을 나타낸다고도 할 수 있습니다.)

future reward 라는 것은 아~주 미래에 얻게 될 보상으로 ```return```이라고도 합니다. 앞으로 얻게 될 보상들의 합이기도 하죠. <br>
$$G_t$$를 시작지점 t 부터 해서 계산해볼까요?:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

Discounting factor $$\gamma \in [0, 1]$$ 는 미래에대한 보상의 영향력을 감소시킵니다, 왜냐하면:
```
- The future rewards may have higher uncertainty; i.e. stock market.
- The future rewards do not provide immediate benefits; i.e. As human beings, we might prefer to have fun today rather than 5 years later ;).
- Discounting provides mathematical convenience; i.e., we don't need to track future steps forever to compute return.
- We don't need to worry about the infinite loops in the state transition graph.
```

만약에 여기서 $$\gamma$$ 가 0이라면 

$$
G_t = R_{t+1} + 0 R_{t+2} + \dots = \sum_{k=0}^{\infty} 0^k R_{t+k+1} = R_{t+1}
$$

이 되기 때문에 현재의 쾌락만을 추구하면서(...) 학습이 되기 때문에 별로 좋은 결과를 가져오지 못할 가능성이 큽니다.<br><br>

```
왜 누적 보상을 최대화 해야 할까요? 그 이유는 현재 보상만 최대화 하는 방향으로 행동하는게 미래에는 안좋을 수 있기 때문입니다. 

예를 들어 우리가 지금 강화학습을 공부하는게 고통스러울 지언정, 미래엔 도움이 될 것이고 (그렇겠죠...?), 지금 랩탑을 덮고 피시방에서 롤 하러 가는건 당장은 해피할 지 몰라도 미래에는...)
```


이게 와닿지 않으실까봐 예제를 가져왔습니다.

위에서도 봤던 예시인데요

![reinforcement_learning_model_free_monte_carlo_three_episodes_fast](https://user-images.githubusercontent.com/48202736/106167816-f7000e80-61d0-11eb-86ef-49c4a802326b.gif)
(Model-free 알고리즘의 예시, 출처 : [link](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html))

이 예시에서 각각의 episode에서 로봇이 $$S_1, A_1, R_2, ... $$ 이런식으로 에피소드가 끝날 때 까지 한 상태,행동,보상을 따라가보도록 하겠습니다

![reinforcement_learning_model_free_monte_carlo_three_episodes_linear](https://user-images.githubusercontent.com/48202736/106168298-74c41a00-61d1-11eb-8124-195252f53fa5.png)

우리는 discount factor가 0.9라고 할 때 최종 return값을 아래와 같이 계산할 수 있습니다.

![reinforcement_learning_model_free_return_first_episode](https://user-images.githubusercontent.com/48202736/106168305-768ddd80-61d1-11eb-9ece-10cd225b6809.png)


다시 Value function으로 돌아가보도록 하겠습니다. 

state s 에서의 ```state-value``` 는 expected return 입니다. (기대값이 엄청 많죠?, 강화학습은 그럴 수 밖에 없는게, 어느 상태에서 취할 수 있는 행동이 많고 또 거기서 다음 상태로 가면 또 action이 많고 하는데 이걸 다 계산해 내야 하기 때문에 어쩔 수 없습니다.) 

t시간에 s 상태에 있을 때의 가치 함수를 정의해보면 다음과 같습니다.

$$
V_{\pi}(s) = \mathbb{E}_{\pi}[G_t \vert S_t = s]
$$

이는 현재 상태 s에서 정책 \pi를 따랐을때 기대 보상(expected return)을 정량적으로 나타낸겁니다. 
(걍 현재 놓여진 상태에서 [상,하,좌,우]로 갈 수 있는 선택지가 있을 때, 그것을 확률적으로 정의한 policy가 있을 것이고, 그 기대값을 통해 기대 보상값을 예측하는 것 같습니다. ) br><br>

유사하게 우리는 ```action-value``` ("Q-value"; Q as "Quality" I believe?) 를 다음과 같이 정의할 수 있습니다:

$$
Q_{\pi}(s, a) = \mathbb{E}_{\pi}[G_t \vert S_t = s, A_t = a]
$$

```action-value```는 예를들어 '현재 상태에서 [상,하,좌,우] 중 오른쪽으로 가는 행동을 선택했을 때, 얻게될 기대 보상값 입니다.<br>

![image](https://user-images.githubusercontent.com/48202736/106173311-20bc3400-61d7-11eb-88c2-150ea4231df6.png)

![image](https://user-images.githubusercontent.com/48202736/106173058-dd61c580-61d6-11eb-8e9e-9b6d10ed3dc6.png)

```state-value``` 함수는 아래와 같이 다시 쓸 수 있습니다. (```action-value```와 ```policy```를 합친거죠)

$$
V_{\pi}(s) = \sum_{a \in \mathcal{A}} Q_{\pi}(s, a) \pi(a \vert s)
$$

행동 가치 함수(action-value)와 상태 가치 함수(state-value)의 차이는 곧 action ```advantage``` function ("A-value")로 정의할 수 있습니다:

$$
A_{\pi}(s, a) = Q_{\pi}(s, a) - V_{\pi}(s)
$$


#### <mark style='background-color: #ffdce0'> Optimal Value and Policy </mark>

![image](https://user-images.githubusercontent.com/48202736/106158313-f2cef380-61c6-11eb-958c-dd9aad64a88d.png)

(Optimal solutions to the gridworld example, 출처 : [link](http://incompleteideas.net/book/first/ebook/node35.html))

위의 그림의 예시를 먼저 봐보도록 합시다.

> States : 5x5 = 총 25개의 가능한 states 
> Action : [상,하,좌,우] 4개의 action 
> Reward : A로 가면 +10 받고 A'로 워프, B로 가면 +5 받고 B'로 워프, 벽에 닿으면 -1
> Discounted Factor : 0.9

위의 환경에서 미래에 얻을 expected return을 최대화 하는 방식으로 학습이 됐을 때 $$V^{\ast}$$ 와 $$\pi^{\ast}$$.

b)를 보면 각 state마다, 현재 상태가 얼마나 좋은가가 정량적으로 나타나 있습니다. 
그렇기 때문에 예를 들어 agent가 맨 왼쪽 위 [22.0]이라는 칸에 있으면 [우,하] 두개의 선택지 중 가장 좋은 상태인 [22.4]점의 상태로 가기 위해서 오른쪽이라는 액션을 택하게 됩니다. 


Optimal value function은 최대 return을 뽑아냅니다:

$$
V_{*}(s) = \max_{\pi} V_{\pi}(s),
Q_{*}(s, a) = \max_{\pi} Q_{\pi}(s, a)
$$

가치 함수들을 최대화 하는 정책은 곧 optimal policy입니다:

$$
\pi_{*} = \arg\max_{\pi} V_{\pi}(s),
\pi_{*} = \arg\max_{\pi} Q_{\pi}(s, a)
$$

위에서 말한대로 optimal policy를 가지고 있다는 것은 곧 가치함수를 최대화 하는 것이기 때문에

$$V_{\pi_{*}}(s)=V_{*}(s)$$ and $$Q_{\pi_{*}}(s, a) = Q_{*}(s, a)$$

입니다.



### <mark style='background-color:black; color:white'>(3/11)</mark> <mark style='background-color: #dcffe4'> Bellman Equations </mark>

벨만 방정식 (Bellman equations)는 가치함수를 즉시 얻는 보상(immediate reward, $$R_{t+1}$$) + discount된 미래의 가치로 나누는 방정식들을 의미합니다.

$$
\begin{aligned}
V(s) &= \mathbb{E}[G_t \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \dots) \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma G_{t+1} \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}) \vert S_t = s]
\end{aligned}
$$

Q-value에 대해서도 마찬가지로 decompose할 수 있습니다.

$$
\begin{aligned}
Q(s, a) 
&= \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}) \mid S_t = s, A_t = a] \\
&= \mathbb{E} [R_{t+1} + \gamma \mathbb{E}_{a\sim\pi} Q(S_{t+1}, a) \mid S_t = s, A_t = a]
\end{aligned}
$$


#### <mark style='background-color: #ffdce0'> Bellman Expectation Equations </mark>

벨만 기대 방정식은 뭘까요? <br>

The recursive update process can be further decomposed to be equations built on both state-value and action-value functions. As we go further in future action steps, we extend V and Q alternatively by following the policy $$\pi$$.


![image](https://user-images.githubusercontent.com/48202736/106113958-60146180-6192-11eb-9631-b0a211e794e0.png)
{: style="width: 60%;" class="center"}
*Fig. 5. Illustration of how Bellman expection equations update state-value and action-value functions.*


$$
\begin{aligned}
V_{\pi}(s) &= \sum_{a \in \mathcal{A}} \pi(a \vert s) Q_{\pi}(s, a) \\
Q_{\pi}(s, a) &= R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a V_{\pi} (s') \\
V_{\pi}(s) &= \sum_{a \in \mathcal{A}} \pi(a \vert s) \big( R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a V_{\pi} (s') \big) \\
Q_{\pi}(s, a) &= R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a \sum_{a' \in \mathcal{A}} \pi(a' \vert s') Q_{\pi} (s', a')
\end{aligned}
$$


#### <mark style='background-color: #ffdce0'> Bellman Optimality Equations </mark>

If we are only interested in the optimal values, rather than computing the expectation following a policy, we could jump right into the maximum returns during the alternative updates without using a policy. RECAP: the optimal values $$V_*$$ and $$Q_*$$ are the best returns we can obtain, defined [here](#optimal-value-and-policy).

$$
\begin{aligned}
V_*(s) &= \max_{a \in \mathcal{A}} Q_*(s,a)\\
Q_*(s, a) &= R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a V_*(s') \\
V_*(s) &= \max_{a \in \mathcal{A}} \big( R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a V_*(s') \big) \\
Q_*(s, a) &= R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a \max_{a' \in \mathcal{A}} Q_*(s', a')
\end{aligned}
$$

Unsurprisingly they look very similar to Bellman expectation equations.

If we have complete information of the environment, this turns into a planning problem, solvable by DP. Unfortunately, in most scenarios, we do not know $$P_{ss'}^a$$ or $$R(s, a)$$, so we cannot solve MDPs by directly applying Bellmen equations, but it lays the theoretical foundation for many RL algorithms.



## <mark style='background-color: #fff5b1'> Common Approaches </mark>

Now it is the time to go through the major approaches and classic algorithms for solving RL problems. In future posts, I plan to dive into each approach further.


### <mark style='background-color:black; color:white'>(4/11)</mark> <mark style='background-color: #dcffe4'> Dynamic Programming </mark>

When the model is fully known, following Bellman equations, we can use [Dynamic Programming](https://en.wikipedia.org/wiki/Dynamic_programming) (DP) to iteratively evaluate value functions and improve policy.


#### <mark style='background-color: #ffdce0'> Policy Evaluation </mark>

Policy Evaluation is to compute the state-value $$V_\pi$$ for a given policy $$\pi$$:

$$
V_{t+1}(s) 
= \mathbb{E}_\pi [r + \gamma V_t(s') | S_t = s]
= \sum_a \pi(a \vert s) \sum_{s', r} P(s', r \vert s, a) (r + \gamma V_t(s'))
$$

#### <mark style='background-color: #ffdce0'> Policy Improvement </mark>

Based on the value functions, Policy Improvement generates a better policy $$\pi' \geq \pi$$ by acting greedily.

$$
Q_\pi(s, a) 
= \mathbb{E} [R_{t+1} + \gamma V_\pi(S_{t+1}) \vert S_t=s, A_t=a]
= \sum_{s', r} P(s', r \vert s, a) (r + \gamma V_\pi(s'))
$$

#### <mark style='background-color: #ffdce0'> Policy Iteration </mark>

The *Generalized Policy Iteration (GPI)* algorithm refers to an iterative procedure to improve the policy when combining policy evaluation and improvement.

$$
\pi_0 \xrightarrow[]{\text{evaluation}} V_{\pi_0} \xrightarrow[]{\text{improve}}
\pi_1 \xrightarrow[]{\text{evaluation}} V_{\pi_1} \xrightarrow[]{\text{improve}}
\pi_2 \xrightarrow[]{\text{evaluation}} \dots \xrightarrow[]{\text{improve}}
\pi_* \xrightarrow[]{\text{evaluation}} V_*
$$

In GPI, the value function is approximated repeatedly to be closer to the true value of the current policy and in the meantime, the policy is improved repeatedly to approach optimality. This policy iteration process works and always converges to the optimality, but why this is the case?

Say, we have a policy $$\pi$$ and then generate an improved version $$\pi'$$ by greedily taking actions, $$\pi'(s) = \arg\max_{a \in \mathcal{A}} Q_\pi(s, a)$$. The value of this improved $$\pi'$$ is guaranteed to be better because:

$$
\begin{aligned}
Q_\pi(s, \pi'(s))
&= Q_\pi(s, \arg\max_{a \in \mathcal{A}} Q_\pi(s, a)) \\
&= \max_{a \in \mathcal{A}} Q_\pi(s, a) \geq Q_\pi(s, \pi(s)) = V_\pi(s)
\end{aligned}
$$


### <mark style='background-color:black; color:white'>(5/11)</mark> <mark style='background-color: #dcffe4'> Monte-Carlo Methods </mark>

First, let's recall that $$V(s) = \mathbb{E}[ G_t \vert S_t=s]$$. Monte-Carlo (MC) methods uses a simple idea: It learns from episodes of raw experience without modeling the environmental dynamics and computes the observed mean return as an approximation of the expected return. To compute the empirical return $$G_t$$, MC methods need to learn from <span style="color: #e01f1f;">**complete**</span> episodes $$S_1, A_1, R_2, \dots, S_T$$ to compute $$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$ and all the episodes must eventually terminate.

The empirical mean return for state s is:

$$
V(s) = \frac{\sum_{t=1}^T \mathbb{1}[S_t = s] G_t}{\sum_{t=1}^T \mathbb{1}[S_t = s]}
$$

where $$\mathbb{1}[S_t = s]$$ is a binary indicator function. We may count the visit of state s every time so that there could exist multiple visits of one state in one episode ("every-visit"), or only count it the first time we encounter a state in one episode ("first-visit"). This way of approximation can be easily extended to action-value functions by counting (s, a) pair.

$$
Q(s, a) = \frac{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a] G_t}{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a]}
$$

To learn the optimal policy by MC, we iterate it by following a similar idea to [GPI](#policy-iteration).

![image](https://user-images.githubusercontent.com/48202736/106113986-66a2d900-6192-11eb-9398-bc2c1fdd0020.png)
{: style="width: 50%;" class="center"}

1. Improve the policy greedily with respect to the current value function: $$\pi(s) = \arg\max_{a \in \mathcal{A}} Q(s, a)$$.
2. Generate a new episode with the new policy $$\pi$$ (i.e. using algorithms like ε-greedy helps us balance between exploitation and exploration.)
3. Estimate Q using the new episode: $$q_\pi(s, a) = \frac{\sum_{t=1}^T \big( \mathbb{1}[S_t = s, A_t = a] \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} \big)}{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a]}$$


### <mark style='background-color:black; color:white'>(6/11)</mark> <mark style='background-color: #dcffe4'> Temporal-Difference Learning </mark>

Similar to Monte-Carlo methods, Temporal-Difference (TD) Learning is model-free and learns from episodes of experience. However, TD learning can learn from <span style="color: #e01f1f;">**incomplete**</span> episodes and hence we don't need to track the episode up to termination. TD learning is so important that Sutton & Barto (2017) in their RL book describes it as "one idea … central and novel to reinforcement learning".


#### <mark style='background-color: #ffdce0'> Bootstrapping </mark>

TD learning methods update targets with regard to existing estimates rather than exclusively relying on actual rewards and complete returns as in MC methods. This approach is known as **bootstrapping**.


#### <mark style='background-color: #ffdce0'> Value Estimation </mark>

The key idea in TD learning is to update the value function $$V(S_t)$$ towards an estimated return $$R_{t+1} + \gamma V(S_{t+1})$$ (known as "**TD target**"). To what extent we want to update the value function is controlled by the learning rate hyperparameter α:

$$
\begin{aligned}
V(S_t) &\leftarrow (1- \alpha) V(S_t) + \alpha G_t \\
V(S_t) &\leftarrow V(S_t) + \alpha (G_t - V(S_t)) \\
V(S_t) &\leftarrow V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_t))
\end{aligned}
$$

Similarly, for action-value estimation:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))
$$

Next, let's dig into the fun part on how to learn optimal policy in TD learning (aka "TD control"). Be prepared, you are gonna see many famous names of classic algorithms in this section.


#### <mark style='background-color: #ffdce0'> SARSA: On-Policy TD control </mark>

"SARSA" refers to the procedure of updaing Q-value by following a sequence of $$\dots, S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}, \dots$$. The idea follows the same route of [GPI](#policy-iteration). Within one episode, it works as follows:

1. Initialize $$t=0$$. 
2. Start with $$S_0$$ and choose action $$A_0 = \arg\max_{a \in \mathcal{A}} Q(S_0, a)$$, where $$\epsilon$$-greedy is commonly applied.
3. At time $$t$$, after applying action $$A_t$$, we observe reward $$R_{t+1}$$ and get into the next state $$S_{t+1}$$.
4. Then pick the next action in the same way as in step 2: $$A_{t+1} = \arg\max_{a \in \mathcal{A}} Q(S_{t+1}, a)$$.
5. Update the Q-value function: $$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)) $$.
6. Set $$t = t+1$$ and repeat from step 3.

In each step of SARSA, we need to choose the *next* action according to the *current* policy.


#### <mark style='background-color: #ffdce0'> Q-Learning: Off-policy TD control </mark>

The development of Q-learning ([Watkins & Dayan, 1992](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)) is a big breakout in the early days of Reinforcement Learning. Within one episode, it works as follows:

1. Initialize $$t=0$$. 
2. Starts with $$S_0$$.
3. At time step $$t$$, we pick the action according to Q values, $$A_t = \arg\max_{a \in \mathcal{A}} Q(S_t, a)$$ and $$\epsilon$$-greedy is commonly applied.
4. After applying action $$A_t$$, we observe reward $$R_{t+1}$$ and get into the next state $$S_{t+1}$$.
5. Update the Q-value function: $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma \max_{a \in \mathcal{A}} Q(S_{t+1}, a) - Q(S_t, A_t))$$.
4. $$t = t+1$$ and repeat from step 3.

The key difference from SARSA is that Q-learning does not follow the current policy to pick the second action $$A_{t+1}$$. It estimates $$Q^*$$ out of the best Q values, but which action (denoted as $$a^*$$) leads to this maximal Q does not matter and in the next step Q-learning may not follow $$a^*$$.


![image](https://user-images.githubusercontent.com/48202736/106124940-fbabcf00-619e-11eb-9129-e43c7f5e3a66.png)
{: style="width: 50%;" class="center"}
*Fig. 6. The backup diagrams for Q-learning and SARSA. (Image source: Replotted based on Figure 6.5 in Sutton & Barto (2017))*


#### <mark style='background-color: #ffdce0'> Deep Q-Network </mark>

Theoretically, we can memorize $$Q_*(.)$$ for all state-action pairs in Q-learning, like in a gigantic table. However, it quickly becomes computationally infeasible when the state and action space are large. Thus people use functions (i.e. a machine learning model) to approximate Q values and this is called **function approximation**. For example, if we use a function with parameter $$\theta$$ to calculate Q values, we can label Q value function as $$Q(s, a; \theta)$$.

Unfortunately Q-learning may suffer from instability and divergence when combined with an nonlinear Q-value function approximation and [bootstrapping](#bootstrapping) (See [Problems #2](#deadly-triad-issue)).

Deep Q-Network ("DQN"; Mnih et al. 2015) aims to greatly improve and stabilize the training procedure of Q-learning by two innovative mechanisms:
- **Experience Replay**: All the episode steps $$e_t = (S_t, A_t, R_t, S_{t+1})$$ are stored in one replay memory $$D_t = \{ e_1, \dots, e_t \}$$. $$D_t$$ has experience tuples over many episodes. During Q-learning updates, samples are drawn at random from the replay memory and thus one sample could be used multiple times. Experience replay improves data efficiency, removes correlations in the observation sequences, and smooths over changes in the data distribution.
- **Periodically Updated Target**: Q is optimized towards target values that are only periodically updated. The Q network is cloned and kept frozen as the optimization target every C steps (C is a hyperparameter). This modification makes the training more stable as it overcomes the short-term oscillations. 

The loss function looks like this:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim U(D)} \Big[ \big( r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta) \big)^2 \Big]
$$

where $$U(D)$$ is a uniform distribution over the replay memory D; $$\theta^{-}$$ is the parameters of the frozen target Q-network.

In addition, it is also found to be helpful to clip the error term to be between [-1, 1]. (I always get mixed feeling with parameter clipping, as many studies have shown that it works empirically but it makes the math much less pretty. :/)


![image](https://user-images.githubusercontent.com/48202736/106124954-ff3f5600-619e-11eb-9fd4-ba2c7aed450f.png)
{: style="width: 75%;" class="center"}
*Fig. 7. Algorithm for DQN with experience replay and occasionally frozen optimization target. The prepossessed sequence is the output of some processes running on the input images of Atari games. Don't worry too much about it; just consider them as input feature vectors. (Image source: Mnih et al. 2015)* 


There are many extensions of DQN to improve the original design, such as DQN with dueling architecture (Wang et al. 2016) which estimates state-value function V(s) and advantage function A(s, a) with shared network parameters. 


### <mark style='background-color:black; color:white'>(7/11)</mark> <mark style='background-color: #dcffe4'> Combining TD and MC Learning </mark>

In the previous [section](#value-estimation) on value estimation in TD learning, we only trace one step further down the action chain when calculating the TD target. One can easily extend it to take multiple steps to estimate the return. 

Let's label the estimated return following n steps as $$G_t^{(n)}, n=1, \dots, \infty$$, then:

{: class="info"}
| $$n$$        | $$G_t$$           | Notes  |
| ------------- | ------------- | ------------- |
| $$n=1$$ | $$G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$$ | TD learning |
| $$n=2$$ | $$G_t^{(2)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})$$ | |
| ... | | |
| $$n=n$$ | $$ G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}) $$ | |
| ... | | |
| $$n=\infty$$ | $$G_t^{(\infty)} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{T-t-1} R_T + \gamma^{T-t} V(S_T) $$ | MC estimation |

The generalized n-step TD learning still has the [same](#value-estimation) form for updating the value function:

$$
V(S_t) \leftarrow V(S_t) + \alpha (G_t^{(n)} - V(S_t))
$$

![image](https://user-images.githubusercontent.com/48202736/106124984-04040a00-619f-11eb-994e-b4132aed263b.png)
{: style="width: 70%;" class="center"}


We are free to pick any $$n$$ in TD learning as we like. Now the question becomes what is the best $$n$$? Which $$G_t^{(n)}$$ gives us the best return approximation? A common yet smart solution is to apply a weighted sum of all possible n-step TD targets rather than to pick a single best n. The weights decay by a factor λ with n, $$\lambda^{n-1}$$; the intuition is similar to [why](#value-estimation) we want to discount future rewards when computing the return: the more future we look into the less confident we would be. To make all the weight (n → ∞) sum up to 1, we multiply every weight by (1-λ), because:

$$
\begin{aligned}
\text{let } S &= 1 + \lambda + \lambda^2 + \dots \\
S &= 1 + \lambda(1 + \lambda + \lambda^2 + \dots) \\
S &= 1 + \lambda S \\
S &= 1 / (1-\lambda)
\end{aligned}
$$

This weighted sum of many n-step returns is called λ-return $$G_t^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$. TD learning that adopts λ-return for value updating is labeled as **TD(λ)**. The original version we introduced [above](#value-estimation) is equivalent to **TD(0)**.


![image](https://user-images.githubusercontent.com/48202736/106125008-06fefa80-619f-11eb-8200-907f34c266e4.png)
{: class="center"}
*Fig. 8. Comparison of the backup diagrams of Monte-Carlo, Temporal-Difference learning, and Dynamic Programming for state value functions. (Image source: David Silver's RL course [lecture 4](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf): "Model-Free Prediction")*


### <mark style='background-color:black; color:white'>(8/11)</mark> <mark style='background-color: #dcffe4'> Policy Gradient </mark>

여태까지 배운 방법들은 모두 행동 가치 함수나 상태 가치 함수를 학습하는 데 중점을 두고, 그에 따라서 액션을 취해왔습니다. 
(가치 함수를 학습하는것이 곧 어떤 상태에서 어떤 액션을 취하라고 하는 policy를 학습시켜왔음) <br>
하지만 앞으로 배우게 될 Policy Gradient 방법들은 direct로 policy를 학습하게 됩니다.
policy는 $$\theta$$로 이루어진 함수이면 $$\pi(a \vert s; \theta)$$로 나타냅니다. <br><br>

자 이제 reward function을 *the expected return*로 정의하도록 합시다, 그리고 이 reward function을 최대화 하는 방식으로 학습을 하도록 합시다. 
(일반적인 머신러닝에서 loss 함수의 반대되는 말, 왜냐면 loss는 줄여야 사는거고 우리가 원하는건 reward를 크게 하는거니까) <br><br>

앞으로의 내용들은 왜 policy gradient theorem이 제대로 동작하는지와 중요한 policy gradient 알고리즘들을 설명하는데 중점을 둘 겁니다. <br>

Discrete space 에서는:

$$
\mathcal{J}(\theta) = V_{\pi_\theta}(S_1) = \mathbb{E}_{\pi_\theta}[V_1]
$$

여기서 $$S_1$$ 는 스타팅포인트 입니다. 

또는 continuous space 에서는:

$$
\mathcal{J}(\theta) = \sum_{s \in \mathcal{S}} d_{\pi_\theta}(s) V_{\pi_\theta}(s) = \sum_{s \in \mathcal{S}} \Big( d_{\pi_\theta}(s) \sum_{a \in \mathcal{A}} \pi(a \vert s, \theta) Q_\pi(s, a) \Big)
$$

여기서 $$d_{\pi_\theta}(s)$$ 는 $$\pi_\theta$$를 위한 마르코프 체인의 stationary distribution 입니다. 

여기서 $$d_{\pi_\theta}(s)$$ is stationary distribution of Markov chain for $$\pi_\theta$$. If you are unfamiliar with the definition of a "stationary distribution," please check this [reference](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/).

우리는 *gradient ascent* 방식을 사용함으로써 가장 큰 return을 뽑아내는 정책의 $$\theta$$를 얻을 수 있을겁니다.
그리고 continuous space 에서는 policy-based methods를 쓰는게 더 낫습니다, 왜냐면 취할 수 있는 액션이나 상태가 무한대에 가깝기 때문에 계산상에서도 그렇고 여러모로 더 자연스럽기 때문입니다.

#### <mark style='background-color: #ffdce0'> Policy Gradient Theorem </mark>

Computing the gradient *numerically* can be done by perturbing θ by a small amount ε in the k-th dimension. It works even when $$J(\theta)$$ is not differentiable (nice!), but unsurprisingly very slow.

$$
\frac{\partial \mathcal{J}(\theta)}{\partial \theta_k} \approx \frac{\mathcal{J}(\theta + \epsilon u_k) - \mathcal{J}(\theta)}{\epsilon}
$$

Or *analytically*,

$$
\mathcal{J}(\theta) = \mathbb{E}_{\pi_\theta} [r] = \sum_{s \in \mathcal{S}} d_{\pi_\theta}(s) \sum_{a \in \mathcal{A}} \pi(a \vert s; \theta) R(s, a)
$$

Actually we have nice theoretical support for (replacing $$d(.)$$ with $$d_\pi(.)$$):

$$
\mathcal{J}(\theta) = \sum_{s \in \mathcal{S}} d_{\pi_\theta}(s) \sum_{a \in \mathcal{A}} \pi(a \vert s; \theta) Q_\pi(s, a) \propto \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi(a \vert s; \theta) Q_\pi(s, a)
$$

Check Sec 13.1 in Sutton & Barto (2017) for why this is the case.

Then,

$$
\begin{aligned}
\mathcal{J}(\theta) &= \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi(a \vert s; \theta) Q_\pi(s, a) \\
\nabla \mathcal{J}(\theta) &= \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \nabla \pi(a \vert s; \theta) Q_\pi(s, a) \\
&= \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi(a \vert s; \theta) \frac{\nabla \pi(a \vert s; \theta)}{\pi(a \vert s; \theta)} Q_\pi(s, a) \\
& = \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi(a \vert s; \theta) \nabla \ln \pi(a \vert s; \theta) Q_\pi(s, a) \\
& = \mathbb{E}_{\pi_\theta} [\nabla \ln \pi(a \vert s; \theta) Q_\pi(s, a)]
\end{aligned}
$$

This result is named "Policy Gradient Theorem" which lays the theoretical foundation for various policy gradient algorithms:

$$
\nabla \mathcal{J}(\theta) = \mathbb{E}_{\pi_\theta} [\nabla \ln \pi(a \vert s, \theta) Q_\pi(s, a)]
$$


#### <mark style='background-color: #ffdce0'> REINFORCE </mark>

REINFORCE, also known as Monte-Carlo policy gradient, relies on $$Q_\pi(s, a)$$, an estimated return by [MC](#monte-carlo-methods) methods using episode samples, to update the policy parameter $$\theta$$.

A commonly used variation of REINFORCE is to subtract a baseline value from the return $$G_t$$ to reduce the variance of gradient estimation while keeping the bias unchanged. For example, a common baseline is state-value, and if applied, we would use $$A(s, a) = Q(s, a) - V(s)$$ in the gradient ascent update.

1. Initialize θ at random
2. Generate one episode $$S_1, A_1, R_2, S_2, A_2, \dots, S_T$$
3. For t=1, 2, ... , T:
	1. Estimate the the return G_t since the time step t.
	2. $$\theta \leftarrow \theta + \alpha \gamma^t G_t \nabla \ln \pi(A_t \vert S_t, \theta)$$.


#### <mark style='background-color: #ffdce0'> Actor-Critic </mark>

If the value function is learned in addition to the policy, we would get Actor-Critic algorithm.
- **Critic**: updates value function parameters w and depending on the algorithm it could be action-value $$Q(a \vert s; w)$$ or state-value $$V(s; w)$$.
- **Actor**: updates policy parameters θ, in the direction suggested by the critic, $$\pi(a \vert s; \theta)$$.

Let's see how it works in an action-value actor-critic algorithm. 

1. Initialize s, θ, w at random; sample $$a \sim \pi(a \vert s; \theta)$$.
2. For t = 1… T:
	1. Sample reward $$r_t  \sim R(s, a)$$ and next state $$s' \sim P(s' \vert s, a)$$.
	2. Then sample the next action $$a' \sim \pi(s', a'; \theta)$$.
	3. Update policy parameters: $$\theta \leftarrow \theta + \alpha_\theta Q(s, a; w) \nabla_\theta \ln \pi(a \vert s; \theta)$$.
	4. Compute the correction for action-value at time t: <br/>
	$$G_{t:t+1} = r_t + \gamma Q(s', a'; w) - Q(s, a; w)$$ <br/>
	and use it to update value function parameters: <br/>
	$$w \leftarrow w + \alpha_w G_{t:t+1} \nabla_w Q(s, a; w) $$.
	5. Update $$a \leftarrow a'$$ and $$s \leftarrow s'$$.

$$\alpha_\theta$$ and $$\alpha_w$$ are two learning rates for policy and value function parameter updates, respectively.


#### <mark style='background-color: #ffdce0'> A3C </mark>

**Asynchronous Advantage Actor-Critic** (Mnih et al., 2016), short for A3C, is a classic policy gradient method with the special focus on parallel training. 

In A3C, the critics learn the state-value function, $$V(s; w)$$, while multiple actors are trained in parallel and get synced with global parameters from time to time. Hence, A3C is good for parallel training by default, i.e. on one machine with multi-core CPU.

The loss function for state-value is to minimize the mean squared error, $$\mathcal{J}_v (w) = (G_t - V(s; w))^2$$ and we use gradient descent to find the optimal w. This state-value function is used as the baseline in the policy gradient update.

Here is the algorithm outline:
1. We have global parameters, θ and w; similar thread-specific parameters, θ' and w'.
2. Initialize the time step t = 1
3. While T <= T_MAX:
	1. Reset gradient: dθ = 0 and dw = 0.
	2. Synchronize thread-specific parameters with global ones: θ' = θ and w' = w.
	3. $$t_\text{start}$$ = t and get $$s_t$$.
	4. While ($$s_t \neq \text{TERMINAL}$$) and ($$t - t_\text{start} <= t_\text{max}$$):
		1. Pick the action $$a_t \sim \pi(a_t \vert s_t; \theta')$$ and receive a new reward $$r_t$$ and a new state $$s_{t+1}$$.
		2. Update t = t + 1 and T = T + 1.
	5. Initialize the variable that holds the return estimation $$R = \begin{cases} 
		0 & \text{if } s_t \text{ is TERMINAL} \\
		V(s_t; w') & \text{otherwise}
		\end{cases}$$.
	6. For $$i = t-1, \dots, t_\text{start}$$:
		1. $$R \leftarrow r_i + \gamma R$$; here R is a MC measure of $$G_i$$.
		2. Accumulate gradients w.r.t. θ': $$d\theta \leftarrow d\theta + \nabla_{\theta'} \log \pi(a_i \vert s_i; \theta')(R - V(s_i; w'))$$;<br/>
		Accumulate gradients w.r.t. w': $$dw \leftarrow dw + \nabla_{w'} (R - V(s_i; w'))^2$$.
	7. Update synchronously θ using dθ, and w using dw.

A3C enables the parallelism in multiple agent training. The gradient accumulation step (6.2) can be considered as a reformation of minibatch-based stochastic gradient update: the values of w or θ get corrected by a little bit in the direction of each training thread independently.


### <mark style='background-color:black; color:white'>(9/11)</mark> <mark style='background-color: #dcffe4'> Evolution Strategies </mark>

[Evolution Strategies](https://en.wikipedia.org/wiki/Evolution_strategy) (ES) is a type of model-agnostic optimization approach. It learns the optimal solution by imitating Darwin's theory of the evolution of species by natural selection. Two prerequisites for applying ES: (1) our solutions can freely interact with the environment and see whether they can solve the problem; (2) we are able to compute a **fitness** score of how good each solution is. We don't have to know the environment configuration to solve the problem. 

Say, we start with a population of random solutions. All of them are capable of interacting with the environment and only candidates with high fitness scores can survive (*only the fittest can survive in a competition for limited resources*). A new generation is then created by recombining the settings (*gene mutation*) of high-fitness survivors. This process is repeated until the new solutions are good enough.

Very different from the popular MDP-based approaches as what we have introduced above, ES aims to learn the policy parameter $$\theta$$ without value approximation. Let's assume the distribution over the parameter $$\theta$$ is an [isotropic](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic) multivariate Gaussian with mean $$\mu$$ and fixed covariance $$\sigma^2I$$. The gradient of $$F(\theta)$$ is calculated:

$$
\begin{aligned}
& \nabla_\theta \mathbb{E}_{\theta \sim N(\mu, \sigma^2)} F(\theta) \\
=& \nabla_\theta \int_\theta F(\theta) \Pr(\theta) && \text{Pr(.) is the Gaussian density function.} \\
=& \int_\theta F(\theta) \Pr(\theta) \frac{\nabla_\theta \Pr(\theta)}{\Pr(\theta)} \\
=& \int_\theta F(\theta) \Pr(\theta) \nabla_\theta \log \Pr(\theta) \\
=& \mathbb{E}_{\theta \sim N(\mu, \sigma^2)} [F(\theta) \nabla_\theta \log \Pr(\theta)] && \text{Similar to how we do policy gradient update.} \\
=& \mathbb{E}_{\theta \sim N(\mu, \sigma^2)} \Big[ F(\theta) \nabla_\theta \log \Big( \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(\theta - \mu)^2}{2 \sigma^2 }} \Big) \Big] \\
=& \mathbb{E}_{\theta \sim N(\mu, \sigma^2)} \Big[ F(\theta) \nabla_\theta \Big( -\log \sqrt{2\pi\sigma^2} - \frac{(\theta - \mu)^2}{2 \sigma^2} \Big) \Big] \\
=& \mathbb{E}_{\theta \sim N(\mu, \sigma^2)} \Big[ F(\theta) \frac{\theta - \mu}{\sigma^2} \Big]
\end{aligned}
$$


We can rewrite this formula in terms of a "mean" parameter $$\theta$$ (different from the $$\theta$$ above; this $$\theta$$ is the base gene for further mutation), $$\epsilon \sim N(0, I)$$ and therefore $$\theta + \epsilon \sigma \sim N(\theta, \sigma^2)$$. $$\epsilon$$ controls how much Gaussian noises should be added to create mutation:

$$
\nabla_\theta \mathbb{E}_{\epsilon \sim N(0, I)} F(\theta + \sigma \epsilon) = \frac{1}{\sigma} \mathbb{E}_{\epsilon \sim N(0, I)} [F(\theta + \sigma \epsilon) \epsilon]
$$


![image](https://user-images.githubusercontent.com/48202736/106125385-7379f980-619f-11eb-9f6e-8817032f662d.png)
{: class="center"}
*Fig. 9. A simple parallel evolution-strategies-based RL algorithm. Parallel workers share the random seeds so that they can reconstruct the Gaussian noises with tiny communication bandwidth. (Image source: Salimans et al. 2017.)*


ES, as a black-box optimization algorithm, is another approach to RL problems (<span style="color: #999999;">*In my original writing, I used the phrase "a nice alternative"; [Seita](https://danieltakeshi.github.io/) pointed me to this [discussion](https://www.reddit.com/r/MachineLearning/comments/6gke6a/d_requesting_openai_to_justify_the_grandiose/dir9wde/) and thus I updated my wording.*</span>). It has a couple of good characteristics (Salimans et al., 2017) keeping it fast and easy to train:
- ES does not need value function approximation;
- ES does not perform gradient back-propagation;
- ES is invariant to delayed or long-term rewards;
- ES is highly parallelizable with very little data communication.


## Known Problems

### <mark style='background-color:black; color:white'>(10/11)</mark> <mark style='background-color: #dcffe4'> Exploration-Exploitation Dilemma </mark>

The problem of exploration vs exploitation dilemma has been discussed in my previous post. When the RL problem faces an unknown environment, this issue is especially a key to finding a good solution: without enough exploration, we cannot learn the environment well enough; without enough exploitation, we cannot complete our reward optimization task.

Different RL algorithms balance between exploration and exploitation in different ways. In [MC](#monte-carlo-methods) methods, [Q-learning](#q-learning-off-policy-td-control) or many on-policy algorithms, the exploration is commonly implemented by ε-greedy; In [ES](#evolution-strategies), the exploration is captured by the policy parameter perturbation. Please keep this into consideration when develop a new RL algorithm.

### <mark style='background-color:black; color:white'>(11/11)</mark> <mark style='background-color: #dcffe4'> Deadly Triad Issue </mark>

We do seek the efficiency and flexibility of TD methods that involve bootstrapping. However, when off-policy, nonlinear function approximation, and bootstrapping are combined in one RL algorithm, the training could be unstable and hard to converge. This issue is known as the **deadly triad** (Sutton & Barto, 2017). Many architectures using deep learning models were proposed to resolve the problem, including DQN to stabilize the training with experience replay and occasionally frozen target network.


## <mark style='background-color: #fff5b1'> Case Study: AlphaGo Zero </mark>

The game of [Go](https://en.wikipedia.org/wiki/Go_(game)) has been an extremely hard problem in the field of Artificial Intelligence for decades until recent years. AlphaGo and AlphaGo Zero are two programs developed by a team at DeepMind. Both involve deep Convolutional Neural Networks (CNN) and Monte Carlo Tree Search (MCTS) and both have been approved to achieve the level of professional human Go players. Different from AlphaGo that relied on supervised learning from expert human moves, AlphaGo Zero used only reinforcement learning and self-play without human knowledge beyond the basic rules.

![image](https://user-images.githubusercontent.com/48202736/106125396-770d8080-619f-11eb-99ca-9bd75c50ce65.png)
{: class="center"}
*Fig. 10. The board of Go. Two players play black and white stones alternatively on the vacant intersections of a board with 19 x 19 lines. A group of stones must have at least one open point (an intersection, called a "liberty") to remain on the board and must have at least two or more enclosed liberties (called "eyes") to stay "alive". No stone shall repeat a previous position.*

With all the knowledge of RL above, let's take a look at how AlphaGo Zero works. The main component is a deep CNN over the game board configuration (precisely, a ResNet with batch normalization and ReLU). This network outputs two values:

$$
(p, v) = f_\theta(s)
$$

- $$s$$: the game board configuration, 19 x 19 x 17 stacked feature planes; 17 features for each position, 8 past configurations (including current) for the current player + 8 past configurations for the opponent + 1 feature indicating the color (1=black, 0=white). We need to code the color specifically because the network is playing with itself and the colors of current player and opponents are switching between steps.
- $$p$$: the probability of selecting a move over 19^2 + 1 candidates (19^2 positions on the board, in addition to passing).
- $$v$$: the winning probability given the current setting.

During self-play, MCTS further improves the action probability distribution $$\pi \sim p(.)$$ and then the action $$a_t$$ is sampled from this improved policy. The reward $$z_t$$ is a binary value indicating whether the current player *eventually* wins the game. Each move generates an episode tuple $$(s_t, \pi_t, z_t)$$ and it is saved into the replay memory. The details on MCTS are skipped for the sake of space in this post; please read the original [paper](https://www.dropbox.com/s/yva172qos2u15hf/2017-silver.pdf?dl=0) if you are interested.


![image](https://user-images.githubusercontent.com/48202736/106125407-7a087100-619f-11eb-8532-e55f39f36b0f.png)
{: class="center"}
*Fig. 11. AlphaGo Zero is trained by self-play while MCTS improves the output policy further in every step. (Image source: Figure 1a in Silver et al., 2017).*

The network is trained with the samples in the replay memory to minimize the loss:

$$
\mathcal{L} = (z - v)^2 - \pi^\top \log p + c \| \theta \|^2
$$

where $$c$$ is a hyperparameter controlling the intensity of L2 penalty to avoid overfitting.

AlphaGo Zero simplified AlphaGo by removing supervised learning and merging separated policy and value networks into one. It turns out that AlphaGo Zero achieved largely improved performance with a much shorter training time! I strongly recommend reading these [two](https://pdfs.semanticscholar.org/1740/eb993cc8ca81f1e46ddaadce1f917e8000b5.pdf) [papers](https://www.dropbox.com/s/yva172qos2u15hf/2017-silver.pdf?dl=0) side by side and compare the difference, super fun.

I know this is a long read, but hopefully worth it. *If you notice mistakes and errors in this post, don't hesitate to contact me at [lilian dot wengweng at gmail dot com].* See you in the next post! :)



## <mark style='background-color: #fff5b1'> References </mark>

1. [Original post written by Lilian Weng](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#common-approaches)

2. [Deep Learning in a Nutshell: Reinforcement Learning (Nvidia)](https://developer.nvidia.com/blog/deep-learning-nutshell-reinforcement-learning/)

3. [Reinforcement Demo from Karpathy](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html)

4. [Massimiliano Patacchiola Blog](https://mpatacchiola.github.io/blog/)

5. [Presentation for Introduction of Deep Reinforcement Learning from Donghyun Kwak](https://www.youtube.com/watch?v=dw0sHzE1oAc)

6. [2015 UCL Lecture from Davide Silver](https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver)

7. [2017 Berkeley Lecture from Pieter Abbeel](https://sites.google.com/view/deep-rl-bootcamp/lectures)
