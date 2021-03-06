---
title: (yet) EM Algorithm
categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true
---

---
< 목차 >
{: class="table-of-content"}
* TOC
{:toc}
---

## <mark style='background-color: #fff5b1'> Models with hidden variables </mark>

우리가 맨처음에 Maximum likelihood 방법에 대해 논할 때 생각했던 그림을 봅시다.

![likelihood1](https://user-images.githubusercontent.com/48202736/106548623-ad0d8480-6552-11eb-8f3f-fc87abfea625.png)

우리는 단지 위의 그림처럼 한개의 봉우리를 가지고있는(unimodal) 가우시안 함수에 대해서 likelihood를 최대화 하는 파라메터(mean, variance)를 찾곤 했습니다.

하지만 우리가 추정하고자하는 분포가 이보다 더 복잡하다면 어떡할까요? 예를들어 봉우리가 여러개인(multimodal)인 경우는요? 

![image1](https://user-images.githubusercontent.com/48202736/106545430-b85db180-654c-11eb-809d-e2a9727670df.png)

이럴 경우 우리는 찾고자 하는 p(x) 분포에 대해서 (물론 파라메터 $$\theta$$를 도입하겠죠) 우리가 알 수 없는 숨겨진 변수인 hidden variable을 도입해 생각할 수 있습니다. 


그림으로 생각해보면 바로 아래와 같은 경우죠

![mog4](https://user-images.githubusercontent.com/48202736/106545469-c6133700-654c-11eb-9bde-f4787a1012ac.png)
{: style="width: 70%;" class="center"}

위에서 말했듯 ```Key idea```는 Pr(x) 분포에 우리가 알 수 없는 hidden variable h를 추가해서 생각해보자는 겁니다.

![hidden](https://user-images.githubusercontent.com/48202736/106546381-6b7ada80-654e-11eb-8dbb-dc703b8c3a5b.png)
{: style="width: 70%;" class="center"}

<center>$$ Pr(x) = \int Pr(x,h) dh $$</center>

우리는 위의 식을 또한 아래와 같이 쓸 수도 있겠죠

<center>$$ Pr(x|\theta) = \int Pr(x,h|\theta) dh $$</center>

과연 위의 pdf를 어떻게하면 데이터에 맞게 제대로 피팅할 수 있을까요? 

<center>$$ \hat{\theta} = \arg \max_{\theta} [ \sum_{i=1}^{I} log [\int Pr(x_i,h_i|\theta) dh_i] ]  $$</center>

잘 알려진 방법중에 하나는, log likelihood에 대한 lower bound를 정의하고 그 bound를 반복을 통해 증가시키는 겁니다.

<center>$$ B[\{ q_i(h_i) \}, \theta] = \sum_{i=1}^{I} \int q_i(h_i) log[\frac{Pr(x,h_i|\theta)}{q_i(h_i)}]dh_{1...I} $$</center>
<center>$$ \leq [ \sum_{i=1}^{I} log [\int Pr(x_i,h|\theta) dh_i] ]  $$</center>




## <mark style='background-color: #fff5b1'> (E)xpectation and (M)aximization </mark>




## <mark style='background-color: #fff5b1'> Lower Bound </mark>

<center>$$ B[\{ q_i(h_i) \}, \theta] = \sum_{i=1}^{I} \int q_i(h_i) log[\frac{Pr(x,h_i|\theta)}{q_i(h_i)}]dh_{1...I} $$</center>

![lowerbound1](https://user-images.githubusercontent.com/48202736/106545443-bc89cf00-654c-11eb-9be5-301120d70938.png)




## <mark style='background-color: #fff5b1'> E-Step & M-Step </mark>

우리가 사용할 log likelihood를 최대화 하는 방법은 E Step과 M Step이 결합된 EM Algorithm입니다.  


여기서 E Step은 주어진 파라메터의 초기값에 대해서 likelihood와 최대한 근사한 likelihood를 얻어내는 것이고
M Step은 말그대로 이 likelihood를 최대화 하는 파라메터값을 수식으로 얻어내는겁니다.


그리고 다시 한 번 그 파라메터를 기점으로 E Step을 진행 (likelihood를 업데이트하고) 또 M Step을 진행하고 ... 의 반복인겁니다.


E Step은 분포 $$q_i(h_i)$$에 대해서 bound를 최대화 하는 것으로 수식으로 나타내면 아래와 같습니다.

<center>$$ \hat{q_i}(h_i) = Pr(h_i|x_i,\theta^{[t]}) = \frac{Pr(x_i|h_i,\theta^{[t]}) Pr(h_i|\theta^{[t]})}{Pr(x_i)} $$</center>

M Step은 분포 $$\theta$$에 대해서 bound를 최대화 하는 것으로 수식으로 나타내면 아래와 같습니다.

<center>$$ \hat{\theta}^{[t+1]} = \arg \max_{\theta} [ \sum_{i=1}^{I} \int \hat{q_i}(h_i) log [Pr(x_i,h_i|\theta)] dh_i ] $$</center>

이 두 가지 step을 iterative하게 그림으로 나타내면 아래와 같습니다.

![em1](https://user-images.githubusercontent.com/48202736/106545453-c01d5600-654c-11eb-9912-9b3dac3d146a.png)
{: style="width: 70%;" class="center"}



## <mark style='background-color: #fff5b1'> Mixture of Gaussian (MoG) Example </mark>

![mog1](https://user-images.githubusercontent.com/48202736/106545456-c1e71980-654c-11eb-9d08-494728c0b5cd.png)
{: style="width: 70%;" class="center"}

![mog2](https://user-images.githubusercontent.com/48202736/106545460-c3184680-654c-11eb-8807-e7d84e8c5076.png)
{: style="width: 70%;" class="center"}

![mog3](https://user-images.githubusercontent.com/48202736/106545466-c4e20a00-654c-11eb-8a0a-acd04d74c6a2.png)
{: style="width: 70%;" class="center"}

![mog4](https://user-images.githubusercontent.com/48202736/106545469-c6133700-654c-11eb-9bde-f4787a1012ac.png)
{: style="width: 70%;" class="center"}

![mog5](https://user-images.githubusercontent.com/48202736/106545473-c7dcfa80-654c-11eb-94dc-f753d31173d1.png)

![mog6](https://user-images.githubusercontent.com/48202736/106545477-c90e2780-654c-11eb-8d12-e4efb84b63ba.png)
{: style="width: 70%;" class="center"}

![mog7](https://user-images.githubusercontent.com/48202736/106545479-cad7eb00-654c-11eb-90a8-b148882e7f9b.png)

![mog8](https://user-images.githubusercontent.com/48202736/106545486-cc091800-654c-11eb-8d07-5a7b85e8286b.png)
