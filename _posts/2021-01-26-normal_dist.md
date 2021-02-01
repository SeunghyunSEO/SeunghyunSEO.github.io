---
title: Gaussian distribution (a.k.a normal distribution)
categories: Probability&Statistics
tag: 

toc: true
toc_sticky: true
---

- <mark style='background-color: #fff5b1'> Why Gaussian Distribution? </mark>


- <mark style='background-color: #fff5b1'> Univariate Gaussian Distribution </mark>

가우시안(Gaussian) 분포 혹은 정규(Normal) 분포는 변수가 1개냐 2개냐 ... 여러개냐에 따라서 (일변량)Univariate 분포 혹은 이변량(Bivariate), 다변량(Multivariate) 분포로 나눠 생각할 수 있습니다.

Univariate Gaussian Distribution의 모양과 수식은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/48202736/106379093-fd6ad200-63ec-11eb-9b6f-f8ad3b5448c1.png)

![image](https://user-images.githubusercontent.com/48202736/106379103-0360b300-63ed-11eb-910a-63e254e01682.png)

일변량 정규 분포는 1개의 연속적인 변수에 대한 확률 값을 뱉는 함수가 되고,

수식을 다시 풀어서 쓰면 아래와 같이 쓸 수 있습니다.

<center>$$ Pr(x) = Norm_x[\mu,\sigma^2] $$</center>

<center>$$ Pr(x) = \frac{1}{\sqrt{w\pi\sigma^2}}exp[-\frac{1}{2} (x-\mu)^2 / \sigma^2 ] $$</center>

데이터 x를 표현하는 위의 분포의 파라메터는 평균(mean, $$\mu$$), 분산(variance, $$\sigma^2$$) 두가지 뿐입니다.

- <mark style='background-color: #fff5b1'> Multivariate Gaussian Distribution </mark>

![image](https://user-images.githubusercontent.com/48202736/106442710-f1e7db80-64be-11eb-9810-954a14c0ed74.png)
(이미지 출처 : [link](https://ko.wikipedia.org/wiki/%EB%8B%A4%EB%B3%80%EB%9F%89_%EC%A0%95%EA%B7%9C%EB%B6%84%ED%8F%AC))

![image](https://user-images.githubusercontent.com/48202736/106379113-13789280-63ed-11eb-9a8a-3ee82f60c4cc.png)

- <mark style='background-color: #dcffe4'> Bivariate Gaussian Distribution </mark>

![image](https://user-images.githubusercontent.com/48202736/106379271-0dcf7c80-63ee-11eb-80b1-8a401837c6a4.png)

- <mark style='background-color: #dcffe4'> Covariance Matrix </mark>

![image](https://user-images.githubusercontent.com/48202736/106379277-1b850200-63ee-11eb-85aa-aceece871413.png)

- <mark style='background-color: #fff5b1'> Conditional Gaussian Distribution </mark>

![image](https://user-images.githubusercontent.com/48202736/106379157-520e4d00-63ed-11eb-91f3-957b610e1eb1.png)

- <mark style='background-color: #fff5b1'> Marginal Gaussian Distribution </mark>

![image](https://user-images.githubusercontent.com/48202736/106379160-55093d80-63ed-11eb-9da8-4cdbac065b18.png)
![image](https://user-images.githubusercontent.com/48202736/106379163-58042e00-63ed-11eb-98de-0b82c005de7c.png)

![image](https://user-images.githubusercontent.com/48202736/106379164-5a668800-63ed-11eb-993b-ea09b72ac61a.png)
![image](https://user-images.githubusercontent.com/48202736/106379166-5c304b80-63ed-11eb-8e3c-761669998966.png)


- <mark style='background-color: #fff5b1'> Conjugate Distribution  </mark>

- <mark style='background-color: #dcffe4'> Inverse Gamma  </mark>

![image](https://user-images.githubusercontent.com/48202736/106379109-0a87c100-63ed-11eb-8e80-c8d642f20d22.png)
![image](https://user-images.githubusercontent.com/48202736/106379111-0c518480-63ed-11eb-8269-26244ea6bfea.png)

- <mark style='background-color: #dcffe4'> Normal X Nomral  </mark>

![image](https://user-images.githubusercontent.com/48202736/106379170-5fc3d280-63ed-11eb-95d1-7e2d91119b90.png)
