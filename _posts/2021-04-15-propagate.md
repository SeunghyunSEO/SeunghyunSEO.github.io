---
title: (미완) (Paper) Propagate Yourself, Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning
categories: DeepLearning
tag: [tmp]

toc: true
toc_sticky: true
---

---
< 목차 >
{: class="table-of-content"}
* TOC
{:toc}
---


이미지 레벨 (인스턴스 레벨)에서 contrastive learning을 하는것은 spatial sensitivity가 부족하다. 
그니까 dense downstream task에 대해서 잘 맞지 않는다는 것.
그래서 픽셀 레벨에서 하겠다.

컨트리뷰션 포인트는 sota임.

SSL 방법론들 -> contrastive가 대세 -> 인스턴스레벨이 대부분이며, negative나 postivie를 둘다 쓰거나 하나만 쓰거나 둘다 쓰곤 함.


근데 네거티브 쓰면 별로 안좋다고함. (일단 진짜 네거티브인지도 모르겠으며, 컴퓨테이셔녈 코스트가 큼)


supervised contrastive learning이라는 것도 있음.


픽셀간 유클리디안 디스턴스를 재서 스레시홀드보다 크면 파시티브 페어, 아니면 네가티브 페어가 되는것(맞지?).


(네트워크에서)
비욜이랑 비슷하게, view2에서 나오는 인코더는 학습이 안됨.

그리고 마지막 PPM 모듈이 셀프 시밀러리티한거랑, 트랜스폼 한거랑 해서 y를 산출하고 이를 x'랑 y랑 비교함.
 
 
마스크 만들어낼 때? 

->
디노이징과 스무딩의 효과가 있다고 하는데 PPM이 가우시안 필터같이 생긴거(샤프닝이랑 뭐랑) 이기 때문에 어떤 녀석이 그럼 스무딩 펙터 역할을 하는거지 (응?)
(코배리언스 매트릭스를 만들어내서 블러링 하는?)

PPM 모듈이 만들어내는 y는 어떤 의미일까?

similirity computation 모듈이 코배리언스 매트릭스가 되고 (가우시안 필터처럼 된다는 느낌?) , 여기에 g(x) 를 곱하면 음 
g 네트워크가 1x1 conv임.


(흠 코배리언스 정리좀 해야겠다.)


암튼 그리고 모멘텀 인코더는 뭐냐 0.99 레즈넷 + 0.01 이전의 인코더를 합쳐서


view1 view2 는 서로 다른 크롭의 이미지고


암튼 그러니까 이렇게 픽셀단위의 레프레젠테이션 벡터 의미가 중요한 (그니까 세그멘테이션 같은거?) 에서는 이게 좋다는거지? 픽셀 와이즈 레프레젠테이션을 배우니까 

(인스턴스레벨은 사람이 누워있는거랑 아닌거랑 같음)

pix contrast는 바인딩 박스에서 겹치는 부분만 파시티브 페어로.


이미지 처리 할 때 인접 픽셀이 유사해야 할거라는 그런 가정이 들어가는데,


결론은 디텍션과 세그멘테이션만 이다.

성능 비교했고


어블레이션으로, 타우를 0.7쓴거랑 다른거 쓴거 차이들

샤프니스텀도 2 쓴게 제일 좋다. (max(sim()),0)^2 인듯


뉴립스 2020에 나오는 VADeR 머가 차이냐? pixpro는 7x7 뽑은 다음에 어쩌고 이고, 베이더는 디코더가 달렸는데 흠 ...

1.베이더는 파라미터가 더 많고 (인코더만 썼고, 베이더는 인코더 디코더 둘다함), pixpro는 와핑해서 생각함.
2.똑같은 강아지의 눈인데 네거티브에 들어갈 수도 있음.
3.베이더는 파지티브 페어를 박스에서 랜덤샘플링함 (랜덤으로 32개)
4.픽스프로는 디스턴스 매칭을 쓰고 네거티브를 쓰지 않는다. 베이더는 모코 방법 씀


## <mark style='background-color: #fff5b1'> Abstract </mark>

## <mark style='background-color: #fff5b1'> Introduction </mark>

![Fig1](/assets/images/PRML_5.1_to_5.2/propagate_fig1.png)
*Fig.*

## <mark style='background-color: #fff5b1'> Related Works </mark>

## <mark style='background-color: #fff5b1'> Method </mark>

![Fig2](/assets/images/PRML_5.1_to_5.2/propagate_fig2.png)
*Fig.*

![Fig3](/assets/images/PRML_5.1_to_5.2/propagate_fig3.png)
*Fig.*

## <mark style='background-color: #fff5b1'> Experiments and Results </mark>

## <mark style='background-color: #fff5b1'> Conclusion and Future Work </mark>

## <mark style='background-color: #fff5b1'> References </mark>

asd
