---
layout: post
title: SLU
---

### 2020/04/08 시작

<div style="font-size: 0.9rem; font-weight:300; line-height: 1.6rem;">
- "Recent Advances in Ent-to-End Spoken Language Understanding"이라는 페이퍼에 대한 리뷰이다.<br>
- SLU란 무엇인가? 전통적인 SLU system은 음성을 입력값으로 받아 ASR system을 통해 이를 문자로 변환하고, 이 문자를 다시 입력으로 받아 NLU system을 통해 Intent Classification 혹은 Slot Filling 같은 task를 푸는 시스템이다. 하지만 Ent-to-End SLU 라는 이름에서도 알 수 있듯이, 최근의 추세는 
전통적인 ( Speech -> ASR -> Text -> NLU -> SF or intent classification ) 흐름에서 중간의 ASR 과정을 생략하고, 바로 ( Speech -> SF or intent classfication ) 으로 이루어진 네트워크를 사용한다.<br>
- 위에서 언급한 task들 중에서 우리가 하고자 풀고자 하는 문제는 Slot Filling 이라는 task이다.<br>
- SLU를 위한 데이터셋은 흔히 다음과 같이 구성되어 있다.<br>
<pre>
<code>
['\ufefftype', 'turn', 'transcript', 'file_path', 'intent', 'slot', 'history', 'history_file_path']
['U', '6', '6명이에요.', 'AAA.wav', 'U-inform()', '<numPeople>6명이에요.</>', "['저 4월 20일 날 오후 5시에 예약 잡으려고 하는데 가능한가요?', '아 네, 가능합니다.', '아 그리고 저 부탁할 게 하나 있는데요.', '네 무엇인가요?', '그...저희가 사람이 많아서 창가쪽에 자리 좀 많은 데로 예약을 하고 싶은데 가능한지 궁금해서요..', '혹시 몇 명이신가요?']", "['BBB.wav', nan, 'CCC.wav', nan, 'DDD.wav', nan]"]
</code>
</pre>
- 자 우리가 하고싶은것은 다시 말하자면 위의 데이터에서 AAA.wav 를 네트워크 입력값으로 받아 결과값으로 <numPeople> : 6명이에요. 가 나오게 하고 싶은 것이다.
- 이 논문에서 제시하는 바는 다음과 같다.<br>
- SLU 네트워크를 통해서 named entity recognition (NER) 과 Semantic Slot filling (SF) 라는 task에 대해 실험을 할 것이고, 
이를 위해서 CTC criterion을 사용하거나 pretraining을 하는 등의 테크닉을 사용하여 성능 향상을 꾀한다.<br>
- 그리고 SLU 문제를 풀기 위해서는 대량의 발화 데이터가 필요한데, 이 방법 말고 SLU 성능을 향상시킬 방법을 제시한다고 한다.<br>
- 
</div>
