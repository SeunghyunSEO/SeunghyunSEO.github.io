---
layout: post
title: SLU
---

### 2020/04/08 시작

<div style="font-size: 0.9rem; font-weight:300; line-height: 1.6rem;">
- "Recent Advances in Ent-to-End Spoken Language Understanding"이라는 페이퍼에 대한 리뷰임.<br>
- SLU란 무엇인가? 음성을 입력값으로 받아 ASR system을 통해 이를 문자로 변환하고, 이 문자를 다시 입력으로 받아 NLU system을 통해 Intent Classification 
혹은 Slot Filling 같은 task를 푸는 것이다.<br>
- 그 중에서 우리가 하고자 풀고자 하는 문제는 Slot Filling 이라는 task이다.<br>
- 이 논문에서 제시하는 바는 다음과 같다.
- SLU 네트워크를 통해서 named entity recognition (NER) 과 Semantic Slot filling (SF) 라는 task에 대해 실험을 할 것이고, 
이를 위해서 CTC criterion을 사용하거나 pretraining을 하는 등의 테크닉을 사용하여 성능 향상을 꾀한다.
</div>
