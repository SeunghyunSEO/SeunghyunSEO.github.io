<pre>
<code>

Consider speech recognition. We have a dataset of audio clips and corresponding transcripts. Unfortunately, we don’t know how the characters in the transcript align to the audio. This makes training a speech recognizer harder than it might at first seem.

Without this alignment, the simple approaches aren’t available to us. We could devise a rule like “one character corresponds to ten inputs”. But people’s rates of speech vary, so this type of rule can always be broken. Another alternative is to hand-align each character to its location in the audio. From a modeling standpoint this works well — we’d know the ground truth for each input time-step. However, for any reasonably sized dataset this is prohibitively time consuming.

This problem doesn’t just turn up in speech recognition. We see it in many other places. Handwriting recognition from images or sequences of pen strokes is one example. Action labelling in videos is another.

음성인식에 대해 생각해보자.
데이터로 음성파일과 그에 해당하는 정답 스크립트가 주어질것이다.
예를들어 음성파일의 내용이 '안녕하세요' 라고 하자.
음성인식에서 가장 우리가 중요하게 생각해야 할 부분은 '안녕하세요'의 '안'이 과연 음성파일의 어느 부분의 정보를 지칭하느냐 이다.

Connectionist Temporal Classification (CTC) is a way to get around not knowing the alignment between the input and the output. As we’ll see, it’s especially well suited to applications like speech and handwriting recognition.

CTC는 바로 이러한 align(각 문자를 음성의 어느 부분에 매칭시켜줄지를 할당하는 것) 문제를 해결해기 위해 고안되었다.

The CTC algorithm overcomes these challenges. For a given XX it gives us an output distribution over all possible YY’s. We can use this distribution either to infer a likely output or to assess the probability of a given output.

Not all ways of computing the loss function and performing inference are tractable. We’ll require that CTC do both of these efficiently.

## algorithm

The CTC algorithm can assign a probability for any Y given an X. The key to computing this probability is how CTC thinks about alignments between inputs and outputs. We’ll start by looking at these alignments and then show how to use them to compute the loss function and perform inference.

https://distill.pub/2017/ctc/assets/naive_alignment.svg

</code>
</pre>
