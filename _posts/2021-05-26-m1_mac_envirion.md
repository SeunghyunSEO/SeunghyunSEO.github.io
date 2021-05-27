---
title: M1 Mac 환경 세팅하기
categories: Coding
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

원래 M1 Macbook을 샀을 당시에 gpu가 없이 딥러닝 모델을 제대로 학습시킬 수 없다고 생각했고, 실제로 수많은 라이브러리에서 에러가 났기 때문에 일단 냅뒀는데,
뭐 나쁘지 않게 돌리 수 있다는 (?) 말을 듣고 설치해보려고한다. (근데 MNIST 정도 되는 사이즈의 이미지 분류 task나 잘 되는 것 같다... 1080ti급이라는 소리도 헛소리일 확률이 크지만 일단 해본다.) 

![mac_benchmark](/assets/images/m1_mac/mac_benchmark.png)
*Fig. The comparison of training performance*



순서는 

- brew 설치
- 가상 환경 (docker or conda) 설치
- torch 등 라이브러리 설치 
- 간단한 네트워크 러닝 해보기

가 될 것 같다.




## Homebrew 설치

우선 [homebrew link](https://brew.sh/index_ko)에서 `brew`를 설치한다. 
(2021년 2월에 m1 맥북용으로 업데이트했다고 한다, 에러가 난다면 [link](https://gist.github.com/nrubin29/bea5aa83e8dfa91370fe83b62dad6dfa) 참조)

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

설치하고 path 설정

```
echo 'export PATH=/opt/homebrew/bin:$PATH' >> ~/.zshrc

source ~/.zshrc
```

그리고 확인

```
echo $PATH

brew --version
```

아예 세팅 처음한다고 생각하고 이것저것 설치해준다.

```
brew install wget

brew install cmake

brew install gcc

brew install libffi

...
```

M1 맥의 경우 파이썬 버전 2.7이 디폴트일테니 알아서 설치해준다, 그리고 체크

```
brew install python

python3 --version
```


## Virtual Environment

### Docker installation

[Docker for apple silicon chip](https://docs.docker.com/docker-for-mac/apple-silicon/)에서 파일을 다운받거나 아래의 command line을 입력하면 된다.

```
brew install docker 
```

그리고 설치됐나 확인

```
docker --version
```

![docker](/assets/images/m1_mac/docker.png)
*Fig. docker의 기본적인 flow, 이미지 출처 : [link](http://moducon.kr/2018/wp-content/uploads/sites/2/2018/12/leesangsoo_slide.pdf)*

### Anaconda (miniconda) installation 

혹은 anaconda를 설치해서 환경을 관리하며 버전컨트롤을 해도 된다. 아래의 명령어를 이용해 macos에 맞는 miniforge를 간단하게 다운받을 수 있다.

```
brew install miniforge

conda --version
```

를 해보니까 안된다. 다시 지우고,

```
brew uninstall miniforge
```

[Miniforge](https://github.com/conda-forge/miniforge#download) 에서 apple silicon chip을 위한 mimiforge를 다운받아야 한다.

![miniforge](/assets/images/m1_mac/miniforge.png)
*Fig. [Miniforge](https://github.com/conda-forge/miniforge#download) 페이지에 명시되어 있음*

다운 받은 폴더로 찾아가서 (보통 `~/Downloads`니까 그리로 가면 된다) shell을 실행해준다.

```
bash Miniforge3-MacOSX-arm64.sh
```

## Torch or Tensorflow installation

docker를 사용하려면 [docker hub](https://hub.docker.com/)에서 원하는 docker image를 pull 하면 되는데, 
현재 apple chip을 위한 마땅한 docker image가 없는 것 같아서 conda를 쓰는게 나을 것 같다.


`tensorflow`를 사용하고 싶다면 [Mac-optimized TensorFlow and TensorFlow Addons](https://github.com/apple/tensorflow_macos)에서 공식 가이드를 따라 진행하면 될 것 같고 `torch`를 사용하고 싶다면 [Guide: ARM64 Mac Native Deep Learning Set Up](https://github.com/oresttokovenko/Guide-ARM64-Mac-Native-Deep-Learning-Set-Up)를 참고하면 될 것 같다.


`troch`를 쓰는 경우에 대해서만 간단하게 얘기하자면


3.8 파이썬으로 가상 환경을 만들어주고

```
conda create -n m1_pytorch python=3.8
```

환경으로 접속해주고

```
. source m1_pytorch
```

라이브러리를 깔아주면 된다.

```
conda install pytorch -c isuruf/label/pytorch -c conda-forge
conda install torchvision -c pytorch
```

환경에서 나가지 말고 잘 설치되었는지 체크하면 된다.

```
python

import torch
torch.__version__
```


## Run NN model

제대로 돌려보려고 했으나... 속도를 보고 바로 접었다. 파이토치는 당연히 엄청 느리고 m1용 tf를 사용해도 엄청 느리다.

자세한건 다음의 블로그에서 직접 비교한 벤치마크를 참고하면 될 것 같다.[Benchmark M1 vs Xeon vs Core i5 vs K80 and T4](https://towardsdatascience.com/benchmark-m1-vs-xeon-vs-core-i5-vs-k80-and-t4-e3802f27421c)


## 결론

혹시나 해서 해봤지만 역시나 안된다. 그냥 colab이나 aws같은 클라우드 서비스를 쓰는 것 말고는 답이 없는 것 같다... 혹은 더 최적화된 tf나 torch를 내주길 기다려야 할 것 같다. 그런 날이 올지는 모르겠지만...


![torch_thread1](/assets/images/m1_mac/torch_thread1.png)
![torch_thread2](/assets/images/m1_mac/torch_thread2.png)
*Fig. 21년 5월 27일 기준 m1칩을 위한 torch의 업데이트 소식은 없냐는 스레드에 수 많은 답글이 달렸으나, 아무 일도 없었다고 한다...*


## Reference
- m1 brew
  - [m1 mac homebrew 설치하기](https://shanepark.tistory.com/m/45?category=1182535)
  - [m1 mac homebrew 설치하기 2](https://cpuu.postype.com/post/9183991)
- m1 docker
  - [Docker Desktop for Apple silicon](https://docs.docker.com/docker-for-mac/apple-silicon/)
  - additional
    - [북리뷰, 도커: 설치에서 운영까지](https://cpuu.postype.com/post/2948749)
    - [AI 딥러닝 시작:: 가상 환경 구현 ( Docker )](https://velog.io/@uonmf97/AI-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%8B%9C%EC%9E%91-%EA%B0%80%EC%83%81-%ED%99%98%EA%B2%BD-%EA%B5%AC%ED%98%84-Docker)
- m1 pytorch, tensorflow
  - [Mac-optimized TensorFlow and TensorFlow Addons](https://github.com/apple/tensorflow_macos)
  - [Install PyTorch natively on Mac M1](https://github.com/edadaltocg/install-pytorch-m1)
  - [Setting up M1 Mac for both TensorFlow and PyTorch](https://naturale0.github.io/machine%20learning/setting-up-m1-mac-for-both-tensorflow-and-pytorch)
  - [Guide: ARM64 Mac Native Deep Learning Set Up](https://github.com/oresttokovenko/Guide-ARM64-Mac-Native-Deep-Learning-Set-Up)
  - [Apple Silicon M1 macOS에서 TensorFlow 활용기](https://cpuu.postype.com/post/9091007)
  - [Benchmark M1 vs Xeon vs Core i5 vs K80 and T4](https://towardsdatascience.com/benchmark-m1-vs-xeon-vs-core-i5-vs-k80-and-t4-e3802f27421c)
