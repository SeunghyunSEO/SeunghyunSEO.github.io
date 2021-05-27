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

아예 세팅 처음한다고 생각하고 `wget`도 깔아주고 한다.

```
brew install wget
```

M1 맥의 경우 파이썬 버전 2.7이 디폴트일테니 알아서 설치해준다, 그리고 체크

```
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

혹은 anaconda를 설치해서 환경을 관리하며 버전컨트롤을 해도 된다. 그럴경우 아래의 명령어

```
brew install miniforge
```

그리고 확인

```
conda --version
```


## Torch or Tensorflow installation

[docker hub](https://hub.docker.com/)에서 원하는 docker image를 pull 하면 되는데, 
현재 apple chip을 위한 마땅한 docker image가 없는 것 같아서 conda를 쓰는게 나을 것 같다.


텐서플로우를 사용하고 싶다면 [Mac-optimized TensorFlow and TensorFlow Addons](https://github.com/apple/tensorflow_macos)에서 공식 가이드를 따라 진행하면 될 것 같고


## Run NN model



## Reference
- m1 brew
  - [m1 mac homebrew 설치하기](https://shanepark.tistory.com/m/45?category=1182535)
  - [m1 mac homebrew 설치하기 2](https://cpuu.postype.com/post/9183991)
- m1 docker
  - [북리뷰, 도커: 설치에서 운영까지](https://cpuu.postype.com/post/2948749)
- m1 pytorch, tensorflow
  - [Mac-optimized TensorFlow and TensorFlow Addons](https://github.com/apple/tensorflow_macos)
  - [Install PyTorch natively on Mac M1](https://github.com/edadaltocg/install-pytorch-m1)
  - [Setting up M1 Mac for both TensorFlow and PyTorch](https://naturale0.github.io/machine%20learning/setting-up-m1-mac-for-both-tensorflow-and-pytorch)
  - [Guide: ARM64 Mac Native Deep Learning Set Up](https://github.com/oresttokovenko/Guide-ARM64-Mac-Native-Deep-Learning-Set-Up)
