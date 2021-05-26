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

원래는 M1 Macbook을 샀을 당시 많은 라이브러리에서 에러가 났고, 특히 딥러닝 모델을 학습 할 수 있다는 말을 제대로 못 들은 것 같아서 일단 냅뒀는데,
된다는 말을 듣고 설치해보려고한다.


순서는 

- brew 설치
- Docker or conda 설치
- torch 등 라이브러리 설치 
- 간단한 네트워크 러닝 해보기

가 될 것 같다.




### Homebrew 설치

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


### Docker installation

[Docker for apple silicon chip](https://docs.docker.com/docker-for-mac/apple-silicon/)에서 파일을 다운받거나 아래의 command line을 입력하면 된다.

```
brew install docker 
```

그리고 설치됐나 확인

```
docker --version
```

혹은 anaconda를 설치해서 환경을 관리하며 버전컨트롤을 해도 된다. 그럴경우 아래의 명령어

```
brew install miniforge
```

그리고 확인

```
conda --version
```

### Torch or Tensorflow installation



### Run NN model



## Reference

1. [국내 blog 1](https://shanepark.tistory.com/m/45?category=1182535)

2. [국내 blog 2](https://cpuu.postype.com/post/9183991)
