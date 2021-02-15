---
title: (미완)Regression (3/3) - Non-linear regression, Kernelization and Gaussian processes

categories: MachineLearning
tag: [MachineLearning,ML]

toc: true
toc_sticky: true

comments: true
---

이전까지 우리는 선형 회귀에 대해서 MLE, MAP 등에 대해 알아봤습니다. ML, MAP로 해를 구하는 것이 약간의 차이가 있지만, 
결국 이들은 우리가 구하고자 하는 파라메터와 입력 데이터간 선형 결합되어 있는 관계식에서 파라메터를 추정하는 것이라는 점은 같았습니다.
하지만 이러한 방식은 한계가 존재합니다. 가령 우리가 가진 데이터를 도저히 직선(Linear)으로 표현하기 힘들다면 어떨까요? (여기서 Linear는 매개변수가 아닌 독립변수인 경우를 말합니다.)
곡선(Non-Linear)으로 만들어야 하지 않을까요??

우리가 이번에 알아보게될 내용은 구하고자 하는 파라메터들 간의 선형 결합은 유지하면서 비선형의 함수를 통해서 non-linearity를 추가해 회귀 곡선의 표현력을 높혀보자는 내용이 될 것입니다.

---
< 목차 >
{: class="table-of-content"}
* TOC
{:toc}
---

## <mark style='background-color: #fff5b1'> Non-Linear Regression </mark>

![reg1](https://user-images.githubusercontent.com/48202736/107945467-03f36f00-6fd3-11eb-86ec-1a68cda77511.png)

위의 그림을 보시면 우리가 이번에 다루게 될 내용에 대해서 감이 오실 것 같습니다.

> 1. 비선형성을 추가해서 좀더 표현력을 높힌 회귀 함수 <br>
> 2. 베이지안 방법론과 비선형성 두 가지를 결합한 강력한 회귀 함수 <br>



## <mark style='background-color: #fff5b1'> Basis Functions </mark>

### <mark style='background-color: #dcffe4'> Radial Basis Functions </mark>

![reg2](https://user-images.githubusercontent.com/48202736/107945481-081f8c80-6fd3-11eb-94c4-71fdea34641d.png)

### <mark style='background-color: #dcffe4'> Arc Tan Functions </mark>

![reg3](https://user-images.githubusercontent.com/48202736/107945484-08b82300-6fd3-11eb-9229-944ad2186d69.png)





## <mark style='background-color: #fff5b1'> ML Solution for Non-Linear Regression </mark>






## <mark style='background-color: #fff5b1'> Bayesian Approach </mark>

![reg4](https://user-images.githubusercontent.com/48202736/107945486-0950b980-6fd3-11eb-917c-87da25117dd2.png)




## <mark style='background-color: #fff5b1'> Kernel Trick </mark>

### <mark style='background-color: #dcffe4'> Gaussian Proccess Regression </mark>

### <mark style='background-color: #dcffe4'> Example : RBF Kernel  </mark>

![reg5](https://user-images.githubusercontent.com/48202736/107945487-09e95000-6fd3-11eb-8087-6a7b8363506b.png)






## <mark style='background-color: #fff5b1'> References </mark>

1. [Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.](http://www.computervisionmodels.com/)

2. [Gal, Yarin. "Uncertainty in deep learning." University of Cambridge 1, no. 3 (2016): 4.](https://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html)

