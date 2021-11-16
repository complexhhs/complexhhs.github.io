---
layout: post
title: "[GP]Gaussian Process _ Kernel"
author: "Hyunseok, Hwang"
categories: journal
tags: [documentation,sample]
image: cards.jpg
---

이번 포스팅은 베이지안 방식을 기반으로한 회귀분석 풀이법 중 하나인 **Gaussian Process**을 다룬다. 
먼저 회귀분석(regression)을 많이 접해본 이들은 통계적인 추론방식등을 이용해서 아래와 같은 문제를 많이 접해봤으리라 믿는다. 

![linear_regression](/assets/img/linear_regression.png)

위 그림에서 데이터인 푸른점들을 선형식, $y=ax+b$ 형태로 가정하고 데이터 $(x,y)$값을 이용해서 $a$와 $b$인 계수를 구하는 공식에 대입하면 끝인 문제이다.
아마 대학교 학부수준에서는 기초통계 수업을 들으면 최소한 한번씩은 접해봤을 문제였을 것이고 위와 같은 문제를 해결하는데 큰 거부감이 없을것이다.

자, 위 회귀분석 문제의 핵심은 '1. 선형방정식을 따른다'와 '2. 선형방정식을 이루는 계수를 구하면 문제해결'이라는 것이다. 이와 같은 문제해법은 ***Parameteric Method***의 전형적인 방식이다.
하지만 우리는 이 단순한 형태의 문제만이 아닌, 더 어렵고 복잡하고 실전에 맞는 문제를 해결할 능력이 필요하다. 변수가 많아지고, 데이터의 형태가 '선형'이 아니고, 방정식을 이루는 계수가 많아지고 구하기가 까다로워 지면 ***Parametric Method***보다는 ***Non-Parametric Method***가 더욱 효과적인 방식으로 해결전략을 세우는 것이 바람직하다.
바로 **Gaussian Process**가 ***Non-Parametric Method*** 회귀분석의 대표예시가 된다.
**Gaussian Process**는 대부분의 베이지안 방식이 그렇듯 많은 수학개념이 필요하기 때문에 하나하나 필요한 배경지식에 대해서 소제목으로 나누어서 다루어 보겠다.

## Gaussian Distribution

자연계에서 가장 많이 나타나는 확률분포표이다. 돈이 많은 사람의 비율-돈이 많은사람의 비율, 공부를 못하는사람-잘하는 사람의 비율 등등 무엇인가 차이가 나는 표본들이 분포한 그림으로써 특이한 사람들은 소수고 중간에 있는 표본들이 많은 그림을 보여주는 그림으로 많이 알고있는 그림이다. 그런데 ***Gaussian Process***를 알기 위해서는 분포표의 수식과 성질을 어느정도 숙지할 필요가 있다.

- 1D Gaussian distribution
![1D normal distribution](/assets/img/1d normal distiribution.png)
$$   \mathcal{N}({x} | \mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^{2}}}exp\left(-\frac{(x-\mu)^2}{2\sigma^2}   \right)  $$. (1) 

미리 말하지만, 1D와 2D gaussian distiribution 그림은 필자가 직접그린 것이 아닌 구글에서 이미지를 그냥 가지고 왔다. 1D상황의 분포는 분포의 평균($\mu$)에서 가장 위로 볼록하게 나타나며 분산값($\sigma$)이 커질수록 마치 반죽이 펼쳐지듯 퍼지는 형태의 그림이 된다. 분산값이 크다는 이야기는 표본 각각이 평균으로 부터 멀리 떨어져 있음을 보이므로 표본 집단의 성질을 평균으로 정의내리기 어렵다는 의미가 된다. 모평균추정을 하부 섹션으로 만들어서 이야기 해보겠다. 돌아와서 평균과 분산값은 1D 상황이므로 더 말할필요 없이 스칼라 형태의 텐서다.

- Multi-Dimension Gaussian distribution
![2D normal distribution](/assets/img/2d normal distiribution.png)
$$  \mathcal{N}({X} | \mu,\Sigma) = \frac{1}{{2\pi}^{d/2}}|\Sigma|^{-1/2}exp\left(-\frac{1}{2}(X-\mu)|\Sigma|^{-1}(X-\mu)^T   \right) $$. (2)

다차원의 gaussian distribution은 조금 생소할수 있다. 무엇이든 차원을 확장하는 첫 단계가 생소하고 어렵듯 1D에서 2D로 넘어가는 과정도 조금 어색할지도 모르겠다. 함수 형태가 볼록하고 펴지고 하는 모습들은 1D에서의 형태와 크게 다를바가 없다. 다만, 변수 $X$와, 평균 $\mu$는 여기에서 $d$차원의 벡터이며, 분산(정확히 공분산행렬) $\Sigma$은 $d \times d$행렬이 된다. 공분산 행렬은 입력변수의 차원간의 상관관계를 표현한 분산으로 자세한 개념은 [Andrew Ng 교수님의 강의](https://www.youtube.com/watch?v=JjB58InuTqM)를 참조하면 좋겠다.


## Inference about a Population Mean

다음 살펴봐야 할 부분은 모평균의 추정개념이다. 알고있다면 과감히 패스하시고 아니면, 고등학교 교과과정에서 다루는 내용이지만 개념을 잊어버렸을 독자들을 위하여 간단하게 요약하고 넘어가는 부분이니 감안하기를 바란다. 
1D gaussian distribution의 식 (1)에서 평균과, 분산값을 알고 있다면 우리는 샘플의 분포를 그릴 수 있다. 하지만 역으로 일부의 샘플을 통해서 역으로 모집단의 평균,분산을 어느정도 예측할수 있지 않을까하는 의문을 삼을 수 있다. 선거철에서 흔히 뉴스에서 나오는 '표본 몇명을 대상으로 95퍼센트의 신뢰도로 어느 후보가 앞선다'는 식의 조사가 바로 모평균의 추정이다. 우선 Gaussian distribution의 Probability Density Function(PDF)값들을 적분한 값은 1이라는 것을 인지하고, 평균으로 부터 1$\sigma$, 2$\sigma$ 등의 값으로 좌우로 샘플되는 구간들이 있다. 특별한 숫자지만 $\sigma$의 1.96배수와, 2.58배수로 좌우로 펼쳐진 구간의 적분면적은 각각 0.95, 0.99값이 나온다(95%, 99%). 글로 해석해보면 샘플 변수 $x$가 평균으로부터 $-1.96\sigma \sim +1.96\sigma$의 구간으로 있다면 전체 샘플링 영역의 95%구간안에 속해있다고 표현하는 것이다. 

$$P\left(-1.96 \leq z \leq 1.96\right)=\frac{95}{100} $$. (3)

위 문단에서 썼던 $\sigma$의 1.96배수로 구한 95%신뢰 구간의 의미를 식(3)으로 표현하였다. 여기서 표준정규분포(평균이 0이고 분산이 1인 **gaussian distribution**)의 변수 $Z$가 나왔는데 이는 우리가 식(1)에서 사용했던 변수 $X$와는 다음과 같은 관계를 가지고 있다.

$$z = \frac{x-\mu}{\sigma/\sqrt{n}}$$. (4)

그러므로, 식(4)를 $x$에 대해서 풀어보면 아래와 같다.

$$x = \mu+z*\frac{\sigma}{\sqrt{n}}$$. (5)

따라서, 샘플링 변수 $x$가 $n$개수 만큼 샘플링되었을때 95%의 신뢰도로 어떤 값($\mu$)을 추정한다는 의미는 아래 수식으로 표현된다.

$$P\left(\mu-1.96\frac{\sigma}{\sqrt{n}} \leq x \leq \mu+1.96\frac{\sigma}{\sqrt{n}} \right) = 0.95 $$.(6)

이제 모평균의 추정을 살펴보았으니 다시 ***Gaussian Process***에 왜 모평균의 추정이 들어갔는지 그림과 함께 보도록 하겠다.

![Gaussian Process](/assets/img/GP_regression.png)

일반적으로 ***Gaussian Process***가 지금까지 살펴본 모평균의 추정 과정과 맥락이 동일하다. 추론하고자 하는 함수값이 곧 모평균의 추정으로 얻은 $\mu$값이며, 추정의 신뢰도를 구할수 있는 $\sigma$값도 계산을 통해서 얻을수 있게 된다. 위 그림에서 데이터 몇 개를 이용해 GP를 계산하면 함수값과 그에 대한 추정 신뢰구간이 범위형태로 그려지는 것을 볼 수 있다. 참고로 위 그림은 함수값이 정답안에 있을 확률이 95%의 신뢰구간이라고 ML이 표현하는 것을 의미한다. 이제 ***Gaussian Process***의 추정과 신뢰구간이라는 개념을 살펴보았으니 이 둘을 어떻게 계산하는지 보다 면밀하게 짚고 넘어 가도록 하겠다. 다음 문단부터 필수 개념인 Kernel에 대해서 살펴보도록 하겠다.

## Kernel

Kernel이 하는 역할을 먼저 알아보도록 하겠다.

- Kernel 1기능: Mapping
수학에서 어떤 문제를 해결할 때 상변환이라는 개념을 도입한다. 직교좌표계에서 원형좌표계로 변형시킨다던가 하는 방법이다. 아래 그림을 보자. 그림처럼 구간을 분리시키고자 하지만 단칼로 한번에 구간을 분리시키기가 굉장히 어려운 문제다. 하지만 우리는 Mapping, 상변환의 개념을 통하여 데이터들을 간단하게 분리시킬 방법을 고안할수 있다. 2D공간에서만 문제를 생각하지말고 3D공간으로 바꾸어 문제를 풀어볼수 있다. Kernel 내부에는 mapping함수가 존재해서 그림과 같이 데이터가 복잡하게 분포하고 있더라도 우리가 선택한 바대로 데이터 상변환을 통해 풀기 쉽게 바꿔 버릴 수 있다. 
    
- Kernel 2기능: Inner product
첫번째의 Mapping만 한다면 Kernel은 반쪽짜리 역할밖에 하지 못한다. kernel은 두개의 벡터를 입력변수로 삼고 둘을 Inner product연산을 통해서 스칼라 값으로 변화하여 다른 차원의 개념으로 변화를 한다. 수식으로 표현하면 다음과 같은 모양이 된다.

$$ K(x_1, x_2) = <\phi(x_1), \phi(x_2)> $$
$$ \: \; \; {\quad} = \phi(x_1)^T\phi(x_2) $$. (7)

즉, Kernel의 기능을 풀어쓰면 입력받은 두 변수를 1) 복잡하게 얽혀진 데이터를 풀기쉽도록 상 변환을 하고, 2) Inner product를 취해서 단일 스칼라 값으로 변환을 한다. 그래서 입력받은 두 변수를 추상적인 어떠한 스칼라로 변환을 하는데 간략히 표현하면 두 입력 변수간의 유사도(Similarity)를 측정하는데 주로 이용된다. 
각론으로 살펴보자면 Kernel함수로 이용되는 형태는 RBF, Step function, Linear function들로 구성되는데 ***Gaussian Process***에서는 주로 RBF kernel이 주로 이용된다. 식 (7)을 골자로 한 RBF식은 다음과 같다.

$$ K_{RBF}(x_1, x_2) = \sigma^2 exp\left(-\frac{\left\Vert x_1-x_2 \right\Vert^2}{2l^2} \right) $$. (8)

여기서 $x_1$과 $x_2$간의 similarity($\left\Vert x_1-x_2 \right\Vert^2$)를 x축으로 삼고, Kernel값을 y축으로 삼으면 개형은 대략 다음과 같이 나온다.

![RBF Kernel개형](/assets/img/RBF_graph.png)

위 그림에서 $\sigma$값을 증가시키면 기준으로 삼은 RBF개형보다 $x=0$에서 더 위로 볼록한 형태의 그래프가 나오며, $l$값을 증가시키면 기준으로 삼은 RBF개형보다 더 완만한 그래프 개형이 나오게 된다. 두 그래프를 비교해서 해석하면 Kernel의 parameter인 $\sigma, l$을 어떻게 설정하는지에 따라 입력변수 $x_1, x_2$간 similarity를 강하게 판단할 것인지, 약하게 판단할것인지 척도를 kernel의 값으로 알겠다는 지표가 된다. $\sigma$값을 증가하면 입력변수 $x_1, x_2$간 similarity를 크게 파악하겠다는 의도를 반영하며, $l$값이 증가하면 입력변수 $x_1, x_2$간 similarity가 낮아도 상호간의 관계를 많이 보겠다는 의도를 반영하는 바가 된다. 

나름 설명한다고 했는데 왜 갑자기 ***Gaussian Process***를 설명하는데 이해하기 어려운 Kernel은 왜나오는 것인지 이쯤 되면 의문을 가지게 될 것이다. 
