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

### Gaussian Distribution
자연계에서 가장 많이 나타나는 확률분포표이다. 돈이 많은 사람의 비율-돈이 많은사람의 비율, 공부를 못하는사람-잘하는 사람의 비율 등등 무엇인가 차이가 나는 표본들이 분포한 그림으로써 특이한 사람들은 소수고 중간에 있는 표본들이 많은 그림을 보여주는 그림으로 많이 알고있는 그림이다. 그런데 ***Gaussian Process***를 알기 위해서는 분포표의 수식과 성질을 어느정도 숙지할 필요가 있다.

- 1D Gaussian distribution
![1D normal distribution](/assets/img/1d normal distiribution.png)
$$$   \mathcal{N}({x} | \mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^{2}}}exp\left(-\frac{(x-\mu)^2}{2\sigma^2}   \right)  $$$. (1) 

미리 말하지만, 1D와 2D gaussian distiribution 그림은 필자가 직접그린 것이 아닌 구글에서 이미지를 그냥 가지고 왔다. 1D상황의 분포는 분포의 평균($\mu$)에서 가장 위로 볼록하게 나타나며 분산값($\sigma$)이 커질수록 마치 반죽이 펼쳐지듯 퍼지는 형태의 그림이 된다. 분산값이 크다는 이야기는 표본 각각이 평균으로 부터 멀리 떨어져 있음을 보이므로 표본 집단의 성질을 평균으로 정의내리기 어렵다는 의미가 된다. 모평균추정을 하부 섹션으로 만들어서 이야기 해보겠다. 돌아와서 평균과 분산값은 1D 상황이므로 더 말할필요 없이 스칼라 형태의 텐서다.

- Multi-Dimension Gaussian distribution
![2D normal distribution](/assets/img/2d normal distiribution.png)
$$$  \mathcal{N}({X} | \mu,\Sigma) = \frac{1}{{2\pi}^{d/2}}|\Sigma|^{-1/2}exp\left(-\frac{1}{2}(X-\mu)|\Sigma|^{-1}(X-\mu)^T   \right) $$$. (2)

다차원의 gaussian distribution은 조금 생소할수 있다. 무엇이든 차원을 확장하는 첫 단계가 생소하고 어렵듯 1D에서 2D로 넘어가는 과정도 조금 어색할지도 모르겠다. 함수 형태가 볼록하고 펴지고 하는 모습들은 1D에서의 형태와 크게 다를바가 없다. 다만, 변수 $X$와, 평균 $\mu$는 여기에서 $d$차원의 벡터이며, 분산(정확히 공분산행렬) $\Sigma$은 $d \times d$행렬이 된다. 공분산 행렬은 입력변수의 차원간의 상관관계를 표현한 분산으로 자세한 개념은 [Andrew Ng 교수님의 강의](https://www.youtube.com/watch?v=JjB58InuTqM)를 참조하면 좋겠다.

다음 살펴봐야 할 부분은 모평균의 추정개념이다. 고등학교 교과과정에서 다루는 내용이지만 개념을 잊어버렸을 독자들을 위하여 간단하게 요약하고 넘어가겠다. 
1D gaussian distribution의 식 (1)에서 평균과, 분산값을 알고 있다면 우리는 샘플의 분포를 그릴 수 있다. 하지만 역으로 일부의 샘플을 통해서 역으로 모집단의 평균,분산을 어느정도 예측할수 있지 않을까하는 의문을 삼을 수 있다. 선거철에서 흔히 뉴스에서 나오는 '표본 몇명을 대상으로 95퍼센트의 신뢰도로 어느 후보가 앞선다'는 식의 조사가 바로 모평균의 추정이다. 우선 Gaussian distribution의 Probability Density Function(PDF)값들을 적분한 값은 1이라는 것을 인지하고, 평균으로 부터 1$\sigma$, 2$\sigma$ 등의 값으로 좌우로 샘플되는 구간들이 있다. 특별한 숫자지만 $\sigma$의 1.96배수와, 2.58배수로 좌우로 펼쳐진 구간의 적분면적은 각각 0.95, 0.99값이 나온다(95%, 99%). 글로 해석해보면 샘플 변수 $x$가 평균으로부터 $-1.96\sigma \sim +1.96\sigma$의 구간으로 있다면 전체 샘플링 영역의 95%구간안에 속해있다고 표현하는 것이다. 
