이번 포스팅은 지난 포스팅의 MCMC_Metropolis-Hasting 방식에 이은 베이지안 통계방식에 근거한 ML의 두번째 방식인 Gaussian Process에 대해 다뤄보겠다.
<br/><br/>
흔히 선거철에 많이 하는 통계조사의 발표에서 주로 나오는 말이 있다. 
<br/><br/> "이 결과는 1000명의 표본으로 95%의 신뢰도로 추정됩니다." 독자들이 고교 수학과정에서 모집단의 평균추정 개념이다. 원리로만 따지면 우리가 어떤 값을 추정할때 추정대상의 분포가 Gaussian distribution을 따른 다는 전제하에 이러쿵 저러쿵 계산을 했었던 것으로 기억한다. Gaussian Process는 주로 회귀문제를 해결할때 자주 쓰이는 방식으로, 원하는 함수값 $f(x_{target})$을 Gaussian distribution을 따른다는 전제(Prior)하에 '모집단의 평균 추정' 비슷한 연산으로 얻는 과정이다.

<br/><br/> 
### Gaussian distribution
<br/> 본격적인 Gaussian Process에 대해 이야기 앞서 사전지식인 Gaussian distribution의 식을 멋들어지게 음미해보자.
<br/><br/>
$$\mathcal{N}(x; \mu,\sigma)={{{1}\over{\sqrt{2\pi{\sigma}^2}}}exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)} $$(1)
<br/><br/>
$$\mathcal{N}(X; M,\Sigma)={{{1}\over{{\left(2{\pi}{|\Sigma|})\right)}}^{\frac{n}{2}}}exp\left(-\frac{1}{2}(X-M)^{T}{\Sigma}^{-1}(X-M)    \right)} $$(2)
<br/>
단일 변수에 대한 Gaussian distribution은 식 (1)처럼 작성되며, 변수($x$), 평균($\mu$), 그리고 분산($\sigma$)은 당연히 스칼라 값이다. 반면에, 다 변수에 대한 Gaussian distribution은 식 (2)처럼 작성되며, 변수($X$), 평균($M$), 그리고 분산($\Sigma$)는 변수의 개수만큼의 벡터($vector$), 혹은 행렬($matrix$)로 표현된다. 
<br/><br/>
필자가 이해했을때 다 변수의 Gaussian distribution에 대해서 가장 껄끄러웠었던 부분이 분산, 엄밀히 공분산(Covariance)라는 개념이었는데 두 변수간의 분산관계를 행렬로 표현한 것으로 말보다는 수식으로 써보자.
<br/>
$$ \Sigma = \begin{bmatrix} K(X_1,X_2) & K(X_1,X_2)  \\
                            K(X_2,X_1) & K(X_2,X_2)  \end{bmatrix}$$(3)
                             
<br/><br/> 다 변수의 개수가 2라고 했을때 공분산행렬을 위와 같이 작성된다. $cov$ 함수는 입력인자($X_1, X_2$)간의 유사도를 표현하는 함수로 입력인자끼리 유사하면 유사도를 높게주고, 반대로 유사도가 낮을 경우 유사도가 낮은 결과를 출력하게 될것이다. 그렇다면 다 변수의 Gaussian distribution은 변수 벡터의 크기가 $\mathbb{R}^n$이 라면, 공분산 행렬은 $\mathbb{R}^{n\times n}$의 크기를 가지게 된다. 

<br/><br/> 우선 여기까지 Gaussian distribution 자체 공식의 개념을 보고 우리가 실제로 풀고싶어하는 문제를 살펴보도록 하자.
<br/> 위 그림은 기본적인 $sin$함수 곡선을 토대로 noise값들이 첨가된 데이터 셋들을 나타낸 것이다. 우리가 이 데이터셋을 Polynomial regression으로 표현한다면 대략 다음과 같이 쓸수 있을 것이다. 
<br/><br/>
$$ \begin{aligned}y &= a_0 + a_1x+a_2x^2+a_3x^3 ... \\
    &= \sum_{i=0}^{\infty}a_ix^i     \end{aligned} $$(4)
<br/><br/> 식 4번은 Polynomial regression으로 표현한 다항함수로 근사시킨 noise가 첨가된 $sin$함수의 식이 되겠다.각 다항함수 앞에 붙여진 계수들이 우리들이 푸는 문제의 핵심이 되는 파라미터들이며 이를 이용하여 문제를 푸는 방식을 Parametric method라고 일컫는다. 허나 Gaussian Process는 이런 파라미터들로 식을 고정적으로 세팅하는 방식이 아닌 앞서도 말한 Gaussian distribution을 살짝 변형시켜 우리가 원하는 함수값을 얻는 방식이다. 
<br/> 그렇다면 Gaussian Process의 기본전제와 문제 세팅 방식을 정리해본다.
<br/>

* Condition 1
<br/> 모든 함수 값은 Gaussian distribution을 따른다 $\leftarrow$ Prior knowledge

* Condition 2
<br/> 주어진 데이터 하나가 각각 단일 변수로 취급된다.

* Condition 3
<br/> Prior knowledge의 Gaussian distribution의 평균값은 0이다. 

* Condition 4
<br/> Prior knowledge의 Gaussian distribution의 covariance matrix를 구성하는 *kernel function*은 user의 뜻대로 조절가능하다. 
 
<br/><br/> 약 4가지의 조건을 들었는데 지금까지 이해하기 난해한 용어가 하나가 나왔을 것이다. 4번째 조건에서 *kernel function* 이라는 용어가 처음 등장했는데 위에서 말을 할 수는 있었지만 *Kernel*의 개념은 Gaussian process이외에도 많은 머신러닝 연구에서도 활발히 이용되는 부분이므로 따로 개념을 정리하겠다. '
<br/>복잡하다고 생각하면 Kernel이란 개념은 **Gaussian Process에서 두 입력변수의 유사도를 측정하는 함수다** 정도로만 이해하고 넘어가도 무방하겠다.
<br/><br/>
## Kernel
<br/> 딥러닝을 왜 해야하는지 당위성을 배울때 처음 맞닥뜨리는 예시로 XOR게이트를 생각 할 수 있다. XOR게이트를 형성하기 위해서 단일 논리게이트로는 불가능 하기 때문에 최소 2개의 논리게이트를 겹쳐서 만드는 방법이다. 이는 달리 표현하자면, 복잡한 문제를 한번에 풀 수가 없으니까 모습을 한번 바꾸어(mapping) 풀기 쉬운 문제로 바꾸어 원하는 결과를 얻는 전략이다. 
<br/> Kernel의 목적중 하나인 Mapping의 성질이 이를 가지고 있다. Covariance matrix의 원소는 입력받은 두 확률 변수간의 유사도를 측정해야하는데 말이 유사도라고 이야기 하는 것이지 구체적으로 어떤 방식이로든가 식을 변형해야하는 목적을 달성시키는것이 Kernel의 역할이다.
<br/> 남은 목적은 내적공간(inner product space)으로의 연산이다. 두 입력변수의 유사도를 측정한다고 하면 두 입력변수가 입력되는 순서가 다를지라도 동일한 결과를 나타내야 하는 것이며, 그 유사도라고 하는 값은 스칼라로 출력되야 하는 것이다. 이를 만족하는 연산이 Inner product이며 Kernel은 이를 만족하고 있다.

<br/><br/> 종합하자면 Kernel은 어려운 함수를 쉽게 풀어주는 Mapping의 기능($1$)과 입력순서에 대한 무관한 결과와 출력값을 스칼라로 출력하게 해주는 내적연산의 기능($2$)를 다 만족하는 함수라고 말할수 있다. 식으로 표현하면 아래와 같다.
<br/>
$$\begin{aligned} kernel &= K(x,y) \\
    &= <\phi(x),\phi(y)>   \\
    &= \phi(x)^T\phi(y)     \end{aligned}  $$(5)
<br/> 아마 용어들을 처음 들어본 독자들은 지금쯤 되면 머리가 조금 아프리라 생각한다. 예시를 통해 접근해보도록 해보겠다. 주로 Gaussian Process의 Kernel function으로 주로 쓰이는 R.B.F, 풀어서 Radial Basis Function을 보자.
<br/><br/>
$$ R.B.F = \sigma^2exp\left(\frac{|x-y|^2}{l^2} \right)$$(6)
<br/> R.B.F의 개형은 $x$축을 input변수간의 거리, $y$축을 유사도 값으로 보면 아주 심플하게 그려진다.

<br/> Gaussian distribution식과 똑같으므로 $x$축의 오른쪽으로 갈수록 유사도 값은 부드럽게 떨어지는 형세가 나온다. 한번 더 음미해보자, 위 **조건 4**에서 Kernel의 형태는 user가 조절 할 수 있는 부분이므로 그래프의 개형도 원하는 대로 조절이 가능하다. 식 (6)에서 우리가 조절 가능한 파라미터는 $\sigma, l$, 두 값으로 $\sigma$값의 역할은 전체적으로 입력 데이터간의 전체적인 유사도를 높여주거나 낮춰주는 역할을 하게된다. 해당 값을 높여주게 된다면 특정 데이터와 멀리 떨어져 있는 데이터가 서로가 서로의 영향이 강하게 역할되므로 Gaussian Process연산을 할때 데이터의 의존성이 굉장히 강하게 작용될것이다. 반대로 낮추면 입력 데이터간 서로의 의존성이 약하게 작용된다. 이제 $l$값의 역할은 R.B.F의 개형을 변화시키는 결과를 가져오는데, 값이 커지면 R.B.F값이 떨어지는 정도가 더 완만하게 떨어진다. 이는 곧, 특정데이터와 거리가 멀리떨어져 있는 값일 지라도 유사도가 높게 판단하고 추론을 진행하게 되는 것이다. 
<br/> R.B.F자체만으로도 우리가 Kernel을 어떻게 변화 시킬지 옵션들이 다양한데, Linear, Polynomial, Step function 등 변형 시킬 수 있는 함수의 종류가 무궁무진하게 많다.

<br/><br/><br/>
<br/><br/> 지금까지 쓰는데 포스팅이 굉장히 길어지는 것 같다. 나머지 Gaussian Process의 이야기와 예시 적용방법등은 다음 포스팅으로 미루도록 하겠다.
