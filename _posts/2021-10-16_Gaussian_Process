이번 포스팅은 지난 포스팅의 MCMC_Metropolis-Hasting 방식에 이은 베이지안 통계방식에 근거한 ML의 두번째 방식인 Gaussian Process에 대해 다뤄보겠다.
<br/><br/>
흔히 선거철에 많이 하는 통계조사의 발표에서 주로 나오는 말이 있다. 
<br/><br/> "이 결과는 1000명의 표본으로 95%의 신뢰도로 추정됩니다." 독자들이 고교 수학과정에서 모집단의 평균추정 개념이다. 원리로만 따지면 우리가 어떤 값을 추정할때 추정대상의 분포가 Gaussian distribution을 따른 다는 전제하에 이러쿵 저러쿵 계산을 했었던 것으로 기억한다. Gaussian Process는 주로 회귀문제를 해결할때 자주 쓰이는 방식으로, 원하는 함수값 $f(x_{target})$을 Gaussian distribution을 따른다는 전제(Prior)하에 '모집단의 평균 추정'비슷한 연산으로 얻는 과정이다.

<br/> 멋들어지게 Gaussian distribution의 식을 음미해보자.
<br/><br/>
$$\mathcal{N}(x; \mu,\sigma)={{{1}\over{\sqrt{2\pi{\sigma}^2}}}exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)} $$(1)
<br/><br/>
$$\mathcal{N}(X; M,\Sigma)={{{1}\over{{\left(2{\pi}{|\Sigma|})\right)}}^{\frac{n}{2}}}exp\left(-\frac{1}{2}(X-M)^{T}{\Sigma}^{-1}(X-M)    \right)} $$(2)
<br/>
단일 변수에 대한 Gaussian distribution은 식 (1)처럼 작성되며, 변수($x$), 평균($\mu$), 그리고 분산($\sigma$)은 당연히 스칼라 값이다. 반면에, 다 변수에 대한 Gaussian distribution은 식 (2)처럼 작성되며, 변수($X$), 평균($M$), 그리고 분산($\Sigma$)는 변수의 개수만큼의 벡터($vector$), 혹은 행렬($matrix$)로 표현된다. 
<br/><br/>
필자가 이해했을때 다 변수의 Gaussian distribution에 대해서 가장 껄끄러웠었던 부분이 분산, 엄밀히 공분산(Covariance)라는 개념이었는데 두 변수간의 분산관계를 
