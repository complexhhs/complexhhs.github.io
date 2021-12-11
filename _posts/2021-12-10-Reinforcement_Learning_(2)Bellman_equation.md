---
layout: post
title: "Reinforcement Learning_[2]"
author: "Hyunseok, Hwang"
categories: journal
tags: [documentation,sample]
---

혹시 강화학습에 대한 기초 개념을 모른다면 [이전 포스팅](https://complexhhs.github.io/Reinforcement_Learning_(1)%EA%B0%9C%EB%85%90)부터 보고 와주기를 바란다. 이번 포스팅에선 지난 포스팅에 이어서 Bellman equation의 이론과 그에 따른 agent의 최적화 방식을 조금 더 면밀히 살펴보고, 그럴싸한 강화학습 문제를 풀어보도록 하겠다.

# Bellman equation

Bellman equation은 특별한 것은 아니고, 사실 지난 포스팅의 Value function과 Action-Value function을 전개한 식이 바로 Bellman equation이다. 하지만 그 이면에 더 세세한 사항들에 대해서 고려할 부분이 많기에 따로 챕터를 잡아 설명한다.RL은 일종의 ***Markov Decision Process***를 푸는 과정과 같으며, 그 알고리즘은 일종의 동적계획법(Dynamic Programming)이라고 먼저 언급했었다. 엄밀히 말하자면 RL은 MDP와는 엄연히 다른 방식인데 그 이유를 포스팅이 진행되는 과정에서 밝히도록 하겠다. 다시 돌아와, 최종보상을 최대화 하는 과정은 현재의 상태와 액션 $s_t, a_t$로부터 Value function, 혹은 Action-Value function(이하, Q-function으로 명명)을 미래의 상태와 액션 $s_{t+1}, a_{t+1}$로 쪼개어 수식으로 전개해서 푸는 과정을 다시 한번 보자.

- Value function
    - $$\begin{align} V_{\pi}(s) &= \mathbb{E}_{\pi}[G_t\vert S_t=s] \\
&= \mathbb{E}_{\pi}[R_{t+1}+\gamma{R_{t+2}}+\gamma^2{R_{t+3}}\cdots \vert S_t=s] \\
&= \mathbb{E}_{\pi}[R_{t+1}+\gamma \left( {R_{t+2}}+\gamma{R_{t+3}}\cdots \right) \vert S_t=s] \\
&= \mathbb{E}_{\pi}[R_{t+1}+\gamma  G_{t+1} \vert S_t=s] \\
&= \mathbb{E}_{\pi}[R_{t+1}+\gamma  V_{\pi}(S_{t+1})  \vert S_t=s] \\
\end{align}.$$(1)

- Q-function
    - $$\begin{align} q_{\pi}(s,a) &= \mathbb{E}_{\pi}[G_t\vert S_t=s, A_t=a] \\
&= \mathbb{E}_{\pi}[R_{t+1}+\gamma{R_{t+2}}+\gamma^2{R_{t+3}}\cdots \vert S_t=s, A_t=a] \\
&= \mathbb{E}_{\pi}[R_{t+1}+\gamma \left( {R_{t+2}}+\gamma{R_{t+3}}\cdots \right) \vert S_t=s, A_t=a] \\
&= \mathbb{E}_{\pi}[R_{t+1}+\gamma  G_{t+1} \vert S_t=s, A_t=a] \\
&= \mathbb{E}_{\pi}[R_{t+1}+\gamma  q_{\pi}(S_{t+1},A_{t+1})  \vert S_t=s, A_t=a] \\
\end{align}.$$(2)


지난 포스팅에서도 그랬지만 말과 수식으로만 설명하려니 한계가 느껴지는 것 같다. 그래서 Value function, Q-function, 그리고 두루뭉술하게 넘어갔던 Policy($\pi$)를 그림이 포함된 일련의 과정과 함께 살펴보자.

![state_1_value_Q]()

그림의 투명한 원은 Value function의 값, 검은 원은 Q-function의 값을 의미한다. 또 environment로부터 agent가 취할수 있는 action의 옵션은 단 두가지로 제한되어있는 상황이라 Q-function으로 나가는 경로가 두  갈래로 표현되었다. 현재 state에서의 value는 모든 action에 대한 return값을 계산하지만, Q-function은 현재 state중에서도 특정한 action에 각각에 대한 return값을 계산하는 것을 고려한다. 따라서, 흰색원으로 부터 검은색 원으로 값이 분화될때 그 갈래로 나뉘는 확률이 바로 policy($\pi)이며, 특정 action이 나오는 확률을 의미한다. 이 과정을 종합하여, Value function과 Q-function간의 관계는 아래와 같다

$$V_{\pi}(s)=\sum_{a \in \mathcal{A}}{\pi(a \vert s) q_{\pi}(s,a)}.$$(3)

![state_2_Q_value]()

현재 state에서 다음 state로 넘어가는 상황에서의 Q-function과 Value function간의 관계는 위와 같다. 수식으로 표현된 문자가 혼용되어 굉장히 독자들에게 미안하지만 작성의 편의를 위해서 양해를 바란다. 현재 state와 action을 $s,a$로, 미래 state와 action을 $s', a'$로 표현함을 미리 밝힌다. $s \rightarrow s'$로 가는 것은 agent가 $a$ action을 취했다는 의미가 된다. 우리가 간과하기 쉬운 부분이 있는데 environment에서 $a$를 취했다고 예상되는 구체적인 $s'$가 생기는 것이 아니다. 실제로 environment의 정보를 완벽하게 아는 agent는 없다. 예를 들어 신호등의 직진신호를 받아 차를 출발했는데 무사히 잘 출발하리라는 보장을 할 수 없다. 왜냐하면, 갑자기 차도를 향해 뛰어드는 무단보행자나 앞차가 신호를 보지 못하고 출발하지 못해서 나도 출발 할 수 없는 상황에 마주하기 때문이다. 이는 우리가 앞서서 **State transition probability**, $P_{ss'}^{a}$를 언급했었던 부분이다. 즉, 현재 $a$를 취한다고 해서 $s \rightarrow s'$의 확률이 1이 아닐수도 있음을 고려 해야한다. 위 그림은 예기치 못한 경우로 분화되는 케이스를 하나를 더 두어 $s'$가 두 가지 케이스로 나뉠 수 있음을 보여준다. 따라서, Q-function을 미래의 Value function로 표현하면 다음과 같다.

$$q_{\pi}(s,a) = R_{s}^{a}+\gamma \sum_{s' \in S}P_{ss'}^{a} V_{\pi}(s').$$(4)
식 (4)에선 미래의 state로 진행됨에 따라 environment로 부터 reward $R_{ss'}^{a}$를 받는 부분을 놓치지 않도록 유의하자.

이제 Value-function $\rightarrow$ Q-function의 관계와 Q-function $\rightarrow$ Value-function의 관계를 종합하여, Value-function과 Q-function 각각의 동적계획법 형태로 풀이된 Bellman equation으로 표현 할 수 있다.

- Bellman equation style Value-function

![state_3_value_Q_value]()

$$\begin{align} V_{\pi}(s)&=\sum_{a \in \mathcal{A}}{\pi(a \vert s) q_{\pi}(s,a)} \\
&=\sum_{a \in \mathcal{A}}{\pi(a \vert s)}\left\{R_{s}^{a}+\gamma \sum_{s' \in S}P_{ss'}^{a} V_{\pi}(s') \right\}. \end{align}$$(5)

- Bellman equation style Q-function

![state_4_Q_value_Q]()

$$\begin{align} q_{\pi}(s,a)&= R_{s}^{a}+\gamma \sum_{s' \in S}P_{ss'}^{a} V_{\pi}(s') \\
&=R_{s}^{a}+\gamma \sum_{s' \in S}P_{ss'}^{a}\left\{ \sum_{a' \in \mathcal{A}}{\pi(a' \vert s') q_{\pi}(s',a')} \right \}. \end{align}$$(6)

# Bellman optimality equation
