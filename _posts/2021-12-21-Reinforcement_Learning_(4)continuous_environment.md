---
layout: post
title: "Reinforcement Learning_[4]"
author: "Hyunseok, Hwang"
categories: journal
tags: [documentation,sample]
---

RL의 기본 개념부터 시작해서 TD prediction 알고리즘인 **SARSA, Off-policy Q-learning**까지 [지난 포스팅](https://complexhhs.github.io/Reinforcement_Learning_(3)policy_update)을 통해서 살펴봤다. 많은 개념들을 다루었지만 아마 독자들은 지금까지 포스팅을 보면서 (내용을 잘 소화했다면)큰 아쉬움을 느꼈을 것이다. 바로, 예시로 든 environment들이 굉장히 정형화 되어있는 격자세계(Grid world)로 구성되었다는 점이다. 격자 환경에서 $s$가 이산화 되어있으나 실제로 풀어야 할 RL문제, 실제로 인간이 마주하고 있는 세계는 연속적인 공간으로 이루어져 있기에 이 환경에 맞도록 RL을 생각해야 한다. 이제는 agent가 인식하는 environment는 이산화된 격자가 아니며, 연속된 수로 이루어진 실제 세계에서 활동할 차례이다. 이번 포스팅에서는 이 해결전략에 대해서 다뤄보고자 한다.

# Example. CartPole
---
[RL 첫번째 포스팅](https://complexhhs.github.io/Reinforcement_Learning_(1)%EA%B0%9C%EB%85%90)에서 잠깐 소개했었던 ***CartPole***문제를 고찰해보자.

![Carpole 그림]()

Agent의 목표는 위 그림에 나오는 Cart를 화면 밖에 나가지 않으면서 동시에 위에 세워진 기둥을 쓰러트리지 않고 최대한 오래 버티는 것이 목표다. Environment의 $s$를 확인해보면 4차원의 벡터로 구성되어있고 각각의 차원이 의미하는 바는 1) Cart의 위치, 2)Cart의 속도, 3)기둥과 Cart와의 각도, 4)기둥의 각속도를 의미한다. 취할 수 있는 $\pi$는 왼쪽-오른쪽의 이항분포를 따른다. 
$\pi$는 지난 포스팅에서 ***Epsilon-greedy***방식을 기반으로 한 Q-function업데이트로 변경되는 부분을 확인했으니 계산을 할 수 있는데, $s$의 정보가 연속적인 수치로 이루어짐을 생각하자. 앞선 코드의 일부
```python
   def set_qvalue(self,state,action,value):
        if state not in self._qvalue: #초기화
            self._qvalue[state] = {}
            for possible_action in self.possible_actions:
                self._qvalue[state][possible_actiosn]=0
        else:
            self._qvalue[state][action]=value
```
와 같이 Q-function을 테이블 형태로 만들기가 어려워진다. 이제 연속적인 환경에서의 agent가 $s$를 인식하는 부분에 대해서 논의하고 본격적인 ***CartPole***문제를 해결해보도록 하자.

#
---
