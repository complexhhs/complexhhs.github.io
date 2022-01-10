---
layout: post
title: "Reinforcement Learning_[6]"
author: "Hyunseok, Hwang"
categories: journal
tags: [documentation,sample]
---

# Introduction
---

여섯번째 RL시리즈다. 역시나 이번 포스팅으로 RL을 처음보게 되었다면 [지난 포스팅](https://complexhhs.github.io/Reinforcement_Learning_(5)DQN)부터 차근차근 진행해주기 바란다. 

이번 포스팅은 agent를 학습하는 방식으로 $\pi$를 최대화시키는 *Policy gradient*학습방식의 이론부분을 점검하고자 한다. 지금까지 배웠던 Q-learning방식, DQN과의 차이점이 무엇인지 비교해보며 어떤 의미를 가지고 있는지 간략하게 살펴보고 예시까지 다뤄보고자 한다.

# Policy gradient
---
