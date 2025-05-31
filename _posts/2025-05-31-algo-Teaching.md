---
title: "[알고리즘] 백준 1062 가르침"
date: 2025-05-31 17:30:00 +0900
categories: [알고리즘, 백준]
tags: [알고리즘, 백준, 백트래킹]
---


오늘 풀이할 알고리즘은 백준 사이트의 [가르침](https://www.acmicpc.net/problem/1062)라는 문제이며 코드는 파이썬으로 작성되어 있습니다.

---
### 요약

언어를 모르는 학생에게 알파벳을 가르치려고 합니다.

**N 개의 단어**가 주어지고 **K 개의 알파벳**을 가르칠 수 있습니다.

**어떤 K 개의 알파벳**을 가르쳐야 **최대한 많은 단어**를 읽을 수 있고

이 때 읽을 수 있는 **단어의 개수를 출력**하는 문제입니다. 

( 모든 단어는 anta 로 시작해서 tica 로 끝이 납니다. )

---

### 문제조건 

1 ≤ N ≤ 50

1 ≤ K ≤ 26

제한 

메모리 128MB , 시간 1초 

( Pypy3 시간 제한: ×3+2 초 메모리 제한: ×2+128 MB )

---

### 풀이

어떤 K 개의 알파벳으로 단어를 읽기 위해서는 우선 **어떤 K 개의 알파벳 조합**을 만들어야 된다.

까지는 바로 떠올릴 수 있을 겁니다.

모든 단어는 anta 로 시작해서 tica 로 끝이 나기 때문에 

알파벳 a ~ z, 26개 중 **a, n, t, i, c 다섯 알파벳을 제외**하고 조합을 만들면 될겁니다.

답을 구하기 위해선 하나의 조합에서 주어진 단어를 순회하며 읽을 수 있는가를 판별해서 개수를 세어가면 됩니다.

1. **어떤 K 개의 알파벳 조합**을 만든다.

2. 각 조합을 **N 개의 단어에 대조**하여 **최대 몇 개**를 읽을 수 있는지를 센다.

---

### 해설

문제 입력은 다음과 같습니다.
```py
3 6 # N K
antarctica # N 개의 단어 순서대로 입력됨
antahellotica
antacartica
```

단어 조합을 만들기에 앞서 모든 단어는 anta, tica 로 끝나기 때문에 입력 값을 조금 조정하겠습니다.

N 개의 단어를 입력받고 앞, 뒤로 4 개씩 자르고

```py
for i in range(N):
    word = _input().rstrip()
    word = word[4:][:-4]
    # antarctica => rc
    # antahellotica => hello
    # antacartica => car
```
전체 단어를 알고 싶은게 아니라 어떤 알파벳 구성인지만을 알면 되기에

각 단어는 맵의 형태로 { 'r' : 1 } 저장할 겁니다. 

현재 {키: 값} 형태에서 값은 크게 의미가 없으므로 어떤 값을 저장해도 무의미 합니다.

a ~ z 알파벳 중 a, n, t, i, c 을 제외하고 주어진 단어에서 사용된 알파벳만을 alphabet 에 저장했습니다.

```py
for al in word:
    if al in 'antic': # 기본적으로 포함되는 a, n, t, i, c 는 제외합니다.
        continue
    words[i][al] = 1
    alphabet[al] = 1 # 입력에서 주어진 단어들에 사용된 알파벳만을 저장
    # antarctica => rc => c 제외 => { r : 1 }
    # antahellotica => hello => { h : 1, e : 1, l : 1, o : 1 }
    # antacartica => car => a, c 제외 => { r : 1 }
```

조합은 dfs 를 사용해서 만들었습니다. 

재귀보다는 반복문의 형태를 선호해서 stack 을 이용해서 만들었습니다.

dfs 에 관한 내용은 [[알고리즘] DFS](/posts/algo-DFS)을 참고해주세요.

알파벳 리스트에서 자신의 인덱스 다음의 인덱스만을 바라보게 해서 순열이 아닌 조합을 만들면 되겠습니다.

조합은 stack 에 저장된 알파벳들로 구성되며 가르칠 수 있는 알파벳 수 K 길이까지만 저장합니다.

```py
def dfs():
    global output, K, max_L
    stack = []
    
    # 알파벳 순회
    for idx, al in enumerate(alphabet):
        stack = [[al, idx]]

        while stack:
            al, vi = stack[-1]

            if vi+1 < len(alphabet) and len(stack)+1 <= K:
                stack[-1][1] += 1
                stack.append([alphabet[vi+1], vi+1])
            else:
                al, vi = stack.pop()
```

위의 코드에서 스택에 있는 정보를 활용해서 N 개의 단어와 대조하여 읽을 수 있는 단어는 몇 개인지를 추가해나가면 되겠습니다.

### 코드

```python
import sys
from collections import defaultdict
_input = sys.stdin.readline
def minput(): return map(int, _input().split())

# N 개의 단어와 현재 스택에 있는 알파벳을 비교 후 읽을 수 있는 단어 개수 반환
def valid_check(stack):
    global K
    cnt = 0
    for word in words:
        if not word:
            continue
        if len(stack) < len(word):
            continue
        for alp in word:
            if not stack.get(alp):
                break
        else:
            cnt += 1
            
    return cnt

# 알파벳 조합 만들기 + 최대 개수 갱신
def dfs():
    global output, K, max_L
    stack = []
    
    for idx, al in enumerate(alphabet):
        stack = [[al, idx]]
        stack_dict = {}
        stack_dict[al] = 1
        
        while stack:
            al, vi = stack[-1]
            
            if len(stack) == K or len(stack) == len(alphabet):
                cnt = valid_check(stack_dict)
                output = max(output, cnt)

            if vi+1 < len(alphabet) and len(stack)+1 <= K:
                stack[-1][1] += 1
                stack_dict[alphabet[vi+1]] = 1
                stack.append([alphabet[vi+1], vi+1])
            else:
                al, vi = stack.pop()
                del stack_dict[al]

N, K = minput()
max_L = 0
alphabet = {}

# a n t i c 를 배우지 못하면 읽을 수 있는 글자가 없음
if K < 5:
    print(0)
    exit()
# a n t i c 을 제외하고 가르칠 수 있는 알파벳 수로 갱신
K -= 5


is_exist = False # 배울 수 있는 단어가 있다면 True 없다면 False
words = [{} for _ in range(N)]
for i in range(N):
    word = _input().rstrip()
    word = word[4:][:-4]
    for al in word:
        if al in 'antic':
            continue
        words[i][al] = 1
        alphabet[al] = 1
    if len(words[i]) <= K:
        is_exist = True
    max_L = max(max_L, len(words[i]))

# 배울 수 있는 단어가 없다면 종료
if not is_exist:
    print(0)
    exit()

default_cnt = 0 # a n t i c 으로만 이루어진 단어 수
output = 0
for word in words:
    if len(word) == 0:
        default_cnt += 1
        
alphabet = tuple(alphabet)

if K == 0:
    print(default_cnt)
else:
    dfs()
    print(output+default_cnt)
```

---

### 고찰

백트래킹보다는 브루트포스에 가까운 풀이라고 생각됩니다만 

그렇더라도 문제를 풀면서 어떤 분기점을 제거하면 연산량을 줄일 수 있는지 고민해보는 것이 좋을 것 같습니다.

브루트포스처럼 풀었지만 관점을 달리하면 알파벳의 모든 조합을 N 개에 단어와 대조해보는 것에서

K 개의 알파벳 조합을 완성했을 때만 N 개의 단와와 대조해보는 것으로 가지치기를 했다고 볼 수도 있을 것 같습니다.

실제 nCr 조합에서 n = 21, r = 1 ~ 21 의 총합이 약 200만 정도로 모든 조합을 다 대조하면 최대 50개의 단어를 비교하면

1억 연산량이 넘어가서 시간 초과가 나올겁니다.

