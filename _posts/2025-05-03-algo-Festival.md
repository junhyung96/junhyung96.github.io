---
title: "[알고리즘] 소프티어 축제"
date: 2025-05-03 09:00:00 +0900
categories: [알고리즘, 소프티어]
tags: [알고리즘, 소프티어, 오일러 경로 테크닉, 리루팅, 누적합, DFS]
---


오늘 풀이할 알고리즘은 소프티어 사이트의 [축제](https://softeer.ai/practice/11005)라는 문제이며 코드는 파이썬으로 작성되어 있습니다.

소프티어 사이트 내의 [해설](https://softeer.ai/connect/blog/76)을 참고하여 풀이하였습니다.

---
### 요약

**양방향** 이동 가능한 **가중치가 있는 간선**으로

임의의 두 정점 사이가 이동 가능한 **트리**의 정보가 주어집니다.

두 가지 쿼리를 해결하는 문제로

첫번째는 **축제 비용**을 출력하는 것
- 축제 비용이란  축제가 열리는 노드가 있고
각 노드에서 해당 노드로 가는 최단 거리 * 인구수를 모두 더한 합을 말함

두번째는 정점의 **인구수를 증가**시키는 것입니다.

![img](/post/2025/2025-05-03/algo_img_1.png)

---

### 문제조건 

1 ≤ N, Q  ≤ 60,000

1 ≤ Gi ≤ 1,000

1 ≤ Xi, Yi ≤ N

1 ≤ Li ≤ 1,000

1 ≤ vi ≤ N

1 ≤ gi ≤ 1,000

메모리 1024MB , 시간 15초 (Python)

---

### 풀이

1번 노드를 기준으로 축제 비용을 계산하고 리루팅을 통해 모든 정점의 축제 비용을 구합니다.

인구수 증가 쿼리가 발생하면 해당 정보를 저장합니다.

특정 정점의 축제 비용을 계산하는 쿼리가 발생하면 인구수가 증가한 노드로 부터 해당 정점까지의 최단거리를 찾아 

리루팅을 통해 기존에 구해두었던 축제 비용에 더해 출력합니다.

인구수 증가 쿼리가 N 의 제곱근 개수만큼 쌓이면 리루팅을 통해 모든 정점에 대한 축제 비용을 다시 한 번 업데이트 합니다.

---

### 해설

1번 쿼리를 해결해 보려 합니다.

노드는 1번부터 N번까지 순서대로 번호가 부여되어있습니다

저는 1번노드를 루트노드로 기준으로 삼고 진행하겠습니다.

1번에서 축제비용은 어떻게 산정할 수 있을 까에 대한 부분은
DFS 를 통해 간선비용 * 서브트리 인구수 를 1번 노드 축제 비용에 전부 더하면 됩니다

그렇다면 각 노드마다 DFS 를 통해 축제비용을 계산한다면 시간초과가 나겠죠

여기서 리루팅의 개념이 들어옵니다.

1번 노드에서 2번 노드로 루트노드를 이동한다면 축제비용은 어떻게 산정할까인데

2번 노드의 서브트리 인구수가 1번 노드로 가는 간선을 이용한 비용을 빼고

1번 노드의 인구가 2번노드로 가는 간선을 이용한 비용을 더하게 됩니다. 

마찬가지로 2번노드에서도 다른 노드로 이동하면서 축제 비용을 산정합니다.

이렇게 모든 노드를 방문해서 축제비용을 계산할 수 있습니다.

우선 코드를 보죠

```python
cur, pre, cost # 현재 노드, 부모 노드, 간선비용
sub_pops[pre] += sub_pops[cur]

s1 = sub_pops[cur] * cost
s2 = (total_pop - sub_pops[cur]) * cost
i = Ein[cur]
o = Eout[cur]

fest[1] += s1 
fest[i] += (s2 - s1) # i에서 축제가 열린다 치면 루트로 가는 s1 은 돌려받고 루트에서 오는 s2 는 더해줘야 함
fest[o+1] -= (s2 - s1) # 누적합이 적용될 구간 밖은 다시 빼주기
```

s1 = 현재 노드의 서브트리 총 인구(sub_pop)가 부모 노드로 가는 간선을 타고 가는 비용

s2 = 전체 인구에서 sub_pop 을 제외한 인구가 부모 노드에서 간선을 타고 오는 비용

DFS 로 순회하며 각 노드 방문 시

1번 노드는 들어오는 간선에 대한 비용만 고려하면 되므로 s1 을 더하기만 하면 됩니다.

해당 노드의 방문 시점에 해당 하는 인덱스에는 간선을 타고 오는 비용(s2)을 더하고 간선을 타고 가는 비용(s1)을 뺍니다.

(s1을 빼는 의미는 1번 노드를 기준으로 누적합 계산을 진행하므로 1번 노드의 축제 비용에는 이미 가는 비용이 더해져 있으니 그 만큼 빼는 겁니다.)

해당 노드의 나간 시점+1에 해당 하는 인덱스에는 s2-s1 을 빼주게 됩니다.

이것에 대한 의미는 진입시점 ~ 나간 시점은 하나의 서브트리로 2번 노드의 경우 fest[2] 와 fest[6] 을 갱신합니다.

누적합을 이용해 업데이트 할경우 2번 노드의 서브트리는 모두 해당 값을 더하게 되므로 현재 노드의 간선을 고려한 비용을 적용받게 됩니다.

![img](/post/2025/2025-05-03/algo_img_3.png)

---

2번 쿼리는 증가 시킨 인구 수를 어딘가에 저장해 둡니다.

1번 쿼리 수행 시 축제가 열리는 노드와 인구 수가 증가한 노드의 최소 공통 부모(LCA) 를 찾아 최단거리를 계산 후 인구수를 곱한 비용만큼 더합니다.

![img](/post/2025/2025-05-03/algo_img_2.png)

2번 쿼리가 루트 N 만큼 쌓이면 모든 노드의 축제 비용을 갱신하는 이유는

2번 쿼리 마다 축제 비용을 갱신하기 위해서 dfs 를 사용하게 되면 O(N) 을 쿼리 수 만큼 시간이 걸리고

인구수 증가 분을 더하는 과정. 즉, LCA 계산에는 O(logN) 약 16번 정도의 연산이 쿼리 2의 개수만큼 계속해서 쌓이게 됩니다.

축제 비용을 갱신하지 않을 때

총 60000개의 쿼리 중 30000개의 인구수 업데이트가 있고 30000개의 축제비용을 질의하는 쿼리가 주어진다 가정한다면

30000(인구수 업데이트 수)*16(거리게산) * 30000(축제비용 쿼리 수) 가 되므로 약 144억 정도의 연산량으로 1초에 2000만번으로 추정하는 파이썬으로는 15초안에 절대 해결할 수 없을 겁니다.

DFS 로 축제 비용을 갱신하는 O(N) 과 거리계산에 소요되는 O(logN) * 쿼리수 를 

적절히 조절해서 연산이 너무 무거워지지 않도록 해야 하며

해당 문제에서는 루트N 정도의 인구수 업데이트가 발생하면 축제 비용을 갱신하도록 했습니다.



### 코드

```python
import sys
_input = sys.stdin.readline
def minput(): return map(int, _input().split())

def _dfs1():
    global time
    cur = 1
    pre = 0
    d = 1
    l = 0

    time += 1
    Ein[cur] = time
    depth[cur] = d
    dist[cur] = l
    par[cur][0] = pre
    visited = [False] * (60_001)
    visited[cur] = True

    stack = [[cur, d, l, 0]]
    while stack:
        cur, d, l, vi = stack[-1]

        if vi < len(adj_ls[cur]):
            nxt, cost = adj_ls[cur][vi]
            stack[-1][3] += 1
            if visited[nxt]:
                continue
            visited[nxt] = True
            time += 1
            Ein[nxt] = time
            depth[nxt] = d+1
            dist[nxt] = l+cost
            par[nxt][0] = cur
            stack.append([nxt, d+1, l+cost, 0])
        else:
            if stack:
                cur, d, l, vi = stack.pop()
                Eout[cur] = time


def _dfs2():
    global total_pop
    cur = 1
    visited = [False] * (60_001)
    visited[cur] = True
    sub_pops[cur] = pops[cur]
    stack = [[cur, 0, 0, 0]] # 현재노드, 부모노드, 거리

    while stack:
        cur, pre, l, vi = stack[-1]
        # print(stack)
        if vi < len(adj_ls[cur]):
            nxt, cost = adj_ls[cur][vi]
            stack[-1][3] += 1
            if visited[nxt]:
                continue
            visited[nxt] = True
            sub_pops[nxt] = pops[nxt]
            stack.append([nxt, cur, cost, 0])

        else:
            if stack:
                cur, pre, cost, vi = stack.pop()
                sub_pops[pre] += sub_pops[cur]

                # s1 = nxt 의 서브트리가 cur 로 가는 비용
                s1 = sub_pops[cur] * cost
                s2 = (total_pop - sub_pops[cur]) * cost
                i = Ein[cur]
                o = Eout[cur]
                fest[1] += s1 
                # 리루팅
                fest[i] += (s2 - s1) # i에서 축제가 열린다 치면 루트로 가는 s1 은 돌려받고 루트에서 오는 s2 는 더해줘야 함
                fest[o+1] -= (s2 - s1) # 누적합이 적용될 구간 밖은 다시 빼주기


def lca(a, b):
    if depth[a] < depth[b]: # a 가 깊도록
        a, b = b, a 
    
    diff = depth[a] - depth[b]
    for k in range(logN):
        if diff & (1<<k):
            a = par[a][k]

    if a == b:
        return b
    
    for k in range(logN-1, -1, -1):
        if par[a][k] != par[b][k]:
            a = par[a][k]
            b = par[b][k]
    return par[a][0]

# 트리정보 정점, 간선, 인구
N = int(_input())
max_N = max(N, 60_000)
logN = (max_N).bit_length()
sqrtN = int(max_N ** 0.5)
pops = [0] * (max_N+1) # 깐프 인구 수
idx = 1
for v in minput():
    pops[idx] = v
    idx += 1
total_pop = sum(pops)
adj_ls = [[] for _ in range(max_N+1)] # 간선 정보
depth = [0] * (max_N+1) # 트리 깊이
dist = [0] * (max_N+1) # 루트로부터 거리
fest = [0] * (max_N+2) # 축제 비용
sub_pops = [0] * (max_N+1) # 서브트리 인구수

for _ in range(N-1):
    a, b, c = minput()
    adj_ls[a].append((b, c))
    adj_ls[b].append((a, c))

# 오일러경로
time = 0
Ein = [0] * (max_N+1)
Eout = [0] * (max_N+1)

# LCA 희소 테이블 binary lifting
par = [[0]*logN for _ in range(max_N+1)]

# 오일러 경로 초기화 dfs1 ett
_dfs1()
# 축제 비용 초기화 dfs2 rerooting 
_dfs2()
for i in range(1, max_N+1): # 누적합 로직
    fest[i] += fest[i-1]

# 희소테이블 초기화
for j in range(1, logN):            
    for i in range(1, max_N+1):
        par[i][j] = par[par[i][j-1]][j-1]

# 쿼리 수행
arr = []
output = []
Q = int(_input())
for _ in range(Q):
    query = tuple(minput())
    if query[0] == 1:
        result = fest[Ein[query[1]]]
        for v, g in arr:
            # 인구수 증가분만큼 적용
            lca_node = lca(query[1], v)
            result += (dist[query[1]] + dist[v] - 2 * dist[lca_node]) * g
        output.append(str(result))
    else:
        # 쿼리 = 번호, 정점, 인구증가분
        arr.append((query[1], query[2]))

        # 루트N 마다 갱신 
        if len(arr) == sqrtN:
            for v, g in arr:
                pops[v] += g
            fest = [0] * (max_N+2)
            total_pop = sum(pops)
            _dfs2()
            for i in range(1, max_N+1):
                fest[i] += fest[i-1]
            arr.clear()

print("\n".join(output))
```

---

### 고찰

사실 풀이하면서 많이 헤맸습니다. 

트리를 선형화하는 방법은 기존에 알고 있던 HLD 로는 접근하기가 어려웠습니다.

처음에는 DFS 를 통해 직접 노드를 순회하면서 리루팅하고 LCA 를 통해 거리를 계산했지만

풀리지 않아 해설을 보고 구현해보는 쪽으로 진행했습니다.

제곱근 쿼리마다 분할한다던가 

오일러 경로 테크닉을 통해 트리를 선형화 해서 누적합을 통해 빠르게 리루팅을 구현한다던지

LCA 를 구하기 위해서는 희소 테이블을 리루팅을 하기 위해서 누적합을 사용하고 하나의 문제이지만

다양한 알고리즘 지식들을 함께 고민해볼 수 있는 좋은 문제라고 생각합니다. 

추가로 DFS 를 처음에 재귀형식으로 구현했는데

첫번째로 파이썬에서 재귀 한도를 풀어야 했고 ( sys.setrecursionlimit(10**5) )

재귀 호출마다 메모리 사용량이 증가하는데 1024MB 를 가뿐히 넘어갑니다.

그래서 DFS 를 stack 을 사용한 loop 형식으로 바꾸었는데 중복 방문으로 인한 시간이 많이 소요되었습니다.

이 부분은 for 문에서 직접 인덱스로 접근하게끔 바꾸었고 시간적인 비용을 많이 낮출 수 있었습니다.

아래에는 다른 정보를 제외한 dfs 만 어떻게 바꾸었는지 에 대한 코드입니다.
```python
def _dfs1():
    # 정보 초기화

    stack = [[cur, d, l]]
    while stack:
        cur, d, l = stack[-1]

        for nxt, cost in adj_ls[cur]: 
            if visited[nxt]:
                continue
            # 방문 노드 정보 업데이트
            break
        else:
            if stack:
                cur, d, l = stack.pop()
```
```python
def _dfs1():
    # 정보 초기화

    stack = [[cur, d, l, 0]] 
    while stack:
        cur, d, l, vi = stack[-1] # vi 간선 인덱스

        if vi < len(adj_ls[cur]):
            nxt, cost = adj_ls[cur][vi] # vi 인덱스로 접근
            stack[-1][3] += 1 # 이용한 간선은 넘어가기
            if visited[nxt]:
                continue
            # 방문 노드 정보 업데이트
        else:
            if stack:
                cur, d, l, vi = stack.pop()
```
