import math
import numpy as np


# Time complexity O(KN^2), Space complexity O(KN)
# 暴力法
def dp1(dp, K, N):
    # 三种边界条件
    if N == 0:
        print('K=%d, N=%d, output=%d' % (K, N, 0))
        return 0
    if N <= 1:
        print('K=%d, N=%d, output=%d' % (K, N, 1))
        return 1
    if K == 1:
        print('K=%d, N=%d, output=%d' % (K, N, N))
        return N
    if dp[K][N] > 0:
        # 已经求解过，直接返回
        print('Has value K=%d, N=%d, output=%d' % (K, N, dp[K][N]))
        return dp[K][N]
    move = math.inf
    for i in range(1, N + 1):
        # 历史最小步数和新的步数比较，取最小值
        move = min(move, max(dp1(dp, K - 1, i - 1), dp1(dp, K, N - i)) + 1)
    dp[K][N] = move
    print('Add value K=%d, N=%d, output=%d' % (K, N, move))
    return move


# Time complexity O(KNlogN), Space complexity O(KN)
# 二分法
def dp2(dp, K, N):
    # print('k = %d, n = %d' % (K, N))
    # 三种边界条件
    if N == 0:
        print('K=%d, N=%d, output=%d' % (K, N, 0))
        return 0
    if N <= 1:
        print('K=%d, N=%d, output=%d' % (K, N, 1))
        return 1
    if K == 1:
        print('K=%d, N=%d, output=%d' % (K, N, N))
        return N
    if dp[K][N] > 0:
        # 已经求解过，直接返回
        print('Has value K=%d, N=%d, output=%d' % (K, N, dp[K][N]))
        return dp[K][N]
    move = math.inf
    low = 1
    high = N
    while low < high:
        mid = low + ((high - low) // 2)
        # print('mid = %d' % (mid))
        safe_move = dp2(dp, K, N - mid)
        break_move = dp2(dp, K - 1, mid - 1)
        # print('mid = %d, safe = %d, break = %d' % (mid, safe_move, break_move))
        if safe_move > break_move:
            # 阈值楼层f在较高楼层，因此改变low
            low = mid + 1
        else:
            # 阈值楼层f在较低部分楼层，因此改变high
            high = mid
        move = min(move, max(break_move, safe_move) + 1)

    dp[K][N] = move
    print('Add value K=%d, N=%d, output=%d' % (K, N, move))
    return move

# Time complexity O(KN), Space complexity O(KN)
# 状态转移方程优化为：dp[k][m] = n表示，k个鸡蛋和步数m能够覆盖的最多楼层数n
# dp[k][m-1]：鸡蛋完整的状态，表示当前楼层以上层数
# dp[k-1][m-1]：鸡蛋破碎的状态，表示当前楼层以下的层数
def dp3(K, N):
    dp = [[0] * (N + 1)] * (K + 1)
    m = 0
    while dp[K][m] < N:
        m += 1
        for k in range(1, K + 1):
            dp[k][m] = dp[k - 1][m - 1] + dp[k][m - 1] + 1
            print('Add value k=%d, m=%d, output=%d' % (k, m, dp[k][m]))
    return m



# 测试代码solution1
K = 2
N = 6

# print('dp1 start')
# memo1 = [[0] * (N + 1)] * (K + 1)
# res = dp1(memo1, K, N)
# print(res)
# print('dp1 end')

print('dp2 start')
# 测试代码solution2
# memo2 = [[0] * (N + 1)] * (K + 1)
memo2 = np.zeros((K+1, N+1))
res = dp2(memo2, K, N)
print(res)
print('dp2 end')

# print('dp3 start')
# # 测试代码solution3
# res = dp3(K, N)
# print(res)
# print('dp3 end')

