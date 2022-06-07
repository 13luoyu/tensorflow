

# 凸性
# 凸集：一个集合X，对于任意a,b属于X，连接a和b的线段也位于X中，则集合X是凸的
# 即对于λ属于[0,1]，a,b属于X，有λa + (1-λ)b属于X
# 凸函数：对于一个凸集X，若对于所有a,b属于X和λ属于[0,1]，有
# λf(a) + (1-λ)f(b) >= f(λa + (1-λ)b)，则函数f为凸函数
# 性质：凸函数的局部极小值也是全局极小值
# 凸函数的下水平集是凸的，下水平集S = {x|x属于X and f(x)<=b}
# 凸性和二阶导数，凸函数二阶导数>=0

# 对于无约束的最优化问题，可以用梯度下降等方法
# 有约束的最优化问题问题：x -> min f(x), subject to ci(x) <=0 for all i in {1..N}
# f是目标函数，ci是约束函数，我们将其转化为无约束的最优化问题：
# 拉格朗日函数：L(x,a1,...,an) = f(x) + 求和(ai*ci(x)) where ai >= 0
# 求解一阶偏导构成的方程组=0，即可解出x和拉格朗日乘数a1...an
# 我们希望L相对于ai大，相对于x小，因此L的鞍点就是原始约束优化问题的最优解


# 在深度学习的背景下，凸函数的主要目的是帮助我们详细了解优化算法。 我们由此得出梯度下降法和随机梯度下降法是如何相应推导出来的。
#
# 凸集的交点是凸的，并集不是。
#
# 根据詹森不等式，“一个多变量凸函数的总期望值”大于或等于“用每个变量的期望值计算这个函数的总值“。
#
# 一个二次可微函数是凸函数，当且仅当其Hessian（二阶导数矩阵）是半正定的。
#
# 凸约束可以通过拉格朗日函数来添加。在实践中，只需在目标函数中加上一个惩罚就可以了。
#
# 投影映射到凸集中最接近原始点的点。