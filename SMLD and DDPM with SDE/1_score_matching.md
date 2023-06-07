# Score matching的一点学习记录

主要记录下关于score matching的一些内容

## 归一化因子[1]

在机器学习领域，假设一个d维的数据$\vec{x}$是满足概率分布$p(\vec{x})$的，但是我们并不知道这个概率分布具体是什么，所以我们需要学习这个概率分布。我们用一个含有参数$\theta$的模型$q'_{\theta}(\vec{x})$来估计实际的概率分布，我们希望这两个概率分布之间的散度越小越好，最好小到0，这就说明这两个概率密度相等了。如果$p(\vec{x})$是$q_{\theta}(\vec{x})$的分布族的一个概率分布，那就可以通过学习来找到最优参数。一般来讲可以用KL散度，即

$$
D_{KL}(p||q_{\theta})=\int_{\vec{x}}p(\vec{x})log\frac{p(\vec{x})}{q_{\theta}(\vec{x})}d\vec{x}
$$

$$
q_{\theta}(\vec{x})=\frac{1}{Z_{\theta}}q'_{\theta}(\vec{x})
$$

这样做是因为$q'_{\theta}(\vec{x})$通常是一个解析式的表示或者神经网络，算出来的$q'_{\theta}(\vec{x})$必须进行归一化，也就是需要令$\int_{\vec{x}}q_{\theta}(\vec{x})=1$。这个归一化因子$Z_{\theta}=\int_{\vec{x}}q'_{\theta}(\vec{x})$是许多概率图问题的一大难题，如果是序列问题的话，需要对所有路径进行采样计算。假设序列长度是n、每一步标签是k，那就有$k^n$种可能的路径，计算量是指数级别的。

## 得分函数（score function）和得分匹配（score matching）[2]

### 得分函数

规避计算归一化因子的一个方法是引入得分函数，得分函数定义为

$$
\begin{align*}
    s_{\theta}(\vec{x})&=\nabla_{\vec{x}}logq_{\theta}(\vec{x})\\
    &=\nabla_{\vec{x}}[logq'_{\theta}(\vec{x})-logZ_{\theta}]
\end{align*}
$$

这样含有归一化因子的项就变成0。目标函数就变成

$$
J(\theta)=\frac{1}{2}\int_{\vec{x}\in \mathbb{R}^n}p(\vec{x})||s_{\theta}(\vec{x})-s(\vec{x})||^2 d\vec{x}
$$

但是怎么求这个得分函数呢？很简单，估计一个出来，也就是得分匹配。

### 得分匹配

我们把$J(\theta)$展开，有

$$
\begin{align*}
    J(\theta)&=\int p(\vec{x}) [\frac{1}{2}||s(\vec{x})||^2+\frac{1}{2}||s_{\theta}(\vec{x})||^2-s(\vec{x})^\dag s_{\theta}(\vec{x})]d\vec{x}\\
    &=\int p(\vec{x}) [\frac{1}{2}||s_{\theta}(\vec{x})||^2-s(\vec{x})^\dag s_{\theta}(\vec{x})]d\vec{x}+const
\end{align*}
$$

由于$\frac{1}{2}||s(\vec{x})||^2$并不依赖于$\theta$，所以将它视为一个常数项const。我们将$s(\vec{x})^\dag s_{\theta}(\vec{x})$单拿出来看，由于$s(\vec{x})$和$s_{\theta}(\vec{x})$都是列向量，因此$s(\vec{x})^\dag s_{\theta}(\vec{x})$可以看作是对应元素相乘之后求和，也就是

$$
-\sum_{i=1}^n \int p(\vec{x})s_i(\vec{x})s_{i,\theta}(\vec{x})d\vec{x}
$$

我们把$s(\vec{x})=\nabla_{\vec{x}}logp(\vec{x})$带入到上面的式子中，有

$$
\begin{align*}
    &-\sum_{i=1}^n \int p(\vec{x})\frac{\partial logp(\vec{x})}{\partial x_i}s_{i,\theta}(\vec{x})d\vec{x}\\
    =& -\sum_{i=1}^n \int \frac{p(\vec{x})}{p(\vec{x})}\frac{\partial p(\vec{x})}{\partial x_i}s_{i,\theta}(\vec{x})d\vec{x}\\
    =& -\sum_{i=1}^n \int \frac{\partial p(\vec{x})}{\partial x_i}s_{i,\theta}(\vec{x})d\vec{x}
\end{align*}
$$

根据部分积分法，有

$$
\begin{align*}
    &\lim_{a\to\infty,b\to-\infty}f(a,x_2,...,x_n)g(a,x_2,...,x_n)-f(b,x_2,...,x_n)g(b,x_2,...,x_n)\\
    =& \int_{-\infty}^{+\infty}f(\vec{x})\frac{\partial g(\vec{x})}{\partial x_1}dx_1+\int_{-\infty}^{+\infty}g(\vec{x})\frac{\partial f(\vec{x})}{\partial x_1}dx_1
\end{align*}
$$

因此，我们可以改写$-\int \frac{\partial p(\vec{x})}{\partial x_i}s_{\theta}(\vec{x})d\vec{x}$为

$$
\begin{align*}
    &-\int \frac{\partial p(\vec{x})}{\partial x_i}s_{\theta}(\vec{x})d\vec{x}\\
    =& -\int [\int \frac{\partial p(\vec{x})}{\partial x_1}s_{1,\theta}(\vec{x})dx_1]d(x_2,...,x_n)\\
    =& \int[\lim_{a\to\infty,b\to-\infty}[p(a,x_2,...,x_n)s_{1,\theta}(a,x_2,...,x_n)-p(b,x_2,...,x_n)s_{1,\theta}(b,x_2,...,x_n)]-\int\frac{\partial s_{1,\theta}(\vec
    x)}{\partial x_1}p(\vec{x})dx_1]d(x_2,...,x_n)
\end{align*}
$$

此时，我们给出一个前提条件假设，即

$$
\lim_{||\vec{x}||\to\infty}p(\vec{x})s_{\theta}(\vec{x})=0
$$

那么

$$
-\int \frac{\partial p(\vec{x})}{\partial x_i}s_{\theta}(\vec{x})d\vec{x}=\int\frac{\partial s_{i,\theta}(\vec
    x)}{\partial x_i}p(\vec{x})d\vec{x}
$$

到这里，我们折回去把$J(\theta)$整理，最后得到

$$
J(\theta)=\int_{\vec{x}\in\mathbb{R}^n}p(\vec{x})\sum_{i=1}^n [\partial_i s_{i,\theta}(\vec{x})+\frac{1}{2}s_{i,\theta}(\vec{x})^2]d\vec{x}
$$

由于$dy=(\frac{\partial y}{\partial X})^\dag dx$，所以有$dy=tr(dy)$，所以上式化简为

$$
J(\theta)=\mathbb{E}_{p(\vec{x})}[tr(\nabla_{\vec{x}}s_{\theta}(\vec{x}))+\frac{1}{2}||s_{\theta}(\vec{x})||^2_2]
$$

但是有个问题就是$tr(\nabla_{\vec{x}}s_{\theta}(\vec{x}))$不能处理高维问题，也就是当$\vec{x}$维度高的时候，就算不了了。这个实在[3]里面提出来的，但是算是支线，不做详细说明了

## 参考文献

[1] Lyu S. Interpretation and generalization of score matching[J]. arXiv preprint arXiv:1205.2629, 2012.

[2] Hyvärinen A, Dayan P. Estimation of non-normalized statistical models by score matching[J]. Journal of Machine Learning Research, 2005, 6(4).

[3] Song Y, Garg S, Shi J, et al. Sliced score matching: A scalable approach to density and score estimation[C]//Uncertainty in Artificial Intelligence. PMLR, 2020: 574-584.

## 补充知识

### 分布族（Families of Distributions）

当概率分布函数（例如高斯分布）中参数（均值、方差）取不同的值时，构成特定的概率分布，而原始参数不确定的高斯分布就是这些参数固定的分布的分布族。