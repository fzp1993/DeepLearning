# 随机微分方程扩展基于分数的生成模型[1]

核心内容就是把SMLD和DDPM的加噪过程使用SDE进行统一，同时使用reverse-time SDE来统一去噪过程，并且更进一步将去噪过程简化为概率流ODE（probability flow ODE，PF ODE）这个PF ODE很重要，一致性模型会用到。个人觉得这篇工作直接将Neural ODE引进来是个不错的想法，可以直接用成熟的模型提高采样的效率，并且还能实现精确的似然计算

## 加噪过程

### SMLD的加噪过程转变为VE SDE

根据之前的笔记，我们已经得到了SMLD的加噪过程，也就是

$$
\tilde{\bold{x}}_i=\bold{x}+\epsilon\sigma_i
$$

其中$\epsilon\sim\mathcal{N}(0,I)$，$p_{\sigma}(\tilde{\bold{x}}|\bold{x})\sim\mathcal{N}(\bold{x},\sigma_i^2 I)$。我们假设

$$
\begin{align*}
    \tilde{\bold{x}}_i&=\bold{x}+\epsilon\sigma_i\\
    \tilde{\bold{x}}_{i-1}&=\bold{x}+\epsilon\sigma_{i-1}
\end{align*}
$$

这里分别服从正态分布$p_{\sigma}(\tilde{\bold{x}}_{i}|\bold{x})\sim\mathcal{N}(\bold{x},\sigma_i^2 I)$和$p_{\sigma}(\tilde{\bold{x}}_{i-1}|\bold{x})\sim\mathcal{N}(\bold{x},\sigma_{i-1}^2 I)$。因此，$\tilde{\bold{x}}_{i}-\tilde{\bold{x}}_{i-1}$服从正态分布$p_{\sigma}(\tilde{\bold{x}}_{i}|\bold{x})\sim\mathcal{N}(0,(\sigma_i^2-\sigma_{i-1}^2) I)$，即

$$
\tilde{\bold{x}}_i=\tilde{\bold{x}}_{i-1}+\epsilon_{i-1}\sqrt{\sigma_i^2-\sigma_{i-1}^2}, i=1,...,N
$$

$\{\tilde{\bold{x}}_i\}_{i=1}^N$实际上是马尔科夫链。当$N\to\infty$时，马尔可夫链变成一种连续的随机过程，也就是$\{\tilde{\bold{x}}(t)\}_{t=1}^N$，其中$t\in[1,N]$。并且噪声系数$\{\sigma_i\}_{i=1}^N$也改写为$\{\sigma(t)\}_{t=1}^N$，$\epsilon_{i}$改写为$\epsilon(t)$，为了和论文[1]一致，写为$z(t)$。将上式改写，有

$$
\begin{align*}
    &\tilde{\bold{x}}(t+\Delta t)=\tilde{\bold{x}}(t)+z(t)\sqrt{\sigma^2(t+\Delta t)-\sigma^2(t)}\\
    &\frac{\tilde{\bold{x}}(t+\Delta t)-\tilde{\bold{x}}(t)}{\Delta t}\Delta t= \sqrt{\frac{\sigma^2(t+\Delta t)-\sigma^2(t)}{\Delta t}\Delta t}z(t)\\
\end{align*}
$$

根据导数的极限定义，当$\Delta t\to 0$时，有

$$
d\tilde{\bold{x}}(t)=\sqrt{\frac{d\sigma^2(t)}{dt}}dw(t)
$$

这里是我们把标准布朗运动带入的结果。可以看出，这是一个偏移参数$f(\tilde{\bold{x}}(t),t)$为0，扩散参数$g(t)$为$\sqrt{\frac{d\sigma^2(t)}{dt}}$的SDE。

### DDPM的加噪过程转变为VP SDE

根据之前的笔记，我们已经得到了DDPM的加噪过程，也就是

$$
\tilde{\bold{x}}_i=\sqrt{1-\beta_i}\tilde{\bold{x}}_{i-1}+\sqrt{\beta_i}z_{i-1}, i=1,...,N
$$

其中$z_{i-1}\sim\mathcal{N}(0,I)$。同样，当$N\to\infty$时，有

$$
\tilde{\bold{x}}(t+\Delta t)=\sqrt{1-\beta(t+\Delta t)\Delta t}\tilde{\bold{x}}(t)+\sqrt{\beta(t+\Delta t)\Delta t}z(t)
$$

根据三角不等式，有

$$
\begin{align*}
    \tilde{\bold{x}}(t+\Delta t)&\approx \tilde{\bold{x}}(t)-\frac{1}{2}\beta(t+\Delta t)\Delta t\tilde{\bold{x}}(t)+\sqrt{\beta(t+\Delta t)\Delta t}z(t)\\
    &\approx \tilde{\bold{x}}(t)-\frac{1}{2}\beta(t)\Delta t\tilde{\bold{x}}(t)+\sqrt{\beta(t)\Delta t}z(t)
\end{align*}
$$

根据导数的极限定义和标准布朗运动，有

$$
d\tilde{\bold{x}}(t)=-\frac{1}{2}\beta(t)\tilde{\bold{x}}(t)dt+\sqrt{\beta(t)}dw(t)
$$

可以看出这是一个偏移参数为$-\frac{1}{2}\beta(t)\tilde{\bold{x}}(t)$，扩散参数为$\sqrt{\beta(t)}$的SDE。

还有一个衍生版本sub-VP SDE，具体证法在论文[1]中，他的结果是

$$
d\tilde{\bold{x}}(t)=-\frac{1}{2}\beta(t)\tilde{\bold{x}}(t)dt+\sqrt{\beta(t)(1-e^{-2\int_0^t \beta(s)ds})}dw(t)
$$

### 加噪过程总结

至此得到了三种基于SDE的加噪过程的概率分布，依次是VE SDE、VP SDE和sub-VP SDE

$$
p_{0t}(\tilde{\bold{x}}(t)|\tilde{\bold{x}}(0))=
\begin{cases}
    \mathcal{N}(\tilde{\bold{x}}(t);\tilde{\bold{x}}(0),[\sigma^2(t)-\sigma^2(0)]I)\\
    \mathcal{N}(\tilde{\bold{x}}(t);\tilde{\bold{x}}(0)e^{-\frac{1}{2}\int_0^t \beta(s)ds},I-Ie^{-\int_0^t \beta(s)ds})\\
    \mathcal{N}(\tilde{\bold{x}}(t);\tilde{\bold{x}}(0)e^{-\frac{1}{2}\int_0^t \beta(s)ds},[1-e^{-\int_0^t \beta(s)ds}]^2 I)
\end{cases}
$$

并且，训练过程时，参数的更新表示为

$$
\bold{\theta}^*=\mathop{\arg\min}\limits_{\theta}\mathbb{E}_t\{\lambda(t)\mathbb{E}_{\bold{x}(0)}\mathbb{E}_{\tilde{\bold{x}}(t)|\bold{x}(0)}[||s_{\theta}(\tilde{\bold{x}}(t),t)-\nabla_{\tilde{\bold{x}}(t)}logp_{0t}(\tilde{\bold{x}}(t)|\tilde{\bold{x}}(0))||^2]\}
$$

## 去噪过程

由于加噪过程是一个SDE过程，那么去噪过程可以使用一个reverse-time SDE来表示。这里用到了Fokker-Planck equation的内容，这段以后再补，先给出一个结论，也就是去噪的reverse-time SDE表示为

$$
d\tilde{\bold{x}}(t)=[f(\tilde{\bold{x}}(t),t)-\frac{1}{2}(g(t)^2-\sigma^2(t))\nabla_{\tilde{\bold{x}}(t)}logp_t(\tilde{\bold{x}}(t))]dt+\sigma(t)dw(t)
$$

当$\sigma(t)=0$时，上式就是一个ODE，即

$$
d\tilde{\bold{x}}(t)=[f(\tilde{\bold{x}}(t),t)-\frac{1}{2}g(t)^2\nabla_{\tilde{\bold{x}}(t)}logp_t(\tilde{\bold{x}}(t))]dt
$$

在[2]中，利用电磁理论改进了ODE过程，提升了效果。

## 结果

现在可以将DDPM和SMLD的加噪过程和去噪过程一般化的表示为

$$
\begin{cases}
    d\tilde{\bold{x}}(t)=f(\tilde{\bold{x}}(t),t)dt+g(t)dw(t)\\
    d\tilde{\bold{x}}(t)=[f(\tilde{\bold{x}}(t),t)-\frac{1}{2}g(t)^2 s_\theta(\tilde{\bold{x}}(t),t) dt
\end{cases}
$$

## 参考文献

[1] Song Y, Sohl-Dickstein J, Kingma D P, et al. Score-based generative modeling through stochastic differential equations[J]. arXiv preprint arXiv:2011.13456, 2020.

[2] Xu Y, Liu Z, Tegmark M, et al. Poisson flow generative models[J]. arXiv preprint arXiv:2209.11178, 2022.