# Stochastic Differential Equation（SDE）超简介绍

只是介绍一下随机微分方程和常微分方程的关系，至于关于鞅、布朗运动等专业知识不做讨论

## ODE

常微分方程可以表示为

$$
\begin{cases}
\dot{x}(t)=f(x(t)), t>0 \\
x(0)=x_0,
\end{cases}
$$

直观来看就是x的导数是以x为自变量的函数f的值，其中$f:\mathbb{R}^n\rightarrow \mathbb{{R}^n}$，$x(t)\in\mathbb{R}^n$，$t\in\mathbb{R}_+$。

## SDE和反向SDE（reverse-time SDE）[1]

可以理解为给$\dot{x}(t)=f(x(t))$加一个随机扰动项。这时$\dot{x}(t)$就变成一个随机变量$\dot{X}(t)$。SDE表示为

$$
\begin{cases}
\dot{X}(t)=f(X(t))+g(X(t))w(t), t>0 \\
X(0)=X_0,
\end{cases}
$$

其中$g:\mathbb{R}^n\rightarrow \mathbb{M}^{n\times m}$，$w(t)$是m维的随机噪声。这里面牵涉太多数学物理相关的内容，就直接给出形式，具体的推导见[1]。

将上式改写为微分形式，有

$$
\frac{dX(t)}{dt}=f(X(t))+g(x(t))\frac{dw(t)}{dt}\\
dX(t)=f(X(t))dt+g(x(t))dw(t)
$$

因为针对的是consistency model，所以就按照[2]的说法。也就是引入Ito SDE，即

$$
dx(t)=f(x,t)dt+g(t)dw(t)
$$

其中$g(t)$变成了$g():\mathbb{R}\to\mathbb{R}$的一标量函数，称为$x(t)$的扩散参数（diffusion coefficient），$f(,t)$称为$x(t)$的偏移参数（drift coefficient）。$w(t)$是一个布朗运动，满足$w(t+\Delta t)-w(t)\sim\mathcal{N}(0,c^2\Delta t)$，所以有$dw(t)=\sqrt{c^2\Delta t}\epsilon$，其中$\epsilon\sim\mathcal{N}(0,I)$。当$c=1$时，是标准布朗运动。

SDE和reverse-time SDE其实就是扩散过程和逆扩散过程严格证明在[1]中。

## 参考文献

[1] Anderson B D O. Reverse-time diffusion equation models[J]. Stochastic Processes and their Applications, 1982, 12(3): 313-326.

[2] Song Y, Sohl-Dickstein J, Kingma D P, et al. Score-based generative modeling through stochastic differential equations[J]. arXiv preprint arXiv:2011.13456, 2020.