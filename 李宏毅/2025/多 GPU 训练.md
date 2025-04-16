
https://zhuanlan.zhihu.com/p/648924115


**Attention 层单步中间激活值显存表**

| 计算步骤                                  | 中间激活                   | 形状               | 占用显存             |
| ------------------------------------- | ---------------------- | ---------------- | ---------------- |
| $Q/K/V = xW_{Q/K/V}$                  | $x$                    | $[b, s, h]$      | $2bsh$           |
| $QK^T$                                | $Q, K$                 | $[b, a, s, h/a]$ | $2 \cdot 2bsh$   |
|                                       |                        | $[b, a, h/a, s]$ |                  |
| $\text{score} = \text{softmax}(QK^T)$ | $QK^T$                 | $[b, a, s, s]$   | $2bs^2a$         |
| $\text{dropout()}$                    | $\text{dropout\_mask}$ | $[b, a, s, s]$   | $bs^2a$          |
| $x = \text{score} \cdot V$            | $\text{score V}$       | $[b, a, s, s]$   | $2bs^2a + 2bsh$  |
|                                       |                        | $[b, a, h/a, s]$ |                  |
| $xW_o$                                | $x$                    | $[b, a, s, h/a]$ | $2bsh$           |
| $\text{dropout()}$                    | $\text{dropout\_mask}$ | $[b, s, h]$      | $bsh$            |
| $\text{layernorm()}$                  | $x$                    | $[b, s, h]$      | $2bsh$           |
| $\text{SUM}$                          |                        |                  | $13bsh + 5bs^2a$ |

**MLP 层单步中间激活值显存表**

| 计算步骤                          | 中间激活          | 形状            | 占用显存    |
|---------------------------------|-----------------|---------------|----------|
| $xW_{\text{up}}$              | $x$           | $[b, s, h]$ | $2bsh$  |
| $f_{\text{gelu}}(xW_{\text{up}})$ | $xW_{\text{up}}$ | $[b, s, 4h]$ | $8bsh$  |
| $xW_{\text{down}}$            | $x$           | $[b, s, 4h]$ | $8bsh$  |
| $\text{dropout}$              | $\text{dropout\_mask}$ | $[b, s, h] \text{ int}$ | $bsh$   |
| $\text{layernorm}$            | $x$           | $[b, s, h]$ | $2bsh$  |
| $\text{Sum}$                  |                 |               | $21bsh$ |

