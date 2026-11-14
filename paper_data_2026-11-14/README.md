# paper_data_2026-11-14

本目录保存 manuscript.tex 中全部表格数值及其来源。

## 文件

- `as_published.json` —— 论文 7 张表的全部数值（已逐格与源数据核对）
- `raw_controls/` —— Case 2/3 对照组（408 000 步）的原始运行 JSON，8 个
- `raw_ablation/` —— Case 2 六配置消融的原始运行 JSON，6 个

## 数据来源（服务器 Djjiang:/home/test1/PINN_singular_perturbation/）

| 表 | 内容 | 源路径 |
|---|---|---|
| Table 2 | Case 1 边界层 | `spp_pinn_package/空间1d/eps_full/case1_eps_rows.json` |
| Table 3 | Case 2 角层 | adaptive: `空间2d/eps_full/case2_eps_rows.json`；fixed: `空间2d/controls/ctrl_case2_*.json` |
| Table 4 | Case 3 移动层 | adaptive: `空间2d/eps_full/case3_eps_rows.json`；fixed: `空间2d/controls/ctrl_case3_*.json` |
| Table 5 | Case 4 曲边界 | `case4_nonrect/runs/paper/case4_eps_rows.json` |
| Table 6 | Case 5 反源 | `case5/runs/e4_big_adapt/history.json`、`e4_big_fixed/history.json` |
| Table 7 | 消融 | `空间2d/ablation/ab_*.json` |

## 关键说明

**对照组已全部重跑。** 旧表的 "fixed" 列只训练了 8 000 步（脚本 bug：`rounds=0`
导致总步数 `(rounds+1)×epochs` 只有一相），与自适应的 408 000 步不可比。
现在的对照组训练 51 相 × 8 000 = 408 000 步，调度与自适应一致，只是不加点。

| 表 | 旧比值 | 新比值 |
|---|---|---|
| Case 2 (L2) | 11.5 / 11.0 / 11.0 / 5.0 | 1.85 / 3.28 / 4.02 |
| Case 3 (L2) | 117 / 113 / 209 / 109 | 52 / 64 / 44 / 24 |
| Case 4 (L2) | 0.80 / 1.26 | 0.80 / 1.26（未重跑，原文即满预算） |

## 核对方式

`as_published.json` 的每个数值都从上述源文件读出后与 `manuscript.tex`
的表格逐格比对，容差 2%。核对结果：7 张表全部一致。

核对中发现并已修正的问题：
1. Table 6 的 $s_0$ 比值应为 4.8（原写 4.0），$v$ 应为 1.8（原写 1.6）
   —— 原值由四舍五入后的百分数算出，与 $A$ 的口径不一致。
2. Case 5 正文的倍数范围随之改为 1.8–4.8。
3. Case 2 表已删去 $\varepsilon=10^{-6}$ 一行，正文比值范围相应为 1.85–28.7。

## 待补

Case 1/4/5 的原始 JSON 在服务器离线时未能取回，`as_published.json`
中的数值取自已核对的运行输出。服务器恢复后可从上表路径补齐原始文件。
