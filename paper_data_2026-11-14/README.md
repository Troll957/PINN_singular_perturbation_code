# paper_data_2026-11-14

`manuscript.tex` 中全部表格数值及其来源。

## 结构

```
as_published.json     论文 7 张表的全部数值（已逐格核对）
raw_controls/         Case 2/3 对照组（408 000 步）原始 JSON ×8
raw_ablation/         Case 2 六配置消融原始 JSON ×6
case1/                Case 1 自适应+对照行
case4/                Case 4 自适应+对照行
case5/                Case 5 反源：自适应/对照 history + 最终点集
```

## 源路径（服务器 `Djjiang:/home/test1/PINN_singular_perturbation/`）

| 表 | adaptive | fixed |
|---|---|---|
| Table 2 Case 1 | `spp_pinn_package/空间1d/eps_full/case1_eps_rows.json` | 同文件 |
| Table 3 Case 2 | `spp_pinn_package/空间2d/eps_full/case2_eps_rows.json` | `空间2d/controls/ctrl_case2_*.json` |
| Table 4 Case 3 | `spp_pinn_package/空间2d/eps_full/case3_eps_rows.json` | `空间2d/controls/ctrl_case3_*.json` |
| Table 5 Case 4 | `case4_nonrect/runs/paper/case4_eps_rows.json` | 同文件 |
| Table 6 Case 5 | `case5/runs/e4_big_adapt/` | `case5/runs/e4_big_fixed/` |
| Table 7 消融 | `spp_pinn_package/空间2d/ablation/ab_*.json` | — |

## 对照组已全部重跑

旧表的 "fixed" 列只训练 8 000 步（脚本 `rounds=0` 使总步数退化为 `1×epochs`），
与自适应的 408 000 步不可比。现对照组训练 51 相 × 8 000 步，调度一致。

| 表 | 旧比值 | 新比值 |
|---|---|---|
| Case 2 (L²) | 11.5 / 11.0 / 11.0 / 5.0 | 1.85 / 3.28 / 4.02 |
| Case 3 (L²) | 117 / 113 / 209 / 109 | 52 / 64 / 44 / 24 |
| Case 4 (L²) | 0.80 / 1.26 | 0.80 / 1.26（原文即满预算，未重跑） |

## 核对结果

7 张表逐格与源数据比对（容差 2%），全部一致。核对中修正：

1. Table 6 的 $s_0$ 比值 4.0 → **4.8**，$v$ 1.6 → **1.8**（原值由取整后的百分数算出，
   与 $A$ 的口径不一致）
2. Case 5 正文倍数范围 1.6–4.1 → **1.8–4.8**
3. Case 2 表删去 ε=10⁻⁶ 一行，正文比值范围 1.85–28.7

## Case 5 点集

`case5/case5_adapt_data.npz` 的 `points` 字段是最终的 **7048 个自适应内点**，
论文 Figure 8 第四格即由此投影得到。旧的 `paper_figs/data/case5_fields.npz`
曾用它之前的 2048 个初始 Halton 点，已修正。
