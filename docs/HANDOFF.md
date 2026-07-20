# 交接 Prompt — 贴进 Claude Code

在 `C:\Users\Mio\Documents\9444\plant-disease-detection` 打开 Claude Code,把下面整段贴进去。

---

我在做 COMP9444 25T1 Project 090(植物病害检测)。先读 `CLAUDE.md` 和
`docs/EXPERIMENTS.md`,那里有假设、控制变量和**预先写好的证伪条件**。

## 项目论点

不是"我们在 PlantVillage 上拿到 99.8%",而是"**99.8% 主要是采集偏置**",并且要
用对照实验证明,不是断言。每个实验室数字必须配一个田间数字。

## 已有结果

零样本(PlantVillage 训练 → PlantDoc 测试):

| 设置 | PlantVillage | PlantDoc |
|---|---|---|
| Baseline 集成(128px) | 99.84 | 17.37(236 张)/ 14.61(2525 张) |
| 强增广(E8 对照) | 99.52 | 23.73 |
| 背景随机化 p=0.7(E8) | 99.26 | 24.15 |
| 训练于 segmented(E9) | 99.24 | 20.34 |
| 训练于 grayscale(E10) | 98.48 | **8.90** |
| **冻结骨干(E15 对照)** | **91.70** | **24.15** |

域适应(**必须与零样本分开报告**):`ft_robust` 20-shot,24.15 → **48.73**

审计探针:E3 背景像素探针 31.6%(随机 2.6%);E5 Grad-CAM 落在叶内 61.3% vs 叶面积
47.5%;E13 严重度 AUC 0.868。

## 三个核心发现

1. **E15 —— 574 倍参数,田间零收益。** 冻结骨干只训 19,494 个参数(0.17%),
   PlantVillage 91.70% / PlantDoc 24.15%;全量微调 11.2M 参数,99.52% / 23.73%。
   多训的 7.8 个实验室分数在真实世界价值为零。这是最强的直接对照。
   注意:H8 原本预测冻结会**更好**,实际是**打平**,已记为部分证伪,不要改。

2. **E10 —— 去掉颜色,实验室只掉 1 分,田间掉 15 分。** 说明 PlantVillage 上
   98.5% 的模型可以完全不看颜色,而颜色正是病害诊断的依据。基准无法区分
   "真在看病征"和"在看别的"。

3. **误差分解 —— 瓶颈是认作物不是认病。** 24.15% = 作物 46.19% × 给定作物后
   判病 52.29%。健康/患病二分类有 74.58%。增广修不了这个,因为 PlantVillage
   里根本没有"整株、多叶、远景"的样本。

## 待办

```powershell
.\push.ps1        # 顺序跑完剩余 9 个 stage,约 3 小时
```

包含:`train_hier`(E17 层级双头,唯一还能推高零样本的方向)、
`finetune_aug_only`(E11 关键对照)、`train_frozen_bg_random`(补完 E15 的 2×2)、
`finetune_baseline`,以及 5 个 `eval_arm_*`(输出作物/病害/受限标签空间的分解)。

另有一个窗口在跑 `finetune_full` + `sweep_shots_5/50/100`(E18 适应曲线)。

跑完之后:

1. **`finetune_aug_only` 决定 E11 怎么写。** `ft_robust`(48.73)和 `ft_baseline`
   的起点差了三个变量(224 vs 128、strong vs standard、p=0.7 vs 0),**不能**把
   差距归给背景随机化。`finetune_aug_only` 只差 `p_random` 一个变量。
2. **`train_hier` 若把作物准确率推到 60%,总数约 31%**(按分解式估)。若没提升,
   照实记 H9 被证伪。
3. 两个待处理的小改动(等 GPU 空闲再做,**绝不能在训练时改源文件**):
   - 套用 float32 补丁:`np.random.normal` 返回 float64,使 `random_background`
     每次多分配 4 倍内存(4.69 → 1.13 MiB)。补丁在 outputs 的 `pending/augment.py`。
   - `scripts/finetune.py` 补一个 `--num-workers`。
4. 数出齐后写**报告(.docx)+ PPT(必须用 UNSW 提供的模板原样填充)**,并最后执行
   `uv run jupyter nbconvert --to notebook --execute --inplace notebooks\plant_disease_detection.ipynb`
   (评分要求 notebook 带输出)。

## 工作方式要求

- 花 GPU 时间之前先验证:`--dry-run` → 单元检查新机制 → 微型数据集冒烟测试 →
  确认已有 tag 未变(缓存的 checkpoint 不能失效)。
- 注释只解释**为什么**,不解释**做了什么**,保持稀疏。
- 任何新实验都要在 `configs/experiments.yaml` 里挂到 `docs/EXPERIMENTS.md` 的某个
  假设上。没有假设、没有对照的臂不进矩阵。
- 结果不如预期时照实写成负面结果,不要事后修改证伪条件。
