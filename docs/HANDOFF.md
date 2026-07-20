# 交接 Prompt

在 `C:\Users\Mio\Documents\9444\plant-disease-detection` 打开 Claude Code,把下面
分隔线以下的内容整段贴进去。`CLAUDE.md` 会被自动加载,不用手动贴。

---

我在完成 COMP9444 25T1 **Project 090 — Automatic Plant Disease Detection Using
Computer Vision**(四人组队)。请先读 `CLAUDE.md`(项目全貌、目录约定、已知的坑)
和 `docs/EXPERIMENTS.md`(假设 H1–H9、实验矩阵 E1–E18、**预先写好的证伪条件**)。

## 项目在做什么

课程 brief 要求一个 CNN 做两件事:判断叶片健康/患病,以及从图像特征估计病害严重度,
目的是"帮助农民和农业专家及时干预"。

关键在于:这个目的是一个**部署主张**。brief 指向的 PlantVillage 数据集是 54,305 张
摄影棚照片 —— 单片摘下的叶子、平铺、纯色背景、统一打光。任何现代 CNN 在上面都能超过
99%,但这个数字回答不了 brief 自己提出的问题。所以工作拆成三问:

- **RQ1** 能否在 PlantVillage 上准确分类?(brief Task 1)
- **RQ2** 这个准确率是否意味着它对 brief 描述的那个农民真的有用,即在真实田间照片上?
- **RQ3** 严重度能否从图像特征估计,估计是否可信?(brief Task 2)

RQ2 不是超纲,它检验 brief 自己的目标有没有达成。

**数据**:PlantVillage 三个变体(color / grayscale / segmented,同一批叶子,索引级
对齐,seed 42,70-15-15);PlantDoc 2,525 张真实田间照片,映射到 27 类,**从不参与
训练**,只作外部验证。

**模型**:三个 baseline(从零训练的 CustomCNN、ResNet-18、MobileNet-V2)+ 软投票
**集成**(权重在验证集上网格搜索,不碰测试集)。后续臂加了背景随机化、冻结骨干探针、
层级"先作物后病害"分类头。

**项目的论点不是"我们拿到 99.8%"**,而是"**99.8% 主要是采集偏置**",并且用对照实验
证明而非断言;真正诚实的部署叙事是域适应。

## 已有结果

零样本(PlantVillage 训练 → PlantDoc 测试):

| 设置 | PlantVillage | PlantDoc |
|---|---|---|
| Baseline 集成(128px) | 99.84 | 17.37(236)/ 14.61(2525) |
| 强增广(E8 对照) | 99.52 | 23.73 |
| 背景随机化 p=0.7(E8) | 99.26 | 24.15 |
| 训练于 segmented(E9) | 99.24 | 20.34 |
| 训练于 grayscale(E10) | 98.48 | **8.90** |
| **冻结骨干(E15 对照)** | **91.70** | **24.15** |

域适应(**必须与零样本分表,不共享表格行**):`ft_robust` 20-shot,24.15 → **48.73**。

审计探针:E3 只用 8 个背景边缘像素分类 **31.6%**(随机 2.6%);E5 Grad-CAM 落在叶内
61.3% vs 叶面积基线 47.5%;E13 严重度 ROC-AUC 0.868(官方掩膜)/ 0.767(Otsu)。

## 三个核心发现

**1. E15 —— 574 倍参数,田间零收益。** 冻结骨干只训 19,494 个参数(0.17%),得到
PlantVillage 91.70% / PlantDoc 24.15%;全量微调 11.2M 参数,得到 99.52% / 23.73%。
多训的 7.8 个实验室分数在真实世界价值为零。这是最强的**直接**对照(E3/E5 是间接
证据)。注意 H8 原本预测冻结会**更好**,实际是**打平**,已记为部分证伪 —— 打平其实
更有力,它把那 7.8 分干净地定性为基准专属。

**2. E10 —— 去掉颜色,实验室只掉 1 分,田间掉 15 分。** 说明 PlantVillage 上 98.5%
的模型可以完全不看颜色,而颜色恰恰是病害诊断的依据(病斑的黄、褐、白)。这个基准
区分不出"真在看病征"和"在看别的东西"的模型。

**3. 误差分解 —— 瓶颈是认作物,不是认病。** 24.15% = 作物 46.19% × 判对作物后判病
52.29%(乘积精确吻合)。健康/患病二分类有 74.58%。增广修不了这个:PlantVillage 里
根本没有"整株、多叶、远景"的样本,增广变不出来。

## 待办

一个窗口在跑 `finetune_full` + `sweep_shots_5/50/100`(E18 适应曲线)。剩下的:

```powershell
.\push.ps1        # 顺序跑完剩余 9 个 stage,约 3 小时
```

含 `train_hier`(E17,唯一还能推高零样本的方向)、`finetune_aug_only`(E11 关键
对照)、`train_frozen_bg_random`(补完 E15 的 2×2)、`finetune_baseline`,以及
5 个 `eval_arm_*`(输出作物/病害/受限标签空间的分解)。

**跑完后需要判断而非仅执行的三点:**

1. **`finetune_aug_only` 决定 E11 怎么写。** `ft_robust`(48.73)和 `ft_baseline`
   的起点差了三个变量(224 vs 128、strong vs standard、p=0.7 vs 0),**不能**把差距
   归给背景随机化。`finetune_aug_only` 只差 `p_random` 一个变量,E11 的结论只能建在
   它上面。
2. **`train_hier`(E17)** 若把作物准确率推到 60%,按分解式总数约 31%;若没提升,
   照实记 H9 被证伪,不要改证伪条件。
3. **E8 目前是负面结果**:背景随机化相对强增广只有 +0.42,远在 n=236 的 ±5.5pp 区间
   内。`eval_arm_*` 在全部 2,525 张上重评会把区间收紧到约 ±1.7,届时才能定论。

**两个等 GPU 空闲再做的小改动**(绝不能在训练运行时改 `src/` 或 `scripts/`):

- 套用 float32 补丁:`np.random.normal` 返回 float64,使 `random_background` 每次
  多分配 4 倍内存(4.69 → 1.13 MiB)。补丁已备好在 outputs 的 `pending/augment.py`。
- `scripts/finetune.py` 补一个 `--num-workers`。

**最后交付**:数出齐后写报告(.docx)和 PPT(**必须用 UNSW 提供的模板原样填充**),
并最后执行
`uv run jupyter nbconvert --to notebook --execute --inplace notebooks\plant_disease_detection.ipynb`
(评分要求 notebook 带可见输出)。团队还需完成 `outputs/severity_annotations.csv`
的 150 张人工分级(四人分),然后 `scripts/audit.py --probe severity-validate` 出 E14。

## 工作方式要求

- **花 GPU 时间之前先验证**:`--dry-run` 看命令展开 → 单元检查新机制(参数量、
  BatchNorm 模式、分布是否和为 1、梯度是否有限)→ 微型合成数据集冒烟测试 → 确认已有
  tag 未变(缓存 checkpoint 不能失效)。一个臂 30–60 分钟,错了就是白烧。
- **注释只解释为什么,不解释做了什么**,保持稀疏,不要 AI 痕迹。
- 任何新实验都要挂到 `docs/EXPERIMENTS.md` 的某个假设上。没有假设、没有对照的臂
  不进矩阵。
- **让对照配对**。下结论前先检查两个臂之间是否只差一个变量。
- 结果不如预期就照实写成负面结果。这个项目里负面结果(E7、E8、H8)恰恰是论证最有力
  的部分。
