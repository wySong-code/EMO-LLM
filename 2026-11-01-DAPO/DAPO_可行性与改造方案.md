# DAPO 说法核查与改造方案

> 核查对象：一段关于「DAPO 是什么、能不能用、以及它能否成为本项目最高价值点」的说法。
> 核查依据：`2026-10-29_14-21-GRPO/train_grpo.py`、`reward.py` 实际代码 + verl 官方文档 + 本项目实测 rollout 统计。

---

## 一、结论（三句话）

1. **DAPO 的技术描述全部正确**（论文号、时间、四项补丁、ε_low 0.2 / ε_high 0.28 都对）。
2. **「改配置即可跑」不成立**：你不在 verl 上，且 verl 对 Qwen2.5-Omni（视频+音频+Thinker 分支）没有一等支持，社区里的 Omni GRPO 都改了 vLLM rollout 逻辑。迁移成本是「重写一个 RL 训练器」，不是改配置。
3. **「你独立摸到了同一个解法」要降级成「你摸到了同一个问题的前一半」**。你把零方差组识别出来了、也量化了、还做了配比侧缓解，但**没有做重采样补齐**——这恰恰是 DAPO 动态采样的核心动作。诚实说出来 + 补上后半截，比现在这个说法更能打。

---

## 二、逐条核查

| 说法 | 判定 | 依据 |
|---|---|---|
| 全称 Decoupled Clip and Dynamic sAmpling Policy Optimization | ✅ | 准确 |
| arXiv 2503.14476，2025 年 3 月，字节 Seed + 清华 AIR | ✅ | 准确 |
| Clip-Higher：ε_low 0.2 / ε_high 0.28 | ✅ | 与 verl `clip_ratio_low/high` 默认值一致 |
| Dynamic Sampling：过滤全对/全错组，过采样补齐 batch | ✅ | verl `filter_groups.enable` + `max_num_gen_batches`，机制描述准确 |
| Token-level Loss：按 batch 内全部 token 归一 | ✅ | verl `loss_agg_mode: token-mean` |
| Overlong Reward Shaping：超长渐进惩罚 | ✅ | verl `overlong_buffer`，线性 0→penalty_factor |
| 「这就是你踩过的坑」 | ✅ **属实** | 7B 540/600 组零方差、有效组 10%，正是 `filter_groups` 要处理的情形 |
| Clip-Higher「✅ 有用，通用」 | ⚠️ **对本实验基本无效** | 见 §五：ratio 根本到不了 1.28 |
| Token-level / Overlong「⚠️ 用不上」 | ✅ 判断正确 | 短 JSON 输出（30~60 token）+ `format_valid_rate=1.0`，截断从未发生 |
| 「verl 原生支持，改配置即可跑」 | ❌ **对本实验的栈不成立** | 见 §四 |
| 「你独立摸到了同一个解法」 | ⚠️ **只对了 60%** | 见 §三：缺「重采样补齐」这一环 |
| 「这段价值超过项目里任何一项单点指标」 | ✅ 方向对 | 但前提是补上后半截，否则一问就穿 |

---

## 三、现有代码里到底有什么（对着行号说）

`train_grpo.py`：

| 位置 | 实际实现 | 对应 DAPO |
|---|---|---|
| L52-59 `choose_pair` | 每步固定 **1 条冲突 + 1 条对齐**（1:1 分层采样） | 不是动态采样，是**静态配比**，4× 于训练集 12.3% 的冲突占比 |
| L108 `--original-conflict-prob 0.67` | 冲突桶内 67% 抽原始、33% 抽合成 | 对**真实冲突过采样**（自然占比 143/411 = 34.8% → 67%，约 1.9×）。这一条是真的，且很有讲头 |
| L267-277 | 每 prompt 8 候选 | 与 DAPO 的 `rollout.n=8` 同构 |
| L278 `advantages()` | 组内 z-score，零方差返回全 0 | 对应 DAPO 的「无方差」判据 |
| `reward.py` L83-86 | 零方差组打标签 `all_correct` / `all_same_not_perfect` | **识别 → 记账 → 跳过**，不重采样 |
| L296-302 | 全零则整步不更新（`updates=0`） | 缺「补齐」 |
| L305-309 | `ratio.clamp(1-ε, 1+ε)`，单 ε=0.2 | 对称裁剪 = DAPO 要解耦的那一项 |
| L312 | `loss / active_count`，按 response 平均 | response-level，非 token-level |

**结论：你有 DAPO 动态采样的「前半截」（识别 + 丢弃 + 定量），缺「后半截」（重采样把 batch 填满）。**
你把退化组当成"这一步白跑"记录下来了，DAPO 把它当成"换一批题继续跑"。

这个差别在技术评审里是决定性的：说"我做了动态采样"，懂行的人问一句"那你重采样了吗、batch 怎么补齐的"就没了。说"我只做了识别和丢弃，重采样是后来补的"，反而稳。

---

## 四、为什么不是「改配置即可」

| 障碍 | 实情 |
|---|---|
| 现有栈 | 单文件 `train_grpo.py` + transformers + peft，手工模型并行（policy / ref 分设备）。不是 verl，也没有 Ray/FSDP/vLLM |
| verl 的 Omni 支持 | verl 多模态一等支持是 Qwen2.5-VL 这类"图/视频 + 文本"。Qwen2.5-Omni 要走音频 + Thinker 分支 + 自定义 chat template；公开的 Omni GRPO 工作（M3-Agent、Echo）都**改了 `vllm_rollout_spmd.py` 的 `generate_sequences`**，还要做 parquet 转换、FSDP checkpoint 合并 |
| 迁移代价 | 保守 1~2 周，且要重跑一遍 smoke test 与显存探针。vLLM rollout 的权重副本会和 policy 争用同一份设备内存 |
| 收益 | 只有 `filter_groups` 一项真正对症（见 §五） |

**结论：不值得迁 verl。改造你现有 `train_grpo.py` 更快、风险更低，而且改造过程本身就有记录价值。**

---

## 五、四项补丁的实际价值（别照单全收）

| 补丁 | 对你 | 原因 |
|---|---|---|
| **Dynamic Sampling** | ✅ **唯一真正对症** | 7B 540/600 组零方差、300 步里只有 60 步真的更新了 |
| Clip-Higher | ⚠️ **加了不亏，别指望涨分** | ratio = exp(now-old)，lr 2e-6 × 2 个 PPO epoch，ratio 实际落在 1±0.01。`clamp(0.8, 1.28)` 几乎永不触发。DAPO 的 Clip-Higher 是为长 CoT 里少数 token ratio 冲到 1.5+ 设计的，本实验输出只有 30~60 token |
| Token-level Loss | ❌ 跳过 | response 长度高度均匀（严格 JSON），seq-mean 与 token-mean 差值可以忽略 |
| Overlong Shaping | ❌ 跳过 | `format_valid_rate=1.0`，4800 条 rollout 里截断 0 次，从未触发 −0.20 |

**这个取舍本身就是内容**：「我读了 DAPO 四项，逐条对照自己的场景，判定只有动态采样对症、Clip-Higher 顺手加、另两项无关」——这是审读能力，比"我用了 DAPO"值钱。

---

## 六、真正的改法：离线预筛版动态采样（Offline Dynamic Sampling）

### 为什么不能照搬在线重采样

| | 有效组占比 | 每步需抽 prompt | 300 步 rollout 数 | 墙钟 |
|---|---|---|---|---|
| 3B 原版 | 18.8% | 2（不补） | 4,800 | 3h44m |
| 3B + 在线重采样 | 18.8% | ≈10.6 | ≈25,500 | **≈20h** |
| 7B 原版 | 10.0% | 2（不补） | 4,800 | 2h51m |
| 7B + 在线重采样 | 10.0% | ≈20 | ≈48,000 | **≈28.6h** |

（按实测 7B 28 rollout/min、3B 21 rollout/min 折算；verl 的 `max_num_gen_batches=10` 恰好也是 10× 上限。）

**10 倍墙钟不可接受。** 但这里有个关键观察：**预筛只需要 policy，不需要 reference。** 参考模型只在算 old/ref logprob 时才用得上，而"这道题模型能不能做对"只需要 policy 采样一次。

### 方案 A（推荐，4 小时出结果）

1. **离线预筛**：对 3352 条训练 prompt 各采 8 候选，只加载 policy，并行分片。
   - 总量 26,816 rollouts，每路 3,352 条，约 **4 小时**跑完。
2. **构造难度谱**：按组内奖励方差标为 `active` / `all_correct` / `all_wrong`，得到
   - 7B：active ≈ 335 条（10%），degenerate ≈ 3,017 条
   - 3B：active ≈ 630 条（18.8%）
3. **训练时静态采样**：每步从 active pool 抽，有效组占比接近 100%，**零额外在线成本**，300 步里有效更新从 60/300 变成 ~300/300。
4. **防遗忘**：每步仍是 1 冲突 + 1 对齐，但其中一条从 active pool 抽、另一条从全量随机抽（比如 70/30）。否则只在 335 条上训 600 次，过拟合到硬样本。

### 方案 B（改动量参考）

- **Clip-Higher**：`train_grpo.py` L308 一行 →
  `ratio.clamp(1 - args.clip_eps, 1 + args.clip_high)`，新增 `--clip-high 0.28`。
- **在线兜底重采样**：把 `choose_pair` 包成循环，`for attempt in range(args.max_resample)`，两个桶分别重抽（保住 1:1），凑够 2 个 active 组或到上限就停。约 30 行。
- **记账**：`rollouts.jsonl` 增加 `attempt` 字段，方便事后算"每步实际消耗的 rollout 数"——这个数本身就是可写进报告的成本证据。

---

## 七、一个必须补的对照臂

**动态采样会改变有效 batch 的构成，一旦改了，7B 的 −0.0006 就不能再和 3B 的 +0.0097 直接比。** 你现在整个项目的价值建立在"同配方对照组"上，这条链一断，前面的归因全部失效。

所以补 DAPO 时必须同时跑：**3B + 同一套 DAPO 改造**（4 小时，单次 3h 量级）。三个臂才成立：

```
3B 原版    +0.0097 (显著)
3B + DAPO  ?          ← 新增，验证改造本身是否有效
7B 原版    −0.0006
7B + DAPO  ?          ← 主结果
```

---

## 八、两个风险

1. **hard pool 收缩**：7B 的 active pool 只有约 335 条，600 次抽样意味着每条被看 1.8 次，容易在这批题上过拟合。必须混入随机样本（§六 第 4 点）。
2. **标注噪声被放大**：过滤后剩下的"有方差"的组，有一部分方差来自**标注本身不一致**——25 条真实冲突的 gold 里 `pred_emotion` 有 60% 跟画面、40% 跟另一模态，没有固定规则。在这批样本上做梯度等于朝噪声优化。**监控指标：特异度不能掉**（3B+GRPO 已经出现过"F1 单调升、特异度单调降"的取舍，来源就是奖励里 F1 权重 0.35 > 特异度 0.2）。

---

## 九、如果只做一件事

先做 §六 的**离线预筛**（4 小时，8 卡，纯推理，无风险）。它同时产出三样东西：难度池、每条题的"模型是否会做"标签、以及一份可直接画图的难度分布。有了它，动态采样、课程式采样、难度分层报告全都免费，而且**不改任何训练代码就能先写进报告**（"把 3352 条训练题按 8 候选方差分为易/难两层，难样本仅占 10%，据此重设采样策略"）。
