## 工程概览

本工程基于 Qlib，覆盖「特征抽取 → 多模型训练 → 信号融合 → 组合回测」全流程。核心模块：

- `feature/qlib_feature_pipeline.py`：封装 Qlib 数据加载、标准化与标签对齐。
- `trainer/trainer.py`：按滚动窗口训练 LGB、MLP、Stack，并记录窗口 IC。
- `models/ensemble_manager.py`：使用 Qlib `AverageEnsemble` 聚合多模型预测。
- `predictor/predictor.py`：加载模型、执行预测、结合 IC 动态权重给出最终信号。
- `portfolio/portfolio_builder.py` + `run_backtest.py`：生成受限组合，并用标签收益回测。

下文重点解释“信号强弱解读”和“回测”两部分，其他模块简述行程参考 README。

---

## 信号强弱解读

### 1. 预测文件结构

`run_predict.py` 会在 `data/predictions/pred_<tag>_<start>_<end>.csv` 输出以下列：

- `final`：IC 动态加权后的综合信号，是工程推荐的交易输入；
- `lgb`、`mlp`、`stack`：单模型预测；
- `qlib_ensemble`：Qlib 平均融合输出（若启用）。

文件索引为 `["datetime", "instrument"]`，单期信号越高，代表相对收益预期越高。若希望获知单模型表现，可直接读取对应列或结合日志（下节）。

### 2. IC 动态权重来源

训练阶段在 `trainer/train` 中，对每个验证窗口记录 4 组 rank-IC：

```105:155:trainer/trainer.py
metric = {
    "ic_lgb": _rank_ic(valid_pred, valid_lbl),
    "ic_mlp": _rank_ic(mlp_valid_pred, valid_lbl),
    "ic_stack": _rank_ic(stack_valid_pred, valid_lbl),
    "ic_qlib_ensemble": _rank_ic(valid_blend, valid_lbl),
}
```

`run_predict.py` 通过 `_load_ic_histories` 读取上述 IC 序列，交给 `RankICDynamicWeighter` 计算 IC-IR 值，并在 `blend` 中对信号做加权。默认设置：

- 半衰期 20 日、滚动窗口 60 日；
- 权重区间 [0.05, 0.7]，并可裁掉负 IC。

因此，若近期某模型 IC 走弱，其权重会被压低，`final` 信号自动弱化；反之则增强。

### 3. 如何量化信号强弱

1. **综合信号**：`final` 列值越大，代表多模型共识越强，可直接作为排序依据。
2. **模型分歧**：比较 `lgb`/`mlp`/`stack`/`qlib_ensemble`，判断是否一致；若 `qlib_ensemble` 与单模型差异大，说明融合策略（标准化+均值）对信号做了显著修正。
3. **历史表现**：查看 `data/logs/training_metrics.csv`，关注最新窗口的 `ic_*`，即可了解某模型近期预测力，辅助判读信号可信度。

如需可视化，可将预测期信号与当期真实收益合并，计算 rank-IC 或收益分层胜率，快速验证“强 signal 是否对应更高收益”。

---

## 回测流程详解

### 1. 输入数据

`run_backtest.py` 需要：

1. `--prediction`: 预测 CSV（包含 `final` 列）；
2. `config/pipeline.yaml`: 内含数据配置、组合约束；
3. 可选 `--industry`: `instrument,industry` 映射，用于行业敞口限制；
4. 标签收益：由 `QlibFeaturePipeline` 重新构建，与预测同指数对齐。

### 2. 组合构建 (`PortfolioBuilder`)

`portfolio/portfolio_builder.py` 根据参数：

- `max_position`: 单期持仓上限（如 0.3 表示最多 30% 仓位参与）；
- `max_stock_weight`: 单只股票权重上限；
- `max_industry_weight`: 行业权重上限；
- `top_k`: 每期挑选的股票数量。

算法流程：

1. 对 `final` 信号排序；
2. 取前 `top_k`；
3. 按信号强度归一化为初始权重；
4. 依次裁剪单股/行业超限部分，并按剩余股票再归一；
5. 若 `max_position < 1`，整体乘以该比例，保留剩余现金头寸。

返回的 `weights` 与 `label` 对齐，用于计算当期收益。

### 3. 回测逻辑

`run_backtest.py` 主循环（伪代码）：

```62:95:run_backtest.py
for dt in sorted(pred_dates):
    score = preds.xs(dt)
    label_slice = labels.xs(dt)
    industry_slice = industry_map.reindex(score.index) if provided
    weights = builder.build(score, industry_slice, top_k)
    realized = label_slice.reindex(weights.index).dropna()
    ret = (weights.loc[realized.index] * realized).sum()
    results.append({"date": dt, "return": ret})
```

即：按每个交易日的信号建组合，用真实标签收益乘以权重求当期收益；若标签缺失或无交集则跳过。

### 4. 绩效指标

生成的 `results` 会写到 `data/backtest/backtest_result.csv`，包含：

- `return`: 单期收益；
- `cum_return`: 累计收益（`(1 + ret).cumprod() - 1`）。

同时打印统计：

- `total_return`: 最终累计收益；
- `avg_return`: 单期平均收益；
- `volatility`: 单期收益标准差；
- `sharpe`: `avg / vol * sqrt(252)`，简单假设日频，并加 `1e-8` 防止除零。

如需扩展，可在该脚本中追加最大回撤、信息比率等指标，或将结果导入 Jupyter 做更细致的分析。

---

## 建议与扩展

1. **信号解释增强**：可在 `predictor/predictor.py` 中保存各模型权重，方便回测时分析“信号强弱 = 值 * 权重”。
2. **多级排序验证**：对预测输出做分层回测（如按 `final` 五分位），衡量强弱分位的盈利能力。
3. **回测细化**：
   - 替换标签收益为实际行情数据（如 T+1 收益），构建更真实的交易级回测；
   - 引入手续费、滑点、换手约束。
4. **配置管理**：利用 `config/pipeline.yaml -> ensemble/models` 增删模型，记得同步更新 IC 日志，以便信号解释时覆盖到新的模型名称。 

