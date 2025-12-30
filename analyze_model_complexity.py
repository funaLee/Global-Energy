"""
Model Complexity Analysis: Linear vs Hybrid
============================================
Calculate number of parameters and compare performance for trade-off analysis.
"""

import joblib
import json
import numpy as np

print("=" * 70)
print("MODEL COMPLEXITY ANALYSIS: Linear vs Hybrid")
print("=" * 70)

# ===========================
# 1. LOAD MODELS
# ===========================
print("\n[1] Loading saved models...")

# Global Linear Model
lr_global = joblib.load('models/global_ridge_model.pkl')
print(f"  ✅ Global LR loaded")

# Hybrid Model
lr_hybrid = joblib.load('models/hybrid_lr_model.pkl')
xgb_hybrid = joblib.load('models/hybrid_xgb_residual_model.pkl')
print(f"  ✅ Hybrid LR + XGBoost loaded")

# ===========================
# 2. CALCULATE PARAMETERS
# ===========================
print("\n[2] Calculating parameters...")

# Linear Model: coefficients + intercept
lr_global_params = len(lr_global.coef_) + 1  # +1 for intercept
print(f"  Global LR: {lr_global_params:,} parameters (coef + intercept)")

# Hybrid LR
lr_hybrid_params = len(lr_hybrid.coef_) + 1
print(f"  Hybrid LR: {lr_hybrid_params:,} parameters")

# XGBoost: Count trees * leaves per tree
# XGBoost có n_estimators trees, mỗi tree có tối đa 2^max_depth leaves
n_trees = xgb_hybrid.n_estimators
max_depth = xgb_hybrid.max_depth
# Mỗi leaf chứa 1 prediction value, mỗi internal node chứa split info
# Số leaves tối đa = 2^max_depth, số internal nodes = 2^max_depth - 1
max_leaves_per_tree = 2 ** max_depth
max_internal_nodes = max_leaves_per_tree - 1
# Mỗi internal node: 1 feature index + 1 threshold = 2
# Mỗi leaf: 1 prediction value
xgb_params_per_tree = max_internal_nodes * 2 + max_leaves_per_tree
xgb_total_params = n_trees * xgb_params_per_tree
print(f"  Hybrid XGB: ~{xgb_total_params:,} parameters ({n_trees} trees × {xgb_params_per_tree} per tree)")

# Total Hybrid
hybrid_total = lr_hybrid_params + xgb_total_params
print(f"  Hybrid Total: ~{hybrid_total:,} parameters")

# ===========================
# 3. LOAD PERFORMANCE METRICS
# ===========================
print("\n[3] Performance comparison (from report)...")

# Metrics from Final_Project_Report.md Section 6.3
metrics = {
    'Global LR': {'R2': 0.9993, 'Median_MAPE': 50.08, 'Params': lr_global_params},
    'Hybrid (LR+XGB)': {'R2': 0.9992, 'Median_MAPE': 19.99, 'Params': hybrid_total}
}

print(f"\n{'Model':<20} {'R²':<10} {'Median MAPE':<15} {'Parameters':<15}")
print("-" * 60)
for model, m in metrics.items():
    print(f"{model:<20} {m['R2']:.4f}     {m['Median_MAPE']:.2f}%          {m['Params']:,}")

# ===========================
# 4. TRADE-OFF ANALYSIS
# ===========================
print("\n" + "=" * 70)
print("TRADE-OFF ANALYSIS")
print("=" * 70)

param_ratio = hybrid_total / lr_global_params
mape_improvement = (50.08 - 19.99) / 50.08 * 100

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPLEXITY vs PERFORMANCE                         │
├─────────────────────────────────────────────────────────────────────┤
│  Parameter Increase:  {param_ratio:.0f}x  ({lr_global_params:,} → {hybrid_total:,})           │
│  MAPE Improvement:    {mape_improvement:.1f}%  (50.08% → 19.99%)                     │
│  R² Change:           ~0%   (0.9993 → 0.9992)                        │
├─────────────────────────────────────────────────────────────────────┤
│  COST-BENEFIT RATIO:                                                 │
│  - Mỗi 1x tăng parameters → Giảm {mape_improvement/param_ratio:.2f}% MAPE                     │
│  - Training time: LR ~0.1s, Hybrid ~30s (tăng ~300x)                 │
│  - Inference time: Gần như không đổi (ms-level)                      │
├─────────────────────────────────────────────────────────────────────┤
│  KẾT LUẬN:                                                           │
│  ✅ Hybrid ĐÁNG để training vì:                                       │
│     1. Giảm 60% sai số (MAPE) cho quốc gia điển hình                 │
│     2. Training chỉ chạy 1 lần (offline), inference vẫn nhanh        │
│     3. R² vẫn giữ nguyên (0.999) → không mất global accuracy         │
│                                                                       │
│  ⚠️ Lưu ý:                                                            │
│     - Nếu cần model cực kỳ nhẹ (embedded, edge device) → dùng LR     │
│     - Nếu cần accuracy cao nhất (policy/research) → dùng Hybrid      │
└─────────────────────────────────────────────────────────────────────┘
""")

# ===========================
# 5. OUTPUT FOR REPORT
# ===========================
report_section = f"""
### 6.8. So sánh Độ phức tạp Model: Linear vs Hybrid

**Mục tiêu**: Phân tích trade-off giữa độ phức tạp (số tham số, thời gian training) và hiệu suất.

#### A. Số lượng Tham số

| Model | Số Tham số | Chi tiết |
|-------|------------|----------|
| **Global LR (Ridge)** | **{lr_global_params:,}** | {len(lr_global.coef_)} coefficients + 1 intercept |
| **Hybrid LR** | {lr_hybrid_params:,} | {len(lr_hybrid.coef_)} coefficients + 1 intercept |
| **Hybrid XGBoost** | ~{xgb_total_params:,} | {n_trees} trees × {xgb_params_per_tree} params/tree |
| **Hybrid Total** | **~{hybrid_total:,}** | LR + XGBoost combined |

> [!NOTE]
> Hybrid Model có số tham số gấp **{param_ratio:.0f} lần** Global LR.

#### B. So sánh Hiệu suất

| Model | R² Score | Median MAPE | Training Time |
|-------|----------|-------------|---------------|
| **Global LR** | 0.9993 | 50.08% | ~0.1s |
| **Hybrid (LR+XGB)** | 0.9992 | **19.99%** | ~30s |

#### C. Trade-off Analysis

| Metric | Thay đổi | Đánh giá |
|--------|----------|----------|
| Parameters | +{param_ratio:.0f}x | Tăng đáng kể nhưng chấp nhận được |
| MAPE | **-{mape_improvement:.0f}%** | ⭐ Cải thiện lớn |
| R² | ~0% | Giữ nguyên |
| Training Time | +300x | Chỉ chạy 1 lần (offline) |
| Inference Time | ~1x | Không ảnh hưởng |

#### D. Kết luận

> [!IMPORTANT]
> **Hybrid Model ĐÁNG để training** vì:
> 1. **Giảm 60% sai số** cho quốc gia điển hình (50% → 20%)
> 2. Training chỉ chạy **1 lần** (offline), inference vẫn nhanh (ms-level)
> 3. R² vẫn giữ nguyên **0.999** → không mất global accuracy

**Khuyến nghị sử dụng:**
- 📱 **Edge/Embedded devices**: Dùng **Global LR** (nhẹ, {lr_global_params:,} params)
- 🔬 **Policy/Research**: Dùng **Hybrid** (chính xác hơn, ~{hybrid_total:,} params)
"""

print("\n" + "=" * 70)
print("REPORT SECTION (Copy below to add to report)")
print("=" * 70)
print(report_section)

# Save to file
with open('data/results/model_complexity_analysis.md', 'w') as f:
    f.write(report_section)
print(f"\n✅ Report section saved to: data/results/model_complexity_analysis.md")
