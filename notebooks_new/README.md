# 📚 Notebooks - Global Energy & CO2 Emissions Forecasting

## Mục lục (Table of Contents)

Các notebooks được tổ chức theo thứ tự logic của báo cáo `Final_Project_Report.md`.

| # | Notebook | Mô tả | Report Section |
|---|---|---|---|
| 01 | `01_Data_Exploration.ipynb` | Khám phá dữ liệu, EDA, Visualizations | Section 2 |
| 02 | `02_Preprocessing_Pipeline.ipynb` | Pipeline tiền xử lý cho 3 thuật toán | Section 3 |
| 03 | `03_Phase0_Random_vs_TimeSeries.ipynb` | So sánh Random Split vs Time-Series Split | Section 4.1 |
| 04 | `04_Phase1_Global_LR_Baseline.ipynb` | Global Linear Regression baseline | Section 4.2-4.3 |
| 05 | `05_Phase2_Hyperparameter_Tuning.ipynb` | GridSearchCV với TimeSeriesSplit | Section 4.7 |
| 06 | `06_Phase3_KMeans_Clustering.ipynb` | K-Means clustering + cluster-based models | Section 4.4, 4.9 |
| 07 | `07_Phase4_Recursive_Forecasting.ipynb` | Multi-step ahead forecasting | Section 4.11 |
| 08 | `08_Phase5_RealWorld_Validation.ipynb` | Validation với World Bank API (2020-2023) | Section 4.10 |
| 09 | `09_Fairness_Robustness.ipynb` | Rolling CV, Feature Importance, Fairness | Section 4.8, 4.12-4.14 |
| 10 | `10_Hybrid_Model.ipynb` | Hybrid Model (LR + XGBoost) | Section 6 |

## Cách sử dụng

1. Mở từng notebook theo thứ tự từ 01 đến 10
2. Chạy tất cả cells từ đầu đến cuối
3. Kết quả sẽ tương ứng với các sections trong báo cáo

## Yêu cầu

- Python 3.12+
- Các packages: pandas, numpy, sklearn, xgboost, matplotlib, seaborn
- Dữ liệu đã được tiền xử lý trong `data/processed/`

## Tác giả

- Sinh viên: [Tên sinh viên]
- MSSV: [MSSV]
- Môn: DS102 - [Tên môn học]
- Trường: UIT - Đại học Công nghệ Thông tin
