# HƯỚNG DẪN CHI TIẾT CÁC NOTEBOOK DỰ ÁN DỰ BÁO CO2

Tài liệu này mô tả chi tiết chức năng, input, output và các lưu ý kỹ thuật cho từng notebook trong thư mục `notebooks_new/`. Cấu trúc dự án được thiết kế theo luồng xử lý từ dữ liệu thô đến mô hình Hybrid cuối cùng.

---

## 🏗️ Giai đoạn 1: Chuẩn bị Dữ liệu

### 1. `01_Data_Exploration.ipynb`
*   **Chức năng**: Khám phá dữ liệu thô (EDA), phân tích phân phối, missing values và tương quan giữa các biến.
*   **Input**: 
    *   File raw data gốc (thường là `data/raw/Global_Energy_Consumption.csv` hoặc tương tự từ Kaggle).
*   **Output**: 
    *   Các biểu đồ phân tích (Histogram, Heatmap).
    *   Không save file dữ liệu mới.
*   **Lưu ý**:
    *   Notebook này giúp xác nhận các giả định (ví dụ: missing value ở các nước nghèo, phân phối lệch của tài chính/CO2).

### 2. `02_Data_Preprocessing.ipynb`
*   **Chức năng**: Pipeline tiền xử lý dữ liệu chuẩn hóa. Thực hiện:
    *   Xử lý missing values (Nội suy tuyến tính cho dữ liệu giữa, Median cho phần còn lại).
    *   Tạo Lag features (`CO2_lag1`, `GDP_lag1`...).
    *   Log transform cho các biến bị lệch (skewed) như `Financial flows`.
    *   Loại bỏ năm 2020 (do COVID gây nhiễu).
    *   Lọc bỏ các quốc gia có dưới 15 năm dữ liệu.
*   **Input**: 
    *   Raw data.
*   **Output**:
    *   `data/processed/common_preprocessed.csv`: File dữ liệu sạch dùng chung cho tất cả các model sau này.
*   **Lưu ý**:
    *   Đây là notebook quan trọng nhất để đảm bảo tính nhất quán.
    *   Biến `Year` được giữ lại để split data.
    *   **Logic Data Quality**: Chỉ giữ quốc gia có >= 15 point dữ liệu.

---

## 🧪 Giai đoạn 2: Thử nghiệm Baseline & Phương pháp

### 3. `03_Phase0_Random_vs_TimeSeries.ipynb`
*   **Chức năng**: Chứng minh "Bẫy nội suy" (Interpolation Trap). So sánh kết quả khi chia dữ liệu kiểu Random vs Time-Series.
*   **Input**: 
    *   `data/processed/common_preprocessed.csv`
*   **Output**:
    *   Bảng so sánh R² của LR, SVR, XGBoost trên 2 phương pháp split.
*   **Lưu ý**:
    *   **Kết luận quan trọng**: Random Split gây Data Leakage (R² ~0.99 giả tạo). Bắt buộc phải dùng Time-Series Split cho dự báo.

### 4. `04_Phase1_Global_LR_Baseline.ipynb`
*   **Chức năng**: Xây dựng model Baseline bằng Ridge Regression.
*   **Input**: 
    *   `data/processed/common_preprocessed.csv`
*   **Output**:
    *   Model Linear Regression baseline.
    *   Phân tích top feature importance (CO2_lag1 là quan trọng nhất).
    *   Đánh giá MAPE theo median.
*   **Lưu ý**:
    *   Sử dụng One-Hot Encoding cho cột `Entity` để bắt đặc trưng từng quốc gia.
    *   Baseline đạt R² cao (~0.999) nhưng Median MAPE còn lớn (~50%).

### 5. `05_Phase2_Hyperparameter_Tuning.ipynb`
*   **Chức năng**: Tối ưu hóa tham số cho các model (Ridge Alpha, XGBoost params) dùng TimeSeriesSplit.
*   **Input**: 
    *   `data/processed/common_preprocessed.csv`
*   **Output**:
    *   Best params (Ví dụ: Ridge `alpha=10.0`).
*   **Lưu ý**:
    *   Sử dụng `TimeSeriesSplit` trong GridSearchCV để tránh leakage khi tuning.

---

## 🔍 Giai đoạn 3: Phân tích Nâng cao & Thất bại

### 6. `06_Phase3_KMeans_Clustering.ipynb`
*   **Chức năng**: Thử nghiệm phân cụm quốc gia (Developed, Developing...) và train model riêng cho từng cụm.
*   **Input**: 
    *   `data/processed/common_preprocessed.csv`
*   **Output**:
    *   So sánh MAPE của từng cụm.
*   **Lưu ý**:
    *   **Kết quả**: Thất bại. Clustering làm tăng bất công (MAPE nước nghèo tăng vọt).
    *   Hiện tượng "Small Pond, Big Fish": Model trong cụm nhỏ bị overfit vào các nước lớn trong cụm đó.

### 7. `07_Phase4_Recursive_Forecasting.ipynb`
*   **Chức năng**: Đánh giá khả năng dự báo dài hạn (multi-step forecasting).
*   **Logic**: Dùng output dự đoán năm t làm input cho năm t+1.
*   **Input**: 
    *   `data/processed/common_preprocessed.csv`
    *   Model baseline.
*   **Output**:
    *   Biểu đồ R² giảm dần theo thời gian (2015-2019).
*   **Lưu ý**:
    *   Linear Regression bị sụt giảm nghiệm trọng (R² từ 0.99 xuống 0.44 sau 5 năm) do tích lũy sai số.

---

## 🚀 Giai đoạn 4: Giải pháp Cuối cùng (Solution)

### 8. `08_Hybrid_Model.ipynb` ⭐
*   **Chức năng**: Cài đặt giải pháp Hybrid Model (Ridge + XGBoost Residuals).
*   **Logic**:
    1.  Dùng Ridge Regression dự đoán xu hướng chính (Trend).
    2.  Tính phần dư (Residuals = Thực tế - Dự báo Ridge).
    3.  Dùng XGBoost để học và dự đoán phần dư này (phi tuyến tính).
    4.  Kết quả = Ridge + XGBoost.
*   **Input**: 
    *   `data/processed/common_preprocessed.csv`
*   **Output**:
    *   Model Hybrid đã train.
    *   File `models/hybrid_model.pkl` (nếu có save).
    *   Kết quả đánh giá: **Median MAPE giảm ~60%** (từ 50% xuống 20%).
*   **Lưu ý**:
    *   Đây là notebook chứa giải pháp tối ưu nhất của dự án.
    *   XGBoost ở đây dùng cây nông (`max_depth=3`) để tránh overfit vào noise của residuals.

### 9. `09_Phase5_RealWorld_Validation.ipynb`
*   **Chức năng**: Kiểm thử model với dữ liệu thực tế bên ngoài (Out-of-sample validity) giai đoạn 2020-2023.
*   **Input**: 
    *   Dữ liệu fetch từ World Bank API hoặc OWID (Our World in Data).
    *   Hybrid Model đã train.
*   **Output**:
    *   Đánh giá model trước cú sốc COVID-19 và phục hồi.
*   **Lưu ý**:
    *   Dùng để chứng minh tính Robust của hệ thống trong thực tế.

### 10. `10_Fairness_Robustness.ipynb`
*   **Chức năng**: Đánh giá tính công bằng (Fairness) của model trên các nhóm quốc gia khác nhau.
*   **Input**: 
    *   Kết quả dự báo từ Hybrid Model.
*   **Output**:
    *   Phân tích MAPE theo từng nhóm (Châu lục, Thu nhập, GDP).
*   **Lưu ý**:
    *   Chỉ ra các hạn chế còn tồn tại (ví dụ: các đảo quốc nhỏ "micro-states" vẫn có sai số cao).

---

## 📝 Quy trình chạy Code khuyến nghị

1.  **Chạy lần đầu/Clean run**:
    *   Chạy `02` (tạo data) -> `04` (baseline) -> `08` (hybrid).
2.  **Để hiểu vấn đề/nghiên cứu**:
    *   Chạy `03` (hiểu tại sao không dùng Random Split).
    *   Chạy `06`, `07` (thấy các phương pháp khác thất bại thế nào).
3.  **Validation cuối cùng**:
    *   Chạy `09`, `10`.
