# Dự báo Phát thải CO2 Toàn cầu (2000 - 2020)

**Đồ án môn học: DS102 - Học máy thống kê**
Dự án tập trung xây dựng và tối ưu hóa các mô hình Học máy để dự báo lượng phát thải Carbon Dioxide ($CO_2$) dựa trên các chỉ số kinh tế và năng lượng bền vững của 176 quốc gia.

---

## 🚀 1. Tóm tắt Kết quả (Executive Summary)

Sau 5 giai đoạn thực nghiệm nghiêm ngặt, chúng tôi đã xác định được chiến lược tối ưu nhất:

| Hạng | Mô hình / Chiến lược | $R^2$ Score | Đánh giá |
| :--- | :--- | :--- | :--- |
| **1 🏆** | **Clustered XGBoost (K-Means)** | **0.7740** | **Mô hình tốt nhất**. Cân bằng giữa độ phức tạp và hiệu quả. |
| 2 | Tuned Global XGBoost | 0.7558 | Hiệu quả cao, dễ triển khai hơn K-Means. |
| 3 | Default Global XGBoost | 0.7528 | Baseline mạnh mẽ. |
| 4 | Panel Linear Regression | 0.6925 | Tốt, nhưng hạn chế trong việc bắt các mối quan hệ phi tuyến. |
| 5 | Ultimate (K-Means + Tuning) | 0.6795 | **Overfitting**. Phức tạp hóa vấn đề không phải lúc nào cũng tốt. |
| 6 | Standard Models (No Panel) | < 0.0 | Thất bại do thiếu thông tin chuỗi thời gian (Panel Info). |

---

## 🔬 2. Quy trình Thực nghiệm (Experimental Pipeline)

Dự án được cấu trúc thành 5 notebook chính, tương ứng với quá trình tư duy khoa học:

### Giai đoạn 1: Thiết lập & Cảnh báo
*   **[Notebook 1: The Illusion of Accuracy](notebooks/final_1_Random_Split.ipynb)**
    *   **Mục tiêu**: Chứng minh sự nguy hiểm của việc chia dữ liệu ngẫu nhiên (Random Split) với dữ liệu chuỗi thời gian.
    *   **Kết quả**: $R^2 \approx 0.99$ (Ảo tưởng do Data Leakage).

*   **[Notebook 2: The Real Benchmark](notebooks/final_2_Forecasting_Time_Split.ipynb)**
    *   **Mục tiêu**: Thiết lập thước đo chuẩn xác với việc chia dữ liệu theo thời gian (Time-Based Split: Train < 2015, Test >= 2015).
    *   **Baseline**: Panel XGBoost đạt $R^2 \approx 0.753$.

### Giai đoạn 2: Tối ưu hóa (Optimization)
*   **[Notebook 3: Divide and Conquer (K-Means)](notebooks/final_3_KMeans_Optimization.ipynb)**
    *   **Chiến lược**: Phân cụm 176 quốc gia thành 3 nhóm (Low/Mid/High Income) dựa trên dữ liệu năm 2014, sau đó train model riêng cho từng nhóm.
    *   **Kết quả**: $R^2$ tăng lên **0.7740** (+2.1%). Đây là chiến lược thành công nhất.

*   **[Notebook 4: Hyperparameter Tuning](notebooks/final_4_Hyperparameter_Tuning.ipynb)**
    *   **Chiến lược**: Tinh chỉnh tham số cho Global Model bằng `RandomizedSearchCV`.
    *   **Kết quả**: $R^2 = 0.7558$. Cải thiện nhẹ nhưng không bằng K-Means.

### Giai đoạn 3: Giới hạn của sự phức tạp
*   **[Notebook 5: The Ultimate Optimization](notebooks/final_5_Ultimate_Optimization.ipynb)**
    *   **Chiến lược**: Kết hợp cả K-Means VÀ Hyperparameter Tuning cho từng cụm.
    *   **Kết quả**: $R^2$ tụt xuống **0.6795**.
    *   **Bài học**: Việc tinh chỉnh quá mức trên tập dữ liệu nhỏ (từng cụm) dẫn đến Overfitting. **"Simple is Better"**.

---

## 📂 3. Cấu trúc Thư mục

```text
Global-Energy/
├── data/
│   ├── raw/                  # Dữ liệu gốc
│   └── processed/            # Dữ liệu đã xử lý (Imputed)
├── notebooks/                # 5 Notebook báo cáo chính
│   ├── final_1_Random_Split.ipynb
│   ├── final_2_Forecasting_Time_Split.ipynb
│   ├── final_3_KMeans_Optimization.ipynb
│   ├── final_4_Hyperparameter_Tuning.ipynb
│   └── final_5_Ultimate_Optimization.ipynb
└── README.md                 # Tài liệu này
```

---

## 🛠 4. Công nghệ sử dụng
*   **Ngôn ngữ**: Python 3.12
*   **Thư viện**: Pandas, NumPy, Scikit-learn (Pipeline, GridSearchCV), XGBoost, Matplotlib/Seaborn.

---

## 👥 5. Đội ngũ thực hiện
**Nhóm 4 - Lớp DS102.Q12.CNVN**
*   **Sinh viên**:
    *   Lê Thị Thanh Trúc (23521667)
    *   Vũ Thị Ngọc Mai (23520913)
*   **Giảng viên hướng dẫn**:
    *   PGS. TS. Nguyễn Lưu Thùy Ngân
    *   TS. Dương Ngọc Hảo
