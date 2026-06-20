# CCR.GB: Compositional Causal Reasoning Evaluation in Language Models (Đánh giá Lý luận Nhân quả Thành phần trong các Mô hình Ngôn ngữ)

Repository này chứa mã nguồn cho bài báo *Compositional Causal Reasoning Evaluation in Language Models* ([ICML 2025](https://arxiv.org/abs/2503.04556)). Bộ kiểm thử CCR.GB được thiết kế để đánh giá LLM về khả năng lý luận nhân quả thành phần (compositional causal reasoning) trên cả ba cấp độ trong Hệ thống phân cấp nhân quả của Pearl: (1) liên tưởng (association), (2) can thiệp (intervention), và (3) phản thực tế (counterfactuals).

Để có cái nhìn toàn diện về nghiên cứu này, vui lòng truy cập [trang dự án chính](https://jmaasch.github.io/ccr/).

Thành viên thực hiện:

* 25C11039 - Tô Tấn Hiệp

* 25C11028 - Nguyễn Hoàng Việt

---

## Tổng quan mã nguồn & Các Notebook

Chúng tôi đã triển khai thêm hai notebook chưa tồn tại trong repository gốc của dự án để có thể tập trung trình bày, xác minh và tái hiện các phát hiện của bài báo:

1.  **[experimental_results.ipynb]:** 
    *   **Mục tiêu:** Tái hiện và trực quan hóa các kết quả thực nghiệm chính từ **Mục 6** của bài báo (trên ví dụ `CandyParty`).
    *   **Tính năng:** Tính toán động các giá trị ground truth và mô phỏng hồ sơ lý luận của các LLM (đối với các mô hình như `o1`, `GPT-4o + CoT`, và `Llama 3`) dựa trên các số liệu thống kê thực nghiệm được báo cáo trong nghiên cứu.
    *   **Kết quả trực quan hóa:** Tái tạo biểu đồ phân tán giữa Tính kiểm chứng ngoài và Tính nhất quán trong (**Figure 10**), hồ sơ lý luận trên Cây Lát Cắt Giao Hoán (CCT) (**Figure 11**), và mức độ tăng trưởng sai số theo độ dài đường đi nhân quả (**Figure 12**).
2.  **[verification.ipynb]:**
    *   **Mục tiêu:** Thực hiện xác minh cấu trúc từng bước của quy trình đánh giá CCR.
    *   **Tính năng:** Xây dựng DAG nhân quả, tạo ngữ cảnh truy vấn (thực tế và phản thực tế), tính toán giá trị PNS ground truth chính xác thông qua mô phỏng Mô hình nhân quả cấu trúc (SCM) ($n = 100,000$), và thực hiện xác minh Định lý 5.1.

---

## Phân tích chi tiết kết quả của `verification.ipynb`

Dưới đây là phân tích chặt chẽ các kết quả thực thi từ [verification.ipynb] dưới cấu hình seed cố định (`np.random.seed(0)` và `random.seed(0)`).

### 1. Đồ thị nhân quả & Xác định nút
Đối với đồ thị gồm 3 thành phần song liên thông (BCC) được cấu hình với `n_per_bcc = [4, 3, 3]` và thuộc kiểu vòng (cycle), bộ sinh nhiệm vụ xây dựng một Đồ thị có hướng không chu trình (DAG) gồm **8 nút** và **2 điểm cắt** (cutpoints / khớp). Các nhãn nút được gán động (lấy ngẫu nhiên từ danh sách tên nữ qua thư viện `Faker`) là:
*   **Gốc ($X$):** `Nereida`
*   **Điểm cắt 1 ($C$):** `Julia`
*   **Điểm cắt 2 ($D$):** `Celie`
*   **Lá ($Y$):** `Tristan`

Chuỗi đường đi trên Cây Lát Cắt Giao Hoán (CCT) kết quả là:
$$X (\text{Nereida}) \to C (\text{Julia}) \to D (\text{Celie}) \to Y (\text{Tristan})$$

### 2. Các giá trị PNS Ground Truth
Sử dụng $100,000$ mẫu mô phỏng SCM, chúng tôi đã ước lượng Xác suất của Tính cần thiết và Tính đủ (PNS) cho tất cả các cặp nguyên nhân - kết quả trong CCT:
*   **PNS toàn cục ($PNS_{XY}$):** $0.000390$
*   **PNS cục bộ ($PNS_{XC}$):** $0.048060$
*   **PNS cục bộ ($PNS_{CY}$):** $0.009670$
*   **PNS cục bộ ($PNS_{XD}$):** $0.005730$
*   **PNS cục bộ ($PNS_{DY}$):** $0.081460$
*   **PNS cục bộ ($PNS_{CD}$):** $0.121130$

**Quan sát chính:** Giá trị PNS toàn cục ($0.000390$) cực kỳ nhỏ. Bởi vì tất cả các hàm nhân quả đều là toán tử logic `or` và các nút được kết nối nối tiếp, xác suất phản thực tế của nút lá thay đổi trạng thái theo nút gốc bị suy giảm nhanh chóng qua nhiều lớp biến trung gian. Các tác động cục bộ lớn hơn đáng kể (ví dụ: $PNS_{CD} \approx 12\%$), chứng minh rằng ảnh hưởng nhân quả có tính chất khu vực hóa mạnh mẽ.

### 3. Xác minh Định lý 5.1 & Sai số Monte Carlo
Định lý 5.1 phát biểu rằng đối với các cấu trúc điểm cắt nối tiếp, PNS toàn cục bằng tích của các giá trị PNS cục bộ dọc theo bất kỳ đường đi nào trong CCT. Notebook đã xác minh định lý này trên ba phép tích thành phần:
*   **Tích thành phần 1 ($PNS_{XC} \times PNS_{CY}$):** $0.000465$ (Sai số tuyệt đối tương đối, **RAE: 19.16%**)
*   **Tích thành phần 2 ($PNS_{XD} \times PNS_{DY}$):** $0.000467$ (Sai số tuyệt đối tương đối, **RAE: 19.68%**)
*   **Tích thành phần 3 ($PNS_{XC} \times PNS_{CD} \times PNS_{DY}$):** $0.000474$ (Sai số tuyệt đối tương đối, **RAE: 21.59%**)

#### **Tại sao cả ba tích thành phần đều lệch so với PNS toàn cục khoảng 19%–22%?**
1.  **Biến động lấy mẫu cỡ mẫu hữu hạn:** Độ lệch này là một sai số kỹ thuật của việc lấy mẫu Monte Carlo với cỡ mẫu hữu hạn, không phải là sự vi phạm Định lý 5.1.
2.  **Độ nhạy của xác suất toàn cục nhỏ:** Xác suất toàn cục thực tế $PNS_{XY} = 0.000390$ tương đương với việc chỉ quan sát thấy $\approx 39$ sự kiện kết hợp trên tổng số $100,000$ mẫu. Sai số chuẩn kỳ vọng của ước lượng này là $\sqrt{39}/100,000 \approx 0.000062$, dẫn đến hệ số biến thiên (sai số chuẩn tương đối) rất cao, khoảng $\approx 16\%$.
3.  **Tích lũy sai số của phép nhân:** Các tích thành phần nhân hai hoặc ba ước lượng độc lập với nhau. Vì mỗi PNS cục bộ đều có phương sai lấy mẫu riêng, việc nhân chúng lại sẽ làm trầm trọng thêm sai số tương đối.
4.  **Xu hướng lệch hướng:** Ở cỡ mẫu hữu hạn $n$, việc nhân các xác suất nhỏ dương bị nhiễu độc lập dẫn đến một xu hướng lệch dương hệ thống (upward bias), giải thích tại sao cả ba phép tích thành phần đều ước lượng cao hơn giá trị PNS toàn cục thực tế.

### 4. Tính nhất quán cấu trúc đa chủ đề
*   **Tính tương đương chủ đề:** Notebook đã xác minh rằng các đồ thị DAG của `FluVaccine` và `FlowerGarden` trùng khớp hoàn toàn về mặt cấu trúc với `CandyParty` (`True`). Điều này xác nhận rằng bộ kiểm thử cô lập được khả năng lý luận khỏi ngữ cảnh ngữ nghĩa: đồ thị nhân quả, SCM, và quan hệ cha-con logic-toán học vẫn giữ nguyên; chỉ có biểu hiện ngôn ngữ tự nhiên bên ngoài thay đổi.
*   **Biến thể ClinicalNotes:** `ClinicalNotes` áp dụng toán tử logic `or` cho tất cả các mối quan hệ trung gian nhưng cố định hàm của nút lá cuối cùng là `and` (thể hiện rằng một khuyến nghị phẫu thuật yêu cầu tất cả các điều kiện chẩn đoán trung gian phải được đáp ứng đồng thời).
*   **Biến thể CellBio:** `CellBio` nhắm mục tiêu vào Tác động can thiệp trung bình (ATE) thay vì PNS. Nó sử dụng một quy trình SCM tuyến tính hoàn toàn khác biệt với các biến liên tục và nhiễu Gaussian, đại diện cho thể tích phiên mã của gen.

---

## Cấu hình & Cách chạy các Notebook

Đảm bảo rằng bạn đã cài đặt các thư viện phụ thuộc trong môi trường ảo.

### 1. Kích hoạt môi trường ảo
```powershell
# Trên Windows PowerShell:
.venv\Scripts\activate
```
hoặc kích hoạt các môi trường khác tương ứng.

### 2. Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```
*(Hãy đảm bảo `pandas`, `numpy`, `networkx`, và `matplotlib` đã được cài đặt.)*

### 3. Khởi chạy Jupyter
```bash
jupyter notebook
```
Mở **[experimental_results.ipynb]** hoặc **[verification.ipynb]** để kiểm tra kết quả. Vì các notebook đã được thực thi sẵn, bạn có thể xem ngay các biểu đồ và bảng biểu kết quả ngay khi mở file.

