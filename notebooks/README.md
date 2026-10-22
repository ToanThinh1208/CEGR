# Hướng dẫn chạy Thực nghiệm CEGR theo Môi trường (Environment Benchmarks)

Thư mục này chứa các Jupyter Notebooks thực nghiệm được tổ chức và phân chia theo **từng môi trường cụ thể** thay vì tách theo từng baseline riêng lẻ. Tất cả các notebook đã được tối ưu hóa kiến trúc theo dạng mô-đun: **Cell 1 giữ nguyên toàn bộ cài đặt & cấu hình toàn cục**, trong khi **các cell còn lại sử dụng trực tiếp các lớp và hàm từ gói thư viện [cegr](file:///d:/CEGR/Cegr-official/cegr)** (`cegr.envs`, `cegr.core`, `cegr.solvers`, `cegr.synthesis`, `cegr.baselines`, `cegr.engine`, `cegr.eval`) thay vì định nghĩa lại mã nguồn thủ công.

---

## 1. Danh sách Notebooks theo Môi trường

| Tên Notebook | Môi trường (`ENV_NAME`) | Kích thước lưới | Horizon DP Kép (`HORIZONS`) | Mô tả & Mục đích kiểm thử |
| :--- | :--- | :--- | :--- | :--- |
| **`doorkey5_benchmark.ipynb`** | `doorkey5` | 5x5 | `(30, 60)` | **Mức 1 - Môi trường chuẩn**: Kiểm chứng toàn diện khả năng sửa chữa và đối sánh với CARD & Eureka. |
| **`doorkey6_benchmark.ipynb`** | `doorkey6` | 6x6 | `(40, 80)` | **Mức 2 - Mở rộng không gian**: Kiểm thử khả năng tổng quát hóa khi tăng kích thước phòng và vị trí cửa/chìa khóa. |
| **`doorkey8_benchmark.ipynb`** | `doorkey8` | 8x8 | `(60, 120)` | **Mức 3/4 - Không gian lớn**: Kiểm thử khả năng mở rộng (scalability) của bộ giải DP và độ bền bỉ của bộ sinh CE. |
| **`metaworld_benchmark.ipynb`** | `metaworld` | Liên tục | `150` | **Điều khiển cánh tay robot liên tục**: Kiểm thử mở rộng CEGR trên các tác vụ Meta-World (Reach, Door Open, Pick-Place). |

---

## 2. Chuẩn hóa Cấu hình Toàn cục (Cell 1)

Mỗi notebook môi trường DoorKey đều đồng bộ chính xác 100% cấu hình thực nghiệm chuẩn hóa của nghiên cứu:
- **Mô hình LLM**: `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` qua batch vLLM engine.
- **Phương pháp đối sánh (`METHODS`)**: `["EUREKA", "CARD", "CARD_REFL", "CEGIS"]` (đồng nhất ngân sách 4 vòng x 2 ứng viên / round).
- **Ablation Studies (`ABLATIONS`)**: `["CEGIS_noPool", "CEGIS_noAnchors", "CEGIS_noClass", "CEGIS_noG2", "CEGIS_noG4", "CEGIS_noG5"]`.
- **Hồ sơ chấm điểm (`SCORING_PROFILE`)**: `relaxed` (Probe episodes: 2500, Tail threshold: 0.50, Gamma: 1.0, Optimistic init: 1.0, Alpha decay: True, Require dense: True).
- **Cơ chế biểu diễn vị trí chìa khóa (`KEYPOS_MODE`)**: `agent_pos` (bảo toàn không gian trạng thái Markov 270 states trên DoorKey-5x5).
- **Ngưỡng biên an toàn (`EPS_MARGIN_REL`)**: `0.01` ($M(R') > 1\% |V_{succ}|$).
- **Ngưỡng Gate 4 (`RHO_THR`)**: `0.75` (Hệ số tương quan thứ hạng Spearman giữa reward và tiến độ thực).
- **Chế độ kiểm tra tín hiệu Gate 2 (`G2_MODE`)**: `merge`.

---

## 3. Hướng dẫn chạy trên Kaggle GPU (T4 / P100 / A100)

1. Tải notebook môi trường cần chạy (ví dụ `doorkey5_benchmark.ipynb`) lên Kaggle Notebook.
2. Bật **Accelerator: GPU T4 x2** hoặc **P100 / A100**.
3. Thêm Kaggle Secret `HF_TOKEN` (hoặc để trống nếu dùng token mặc định trong code).
4. Nhấn **Run All**:
   - **Cell 1**: Tự động gỡ các gói xung đột và cài đặt `torch==2.6.0+cu124` cùng `vllm`.
   - **Cell 2 & 3**: Khởi tạo môi trường, DP Solver chính xác và sandbox an toàn.
   - **Cell 4 & 5**: Khởi tạo Oracle, bộ chấm điểm 2 tầng (`tier1_safe_aligned` & `tier2_dense_vf`) và Q-learning probe v2.
   - **Cell 6 & 7**: Khởi tạo Quỹ đạo tham chiếu, Counterexample Pool và bộ kiểm tra tín hiệu.
   - **Cell 8 & 9**: Thiết lập GateKeeper 5 cổng, Attacker, Invariant Assimilator và Architect prompt builder.
   - **Cell 10 & 11**: Nạp model AWQ vào vLLM batch engine.
   - **Cell 12 & 13 (Pha 1)**: Sinh 12 hệ mutant + sinh reward natural hacked, xuất ra file manifest JSON tương ứng môi trường.
   - **Cell 14 & 15 (Pha 2)**: Chạy vòng lặp đánh giá song song 4 phương pháp chính và 6 biến thể ablation.
   - **Cell 16 & 17 (Pha 3)**: Thống kê Bootstrap cấp độ reward, vẽ 3 hình phân tích và xuất 4 bảng LaTeX.
   - **Cell 18 & 19**: Chạy bộ kiểm thử nghiệm thu T2-T12 để đảm bảo độ tin cậy tuyệt đối của kết quả.
