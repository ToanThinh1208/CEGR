# Tài liệu Kỹ thuật: Kiến trúc Hệ thống CEGR & Đặc tả Logic Đối sánh (Eureka, CARD)

**Dự án:** CounterExample-Guided Reward Design with Multi-Tier GateKeeper Verification (CEGR)

**Mục tiêu tài liệu:** Chuẩn hóa cấu trúc mã nguồn tích hợp 3 phương pháp (`Eureka`, `CARD`, `CEGR`) trong cùng một repository trên cả 2 miền môi trường: **Rời rạc dạng bảng (`Minigrid DoorKey`)** và **Điều khiển robot liên tục (`Meta-World`)**; định nghĩa chính xác phần lõi thuật toán cần kế thừa từ hai bài báo gốc và cung cấp danh sách kiểm tra (checklist) nghiệm thu logic.

---

## 1. Kiến trúc Dự án Thống nhất (Project Architecture)

Để đảm bảo nguyên tắc **So sánh công bằng tuyệt đối (Apples-to-Apples Comparison)** trên cả hai hệ môi trường (`Minigrid DoorKey` và `Meta-World`), cùng mô hình nền tảng `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` và cùng ngân sách gọi LLM ($N = 4$), dự án được chia thành 3 tầng rõ rệt:

1. **Tầng Hạ tầng & Dữ liệu Dùng chung (`data/`, `configs/`, `cegr/envs/`, `cegr/core/`, `cegr/solvers/`, `cegr/engine/`):** Cung cấp giao diện môi trường đa miền (Tabular & Continuous), tập dữ liệu đóng băng, sandbox biên dịch code, bộ giải tìm lỗi/đánh giá (DP Solver cho miền rời rạc, Rollout/SAC/PPO Probe cho miền liên tục) và động cơ suy luận vLLM thống nhất.
2. **Tầng Thuật toán (`cegr/baselines/` và `cegr/synthesis/`):** Chứa logic tìm kiếm và tinh chỉnh hàm phần thưởng riêng biệt của `ORIG`, `EUREKA`, `CARD`, `CARD_REFL` và `CEGR` (cùng 6 biến thể Ablation).
3. **Tầng Đánh giá & Thực thi (`cegr/eval/`, `notebooks/`):** Chấm điểm độc lập qua Oracle Verified Fix, chạy kiểm định thống kê Cluster Bootstrap và điều phối thực thi trên Kaggle GPU.

### 1.1. Cây thư mục chi tiết (Đã tích hợp Meta-World)

```text
CEGR/
├── data/                                   # DỮ LIỆU ĐÓNG BĂNG (Đảm bảo tính tái lập 100%)
│   ├── doorkey/                            # Dữ liệu cho miền Rời rạc (Minigrid DoorKey)
│   │   ├── manifest_planted_115.json       # 115 hàm reward gốc bị hack (10 mutants + 105 natural)
│   │   ├── scratch_prompts_6.json          # 6 cấu hình sinh mới từ con số 0 (Scratch)
│   │   └── reference_trajectories/         # Tập cặp quỹ đạo tham chiếu P_0 (T_0 - T_3)
│   │       ├── doorkey_5x5_p0.pkl
│   │       ├── doorkey_6x6_p0.pkl
│   │       └── doorkey_8x8_p0.pkl
│   │
│   └── metaworld/                          # [MỚI] Dữ liệu cho miền Liên tục (Meta-World Manipulation)
│       ├── manifest_metaworld.json         # Tập hàm reward gốc bị lỗi (reach-v2, door-open-v2, pick-place-v2...)
│       ├── task_descriptions.json          # Đặc tả không gian quan sát (obs dict) & mốc tiến độ robot
│       └── reference_trajectories/         # Tập cặp quỹ đạo mẫu P_0 (Expert demo vs Sub-optimal/Hacked rollouts)
│           ├── reach_v2_p0.pkl
│           ├── door_open_v2_p0.pkl
│           └── pick_place_v2_p0.pkl
│
├── configs/                                # QUẢN LÝ SIÊU THAM SỐ & BIẾN THỂ THỰC NGHIỆM
│   ├── doorkey_config.yaml                 # Ngân sách N=4, T=30, 2T=60, DP Solver & Q-learning params
│   ├── metaworld_config.yaml               # [MỚI] Ngân sách N=4, T=500, SAC/PPO Probe & Rollout Attacker params
│   └── ablations.yaml                      # Cấu hình 6 biến thể: noPool, noAnchors, noClass, noG2, noG4, noG5
│
├── cegr/                                   # PACKAGE PYTHON CHÍNH
│   ├── __init__.py
│   ├── envs/                               # 1. Module Môi trường Đa miền (Tabular & Continuous)
│   │   ├── __init__.py
│   │   ├── base_env.py                     # Interface chuẩn hóa: trích xuất state dict `s`, action `a`, next_state `s2`
│   │   ├── doorkey.py                      # DoorKey đa kích thước: 5x5 (270 states), 6x6, 8x8
│   │   └── metaworld_env.py                # [MỚI] Wrapper Meta-World (chuyển vector obs/info thành state dict có ngữ nghĩa)
│   │
│   ├── core/                               # 2. Tầng Thực thi & Đánh giá cơ sở dùng chung
│   │   ├── __init__.py
│   │   ├── code_sandbox.py                 # Biên dịch chuỗi code LLM an toàn, hỗ trợ math/numpy cho Meta-World (G1)
│   │   ├── tpe_evaluator.py                # Thuật toán TPE dùng chung cho cả CARD, CARD_REFL và CEGR (G3)
│   │   └── base_runner.py                  # Class giao diện chuẩn hóa đầu vào/đầu ra cho mọi runner
│   │
│   ├── solvers/                            # 3. Bộ giải Toán học, RL Probe & Chấm điểm Chân lý (Oracle)
│   │   ├── __init__.py
│   │   ├── dp_solver.py                    # [DoorKey] Giải quy hoạch động chính xác (V_succ, V_fail) & DP Attacker
│   │   ├── trajectory_attacker.py          # [MỚI - Meta-World] Tìm quỹ đạo hack trên miền liên tục (CEM / Perturbation / Buffer Mining)
│   │   ├── rl_probe.py                     # [DoorKey] Q-learning probe ngắn (G5) & Full Tabular RL
│   │   ├── continuous_rl_probe.py          # [MỚI - Meta-World] SAC/PPO probe ngắn (G5) & Full Continuous RL
│   │   └── verified_fix.py                 # Bộ chấm điểm Oracle Verified Fix (Hỗ trợ cả chế độ Tabular DP & Continuous Rollout)
│   │
│   ├── engine/                             # 4. Động cơ LLM dùng chung
│   │   ├── __init__.py
│   │   └── llm_vllm.py                     # Wrapper vLLM batch inference (Qwen2.5-Coder-7B-AWQ, TP=2)
│   │
│   ├── baselines/                          # 5. MÃ NGUỒN CHUYỂN ĐỔI CHUẨN LOGIC TỪ 2 REPO ĐỐI SÁNH
│   │   ├── __init__.py
│   │   ├── orig_runner.py                  # Đánh giá trực tiếp hàm gốc R_0 không qua sửa chữa
│   │   │
│   │   ├── eureka/                         # [Kế thừa logic từ repo eureka-research/Eureka]
│   │   │   ├── __init__.py
│   │   │   ├── evolutionary_search.py      # Quản lý quần thể K=2 candidates x 2 thế hệ & chọn lọc Elitism
│   │   │   ├── reflection_prompt.py        # Trích xuất chuỗi thống kê tiến trình huấn luyện RL (Q-learning hoặc SAC/PPO)
│   │   │   ├── prompts.py                  # System prompt & Mutation prompt (Tự động nạp template DoorKey hoặc Meta-World)
│   │   │   └── runner.py                   # Bộ điều phối chạy giao thức Eureka
│   │   │
│   │   └── card/                           # [Kế thừa logic từ repo ShengjieSun419/CARD]
│   │       ├── __init__.py
│   │       ├── trajectory_buffer.py        # Bộ đệm lưu trữ quỹ đạo thành công/thất bại (tau+, tau-)
│   │       ├── feedback_generator.py       # Bộ sinh 3 khối phản hồi: Process, Trajectory, Preference
│   │       ├── prompts.py                  # Coder prompt & Introspection prompt của CARD
│   │       └── runner.py                   # Điều phối cả 2 chế độ: CARD chuẩn và CARD_REFL (có bộ nhớ chữ)
│   │
│   ├── synthesis/                          # 6. PHƯƠNG PHÁP ĐỀ XUẤT CEGR
│   │   ├── __init__.py
│   │   ├── gatekeeper.py                   # Điều phối kiểm định 5 tầng G1 -> G5 (Tự động chọn DP hoặc Continuous Attacker)
│   │   ├── static_antihack.py              # [Cổng G2] Quét AST phát hiện thiếu goal, hằng số dương, lỗi khoảng cách L2 dương
│   │   ├── signal_checker.py               # [Cổng G4] Kiểm tra margin mốc tiến độ, Spearman >= 0.75, phương sai
│   │   ├── ce_pool.py                      # Kho phản ví dụ tích lũy FIFO 40, băm khử trùng, phân loại CE
│   │   ├── assimilator.py                  # Phân tích chữ ký hành vi lỗi (Tabular loops & Continuous action saturation)
│   │   ├── dynamic_anchors.py              # Quản lý nâng cấp đặc tả bất biến Anchors (v0 -> v1)
│   │   └── runner.py                       # Bộ điều phối CEGR (chính thức & 6 biến thể Ablation)
│   │
│   └── eval/                               # 7. Thống kê & Xuất bản Báo cáo
│       ├── __init__.py
│       ├── bootstrap.py                    # Paired Cluster Bootstrap 5000 mẫu, tính p-value & CI 95%
│       └── report_generator.py             # Tự động xuất bảng LaTeX & biểu đồ PNG 300 DPI cho cả 2 môi trường
│
├── notebooks/                              # THỰC THI TRÊN KAGGLE (2x GPU T4)
│   ├── cegr_runner.ipynb                   # Notebook chính: Chạy End-to-End (DoorKey & Meta-World)
│   ├── metaworld_benchmark.ipynb           # [MỚI] Notebook chuyên biệt chạy thực nghiệm mở rộng trên Meta-World
│   ├── baseline_eureka.ipynb               # Notebook chạy kiểm thử độc lập Eureka
│   └── baseline_card.ipynb                 # Notebook chạy kiểm thử độc lập CARD
│
├── tests/                                  # KIỂM THỬ ĐƠN VỊ (Chạy nhanh trên CPU trước khi đẩy lên Kaggle)
│   ├── test_gatekeeper_on_mutants.py       # Kiểm tra G1-G5 trên 10 mutants DoorKey
│   ├── test_metaworld_adapter.py           # [MỚI] Kiểm tra wrapper trạng thái & TPE/Attacker trên Meta-World
│   └── test_baselines_logic.py             # Kiểm tra luồng chạy của Eureka và CARD với Mock LLM
│
├── requirements.txt
├── setup.py                                # Cài đặt gói ở chế độ editable: pip install -e .
└── README.md
```

---

### 1.2. Cơ chế Thích nghi Đa miền: Từ `Minigrid DoorKey` sang `Meta-World`

Việc bổ sung `Meta-World` giúp chứng minh CEGR không chỉ hoạt động trên không gian rời rạc giải được bằng Quy hoạch động (DP) mà còn tổng quát hóa sang **không gian trạng thái và hành động liên tục ($\mathcal{S} \subseteq \mathbb{R}^d, \mathcal{A} \subseteq [-1, 1]^4$)**. Để giữ nguyên chữ ký hàm `reward_components(s, a, s2)` xuyên suốt cả dự án, `metaworld_env.py` và `gatekeeper.py` hoạt động theo cơ chế đối xứng sau:

| Thành phần | Miền Rời rạc (`Minigrid DoorKey`) | Miền Liên tục (`Meta-World` - ví dụ: `door-open-v2`, `pick-place-v2`) |
| :--- | :--- | :--- |
| **Biểu diễn trạng thái `s` truyền vào hàm Reward** | Dict tọa độ lưới & cờ logic: `agent_pos`, `agent_dir`, `has_key`, `door_open`, `goal_pos` | Dict tọa độ 3D vật lý trích xuất từ MuJoCo: `hand_pos`, `obj_pos`, `goal_pos`, `gripper_open_amount`, `obj_grasped`, `in_target` |
| **Cổng G2 (`static_antihack.py`)** | Chặn thưởng dương khi đứng yên (`no-op`), thiếu `goal_reached`, hằng số dương vô điều kiện | Chặn thêm lỗi **thưởng thuận theo khoảng cách Euclid** (`+np.linalg.norm(hand - goal)` khiến tay máy bay ra xa vô cực) hoặc thưởng lớn cho vận tốc tay máy gây rung lắc |
| **Cổng G3 (`gatekeeper.py`)** | TPE trên $P_0 \cup \text{CE Pool}$ + **DP Attacker** (giải chính xác trên 270–5000 states) | TPE trên $P_0 \cup \text{CE Pool}$ + **Continuous Trajectory Attacker (`trajectory_attacker.py`)** dùng tìm kiếm Cross-Entropy Method (CEM) / nhiễu hành động để tìm quỹ đạo cày điểm mà không chạm đích |
| **Cổng G5 (`rl_probe`)** | Tabular Q-learning ngắn (1000 episodes $\times$ 2 seeds) | Continuous RL Probe ngắn dùng SAC/PPO hoặc Policy Optimization trên tập buffer (20.000–50.000 environment steps) |

---

## 2. Đặc tả Logic Hệ thống của Eureka và CARD cần giữ nguyên

Khi chạy trên cả `Minigrid DoorKey` lẫn `Meta-World`, ta **thay thế tầng môi trường và LLM API bằng hạ tầng dùng chung**, nhưng **bắt buộc bảo toàn 100% logic thuật toán cốt lõi** dưới đây.

### 2.1. Logic cốt lõi của EUREKA (`cegr/baselines/eureka/`)

Bài báo gốc: *Eureka: Human-level reward design via coding large language models (Ma et al., ICLR 2024)*.

Bản chất của Eureka dựa trên 3 trụ cột: **Tìm kiếm tiến hóa quần thể (Evolutionary Search)**, **Đánh giá bằng huấn luyện RL thực tế cho mọi ứng viên (Full RL Evaluation)**, và **Phản hồi thống kê theo tiến trình huấn luyện (Reward Reflection)**.

#### A. Luồng thuật toán từng bước (Với ngân sách $N = 4$ lần gọi LLM)

1. **Đánh giá khởi tạo ($R_0$):**
   * Nhận hàm phần thưởng gốc bị lỗi $R_0$ (từ `manifest_planted_115.json` của DoorKey hoặc `manifest_metaworld.json` của Meta-World).
   * Nếu $R_0$ thực thi hợp lệ qua `code_sandbox.py`, chạy huấn luyện RL đầy đủ (`rl_probe.py` cho DoorKey hoặc `continuous_rl_probe.py` cho Meta-World) để thu thập lịch sử tiến trình học (learning curves) của từng thành phần phần thưởng và tỷ lệ thành công nhiệm vụ (`task_success_rate`).
2. **Thế hệ 1 (Generation 1 — $K = 2$ ứng viên):**
   * Từ lịch sử huấn luyện của $R_0$, `reflection_prompt.py` tạo bản tóm tắt thống kê (Reward Reflection).
   * Gọi Coder LLM (`temperature = 0.7`) để sinh song song $K = 2$ hàm phần thưởng đột biến: $R_{1}^{(1)}$ và $R_{1}^{(2)}$ (tiêu tốn 2 lần gọi LLM).
   * **Bắt buộc huấn luyện RL:** Chạy huấn luyện RL cho cả $R_{1}^{(1)}$ và $R_{1}^{(2)}$ (nếu hàm không lỗi cú pháp).
   * **Chọn lọc tự nhiên (Elitism):** So sánh điểm fitness thực tế (`tail_success_rate` của RL policy) giữa các ứng viên trong thế hệ 1. Chọn ra ứng viên có fitness cao nhất làm cá thể cha/mẹ $R_{1}^*$ cho thế hệ tiếp theo.
3. **Thế hệ 2 (Generation 2 — $K = 2$ ứng viên):**
   * Trích xuất bản tóm tắt thống kê Reward Reflection từ quá trình huấn luyện RL của cá thể tốt nhất $R_{1}^*$.
   * Gọi Coder LLM sinh tiếp $K = 2$ ứng viên mới: $R_{2}^{(1)}$ và $R_{2}^{(2)}$ (tiêu tốn 2 lần gọi LLM còn lại, tổng cộng đúng $N = 4$).
   * Chạy huấn luyện RL cho cả $R_{2}^{(1)}$ và $R_{2}^{(2)}$.
4. **Xuất kết quả cuối cùng:**
   * Chọn hàm phần thưởng đạt `task_success_rate` cao nhất trong toàn bộ các ứng viên đã sinh ra làm kết quả cuối cùng $R_{\text{final}}$.

#### B. Cấu trúc "Reward Reflection" bắt buộc phải có (`reflection_prompt.py`)

Eureka theo dõi sự thay đổi của **từng thành phần trong dict trả về của `reward_components(s, a, s2)`** tại 5 mốc thời gian đều nhau trong quá trình huấn luyện RL (`20%, 40%, 60%, 80%, 100%` số episodes/steps):

* **Ví dụ trên Meta-World (`pick-place-v2`):**
  ```text
  We trained a continuous RL policy (SAC/PPO) using your provided reward_components function. 
  Below are the values of each reward component and task metrics tracked at every 20% of training steps:
  - task_success_rate: [0.00, 0.00, 0.05, 0.02, 0.00] (Max: 0.05, Final: 0.00)
  - final_target_distance: [0.42, 0.41, 0.39, 0.45, 0.48]
  - component 'reach_dist_reward': [2.10, 8.50, 19.40, 35.00, 42.10]
  - component 'grasp_bonus': [0.00, 12.00, 45.00, 80.00, 95.00]
  - component 'place_goal': [0.00, 0.00, 0.05, 0.00, 0.00]
  Please carefully analyze the scale and progression of each reward component above, identify why the policy failed to maximize task_success_rate, and provide a new improved reward_components(s, a, s2) function.
  ```

---

### 2.2. Logic cốt lõi của CARD & CARD_REFL (`cegr/baselines/card/`)

Bài báo gốc: *A Large Language Model-Driven Reward Design Framework via Dynamic Feedback for Reinforcement Learning (Sun et al., 2024)*.

Bản chất của CARD (vốn được tác giả gốc thiết kế cho các bài toán điều khiển liên tục như ManiSkill và Meta-World) dựa trên 3 trụ cột: **Cổng lọc sở thích quỹ đạo TPE (Trajectory Preference Evaluation)**, **Bộ đệm quỹ đạo** $(\tau^+, \tau^-)$, và **Cơ chế 3 loại phản hồi động (Process, Trajectory, Preference Feedback)**.

#### A. Luồng thuật toán từng bước (Với ngân sách $N = 4$ lần gọi LLM)

1. **Khởi tạo Bộ đệm Quỹ đạo (`trajectory_buffer.py`):**
   * Nạp tập các cặp quỹ đạo tham chiếu cố định $P_0 = \{(\tau_k^+, \tau_k^-)\}$:
     * **Trên DoorKey:** $T_3$ (mở cửa về đích) $\succ T_2$ (mở cửa nhưng đứng yên) $\succ T_1$ (nhặt chìa khóa rồi lặp hành động) $\succ T_0$ (xoay vòng tại chỗ).
     * **Trên Meta-World:** $T_3$ (tay máy gắp vật và đặt trúng đích) $\succ T_2$ (tay máy gắp được vật nhưng giữ nguyên trên không) $\succ T_1$ (tay máy chỉ chạm vào vật rồi đóng/mở kẹp liên tục) $\succ T_0$ (tay máy vung ra biên bàn làm việc).
2. **Đánh giá TPE trên hàm hiện tại (`tpe_evaluator.py`):**
   * Tại mỗi vòng lặp $i \in \{1, \dots, N\}$:
   * Bước 2a: Kiểm tra thực thi hàm phần thưởng trên các chuyển trạng thái $(s, a, s')$ trong $P_0$. Nếu lỗi biên dịch hoặc trả về `NaN/Inf`, sinh **Process Feedback** báo lỗi.
   * Bước 2b: Tính tổng phần thưởng chiết khấu $G(\tau) = \sum_{t=0}^{\vert\tau\vert-1} \gamma^t R(s_t, a_t, s_{t+1})$ cho mọi quỹ đạo trong $P_0$.
   * Bước 2c (**Kiểm tra bảo toàn thứ tự - Order-Preserving Test**): Kiểm tra xem với mọi cặp $(\tau^+, \tau^-) \in P_0$, bất đẳng thức sau có thỏa mãn không:
     $$G(\tau^+) > G(\tau^-)$$
3. **Điều kiện Dừng sớm (Early Stopping của CARD) vs. Tiếp tục sửa lỗi:**
   * **Nếu $G(\tau^+) > G(\tau^-)$ đúng với mọi cặp trong $P_0$:** Thuật toán kết luận hàm phần thưởng đã đạt chuẩn TPE (**PASS**) và dừng vòng lặp sớm.
   * **Nếu tồn tại ít nhất một cặp vi phạm $G(\tau^+) \le G(\tau^-)$:** Thuật toán gọi `feedback_generator.py` tổng hợp đủ **3 loại Feedback** và gọi Coder LLM sinh hàm phần thưởng mới cho vòng tiếp theo.

#### B. Cấu trúc 3 loại Feedback bắt buộc phải có của CARD (`feedback_generator.py`)

1. **Process Feedback (Phản hồi quy trình & thực thi):**
   * Báo cáo trạng thái biên dịch Python, kiểm tra kiểu trả về và cảnh báo các **thành phần phần thưởng bị thoái hóa (degenerate components)** (luôn bằng $0$ hoặc không đổi trên suốt quỹ đạo).
2. **Trajectory Feedback (Phản hồi phân rã theo nhóm quỹ đạo):**
   * Thống kê giá trị trung bình và độ lệch chuẩn của tổng return cũng như từng thành phần phần thưởng riêng lẻ trên nhóm quỹ đạo thành công ($\tau^+$) so với nhóm quỹ đạo thất bại ($\tau^-$).
3. **Preference Feedback (Phản hồi vi phạm thứ tự ưu tiên TPE):**
   * Liệt kê cụ thể các cặp quỹ đạo $(\tau^+, \tau^-) \in P_0$ bị hàm phần thưởng xếp hạng ngược (cả trên DoorKey lẫn Meta-World).

#### C. Phân biệt chính xác giữa `CARD` và `CARD_REFL`

* **`CARD` (Standard Static-TPE):** Tại vòng sửa thứ $i$, prompt chỉ chứa đặc tả bài toán, code của vòng $i-1$ và 3 khối Feedback của riêng vòng $i-1$.
* **`CARD_REFL` (CARD + Text Reflection Memory):** Tại vòng sửa thứ $i$, hệ thống duy trì một bộ đệm lịch sử văn bản (text memory) nối toàn bộ các lỗi TPE đã gặp và lời tự nhận xét từ vòng $0$ đến vòng $i-1$ vào prompt.

---

## 3. Checklist Nghiệm thu Logic của Eureka, CARD và Tích hợp Meta-World

Trước khi bấm chạy toàn bộ thực nghiệm trên Kaggle GPU, hãy dùng bảng kiểm tra (Checklist) dưới đây để rà soát từng module.

### 3.1. Checklist kiểm tra tính công bằng chung (System-wide Sanity Check)

* [ ] **Cùng điểm xuất phát:** Cả `ORIG`, `EUREKA`, `CARD`, `CARD_REFL` và `CEGR` đều đọc từ đúng một file manifest (`manifest_planted_115.json` cho DoorKey và `manifest_metaworld.json` cho Meta-World).
* [ ] **Cùng ngân sách LLM ($N = 4$):** Tổng số lần hàm sinh code của LLM được gọi trong mỗi job không bao giờ vượt quá $4$ lần cho bất kỳ phương pháp nào (Eureka: $2 \text{ vòng} \times 2 \text{ mẫu} = 4$; CARD / CARD_REFL / CEGR: tối đa $4 \text{ vòng} \times 1 \text{ mẫu} = 4$).
* [ ] **Cùng chữ ký hàm đầu ra:** Mọi prompt trên cả DoorKey và Meta-World đều yêu cầu LLM trả về hàm Python có chữ ký thống nhất `def reward_components(s, a, s2) -> dict:`.
* [ ] **Độc lập hoàn toàn với bộ chấm điểm Oracle:** Tuyệt đối không có phương pháp nào trong `baselines/` hay `synthesis/` được phép gọi hàm `verified_fix.py` trong quá trình tìm kiếm.

### 3.2. Checklist nghiệm thu riêng cho EUREKA (`cegr/baselines/eureka/`)

* [ ] **Kích thước quần thể & Số thế hệ:** Code trong `evolutionary_search.py` chạy đúng $G = 2$ thế hệ, mỗi thế hệ sinh $K = 2$ candidates độc lập (với `temperature > 0`).
* [ ] **Thực sự chạy RL trong vòng lặp:** Mỗi candidate hợp lệ sinh ra ở Thế hệ 1 và Thế hệ 2 thực sự gọi `rl_probe.py` (trên DoorKey) hoặc `continuous_rl_probe.py` (trên Meta-World) và cộng dồn số episodes/steps đã train vào biến `rl_episodes_used`.
* [ ] **Logic chọn lọc Elitism:** Sau Thế hệ 1, code chọn đúng candidate đạt `task_success_rate` cao nhất để làm gốc (parent) sinh ra Thế hệ 2.
* [ ] **Định dạng bảng thống kê Reflection:** Hàm `reflection_prompt.py` trích xuất được mảng giá trị theo 5 mốc thời gian huấn luyện cho **từng key thành phần** trong `reward_components`.
* [ ] **Không dùng bộ lọc của CEGR:** Đảm bảo Eureka hoàn toàn **không** gọi `static_antihack.py` (G2), **không** gọi `tpe_evaluator.py` (G3) và **không** dùng `dp_solver.py` / `trajectory_attacker.py`.

### 3.3. Checklist nghiệm thu riêng cho CARD & CARD_REFL (`cegr/baselines/card/`)

* [ ] **Đánh giá TPE chính xác:** Hàm `tpe_evaluator.py` tính đúng tổng chiết khấu $G(\tau) = \sum_{t=0}^{\vert\tau\vert-1} \gamma^t R(s_t, a_t, s_{t+1})$ và kiểm tra điều kiện bảo toàn thứ tự nghiêm ngặt $G(\tau^+) > G(\tau^-)$ trên mọi cặp $(\tau^+, \tau^-) \in P_0$.
* [ ] **Tính chất Tĩnh của bộ đệm trong Static-TPE:** Đảm bảo trong suốt 4 vòng lặp của `CARD` và `CARD_REFL`, tập quỹ đạo kiểm tra cố định là $P_0$ (không gọi DP Attacker hay Continuous Trajectory Attacker của CEGR để chèn thêm phản ví dụ mới vào $P_0$).
* [ ] **Đủ 3 khối Feedback trong Prompt:** Chuỗi prompt sinh ra từ `feedback_generator.py` khi trượt TPE chứa đủ 3 mục: *Process Feedback*, *Trajectory Feedback*, và *Preference Feedback*.
* [ ] **Phân tách rõ cờ `use_reflection_memory`:** Phân biệt rõ chế độ không lưu lịch sử (`CARD`) và chế độ nối chuỗi lịch sử phản hồi văn bản (`CARD_REFL`).
* [ ] **Ghi nhận đúng chỉ số Lỗi tái phát (`recurrence_count`):** Ở bước hậu kiểm (post-hoc logging), hệ thống đếm chính xác số phản ví dụ đã gặp ở các vòng trước mà hàm phần thưởng cuối cùng vẫn vi phạm.

### 3.4. Checklist nghiệm thu tích hợp môi trường Meta-World (`cegr/envs/metaworld_env.py` & `solvers/`)

* [ ] **Chuẩn hóa State Dictionary có ngữ nghĩa:** `metaworld_env.py` phải chuyển đổi mảng numpy thô `obs` (39 chiều của Meta-World) thành dictionary rõ nghĩa cho LLM (`s["hand_pos"]`, `s["obj_pos"]`, `s["goal_pos"]`, `s["gripper_open"]`, `s["obj_grasped"]`, `s["in_target"]`) để LLM viết hàm thế năng khoảng cách chính xác.
* [ ] **Sandbox hỗ trợ toán học vector (`code_sandbox.py`):** Cho phép hàm `reward_components(s, a, s2)` sử dụng các hàm an toàn trong `math` và `numpy` (như `np.linalg.norm`, `np.clip`, `np.exp`, `np.tanh`) để tính khoảng cách Euclid 3D mà không vi phạm quy tắc bảo mật AST ở Cổng G1.
* [ ] **Hoạt động của Continuous Attacker ở Cổng G3 (`trajectory_attacker.py`):** Khi chạy trên Meta-World (không giải được DP dạng bảng), Cổng G3 phải kích hoạt bộ tìm kiếm quỹ đạo đối kháng liên tục (dùng Cross-Entropy Method hoặc khai thác tập quỹ đạo nhiễu) để phát hiện các hành vi cày điểm như: *đưa tay máy tới sát vật thể rồi rung lắc qua lại*, hoặc *đóng/mở kẹp gắp liên tục tại chỗ*.