# ĐẶC TẢ HỆ THỐNG LOGIC CHUẨN: CEGR & BASELINES (EUREKA, CARD)

**Tài liệu tham chiếu chuẩn kiểm tra logic mã nguồn (Single Source of Truth for Logic Audits) — phiên bản hợp nhất**

> **CHỈ DẪN BẮT BUỘC CHO AI ASSISTANT:**
> Mỗi khi người dùng yêu cầu: _"kiểm tra lỗi logic"_, _"kiểm tra code"_, _"so sánh logic"_ hoặc rà soát mã nguồn trong dự án này:
>
> 1. **BẮT BUỘC ĐỌC FILE NÀY ĐẦU TIÊN** để làm thang đo và tiêu chuẩn nghiệm thu logic cho từng file và hàm.
> 2. **ƯU TIÊN ĐỘ CHÍNH XÁC HỌC THUẬT TUYỆT ĐỐI:** Không áp dụng các thủ thuật cắt giảm tính toán làm sai lệch bản chất thuật toán.
> 3. Luôn đối chiếu với các quy tắc bất biến, cấu trúc dữ liệu và điều kiện biên dưới đây.
> 4. Khi code khác spec: **báo lỗi và nêu vị trí**, không tự sửa spec cho khớp code. Khi spec mâu thuẫn với báo cáo thực nghiệm: đối chiếu **Mục 6** (các khác biệt đã được chủ ý) và **Mục 7** (các điểm chưa chốt) trước khi kết luận.
> 5. Các mục đánh dấu **〔Mục 7.x〕** là quyết định thiết kế hoặc giá trị chưa được báo cáo xác nhận; khi rà soát chỉ cần kiểm tra code có khớp quyết định đó, không coi là lỗi nếu đã khớp.

---

## 0. QUY ƯỚC TÊN VÀ HẰNG SỐ CHUẨN

### 0.1. Quy ước tên

| Tên dùng trong spec                                                | Tên trong báo cáo / cấu hình | Ghi chú                                        |
| :----------------------------------------------------------------- | :--------------------------- | :--------------------------------------------- |
| `CEGR`                                                             | CEGR, `CEGR` (đầy đủ)        | Phương pháp đề xuất                            |
| `CEGR_noPool`, `_noAnchors`, `_noClass`, `_noG2`, `_noG4`, `_noG5` | Giống nhau                   | Tên ablation giữ nguyên trong `ablations.yaml` |
| `ORIG`, `EUREKA`, `CARD`, `CARD_REFL`                              | Giống nhau                   | Baseline                                       |

### 0.2. Bảng hằng số

| Hằng số                                           | Giá trị                                                                                               | Nơi dùng                             |
| :------------------------------------------------ | :---------------------------------------------------------------------------------------------------- | :----------------------------------- |
| $N$ (số lần gọi Coder LLM tối đa mỗi job)         | 4                                                                                                     | Mọi phương pháp                      |
| Số seed LLM độc lập mỗi (reward gốc, phương pháp) | 3                                                                                                     | Điều phối thực nghiệm                |
| Chân trời DoorKey                                 | $T = 30$; kiểm tra kép $2T = 60$                                                                      | DP, Oracle                           |
| Số trạng thái khả đạt DoorKey 5x5                 | 270                                                                                                   | DP                                   |
| `POOL_MAX`                                        | 40 (FIFO)                                                                                             | CE pool                              |
| `eps_rel`                                         | 0.02 (tương đối)                                                                                      | `classify_pair`, G4                  |
| Ngưỡng Spearman G4                                | $\rho \ge 0.75$                                                                                       | G4                                   |
| Giới hạn thang đo reward                          | $\lvert v\rvert \le 10^4$                                                                             | G1                                   |
| G5 DoorKey                                        | 1000 episodes × 2 seeds, `tail_success_rate` ≥ 0.50                                                   | G5                                   |
| Probe của Eureka và Oracle (DoorKey)              | 2500 episodes × 3 seeds, $\gamma = 1.0$, khởi tạo lạc quan, giảm $\alpha$, `tail_success_rate` ≥ 0.50 | Eureka fitness, Oracle               |
| Số cặp nhiễu tối đa trong $P_0$                   | 20                                                                                                    | TPE, G3                              |
| Seed                                              | CE/tìm kiếm = `1000`, G4 = `2000`, Oracle = `3000`                                                    | Toàn hệ thống                        |
| Nhiệt độ sinh / sửa                               | 0.95 / 0.35 〔Mục 7.1〕                                                                               | Mọi phương pháp (cùng bảng nhiệt độ) |

---

## 1. NGUYÊN TẮC BẤT BIẾN CỐT LÕI (SYSTEM-WIDE INVARIANTS)

1. **Tính công bằng ngân sách (Budget Fairness):**
   - Mọi phương pháp dùng chung mô hình `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` và tuân thủ ngân sách tối đa $N = 4$ lần gọi **Coder** LLM mỗi job.
   - Lần gọi LLM của **Reward Architect** (sinh `anchors_v0` của CEGR) **không tính vào $N$**; phải được ghi nhận riêng trong log (`architect_calls`) và nêu rõ khi báo cáo "gọi LLM".
   - `ORIG`: 0 lần gọi (đánh giá trực tiếp $R_0$).
   - `EUREKA`: đúng $G = 2$ thế hệ $\times$ $K = 2$ ứng viên = 4 mẫu sinh.
   - `CARD` / `CARD_REFL` / `CEGR`: tối đa $N = 4$ vòng $\times$ 1 ứng viên. **Chỉ dừng sớm khi PASS toàn bộ cổng kiểm định của chính phương pháp đó.** Không dừng sớm vì pool bão hòa hay không thêm được CE mới.
   - Bộ nhớ văn bản của `CARD_REFL` chỉ là chuỗi nối feedback do luật sinh ra, **không** tốn thêm lần gọi LLM nào.
2. **Chữ ký giao diện thống nhất:**
   - Mọi hàm phần thưởng do LLM sinh trên `Minigrid DoorKey` và `Meta-World` có chữ ký:
     ```python
     def reward_components(s, a, s2) -> dict:
     ```
   - `s` là dict trạng thái trước hành động, `a` là hành động, `s2` là dict trạng thái sau hành động.
   - Trả về `{component_name: float/int/bool}`; reward tại một bước chuyển tiếp là tổng các component.
3. **Độc lập với bộ chấm Oracle:**
   - Không module nào trong `synthesis/` hoặc `baselines/` được gọi `evaluate_verified_fix()` của `cegr/solvers/verified_fix.py`. Oracle chỉ chạy post-hoc sau khi tìm kiếm kết thúc, trên reward cuối.
   - Ba luồng seed tách biệt hoàn toàn: **1000** (sinh CE, tìm kiếm, G3, G5, RL fitness của Eureka), **2000** (G4), **3000** (Oracle). Không bao giờ dùng seed 3000 trong tìm kiếm.
   - **Giới hạn đã biết:** độc lập này chỉ ở mức đường gọi và seed. Oracle DoorKey dùng cùng `dp_solver.py` với attacker G3, và Oracle Meta-World dùng cùng cơ chế attacker (ngân sách mở rộng) với G3. Do đó chỉ số hack theo DP/attacker có thiên lệch có lợi cho CEGR; chỉ số `Final RL success` và `hack rate RL` độc lập hơn (xem Mục 6.3).
4. **Điểm xuất phát chung:** Mọi phương pháp đọc cùng một manifest reward gốc (`manifest_planted_115.json` cho DoorKey, `manifest_metaworld.json` cho Meta-World) và cùng tập $P_0$.
5. **Đối xử với $R_0$ nhất quán:** $R_0$ không tính vào số vòng $N$. Với `CARD`, `CARD_REFL`, `CEGR`, $R_0$ được chấm **một lần** bằng chính bộ kiểm tra của phương pháp đó để sinh feedback cho vòng 1 (không tốn lần gọi LLM). 〔Mục 7.2〕

---

## 2. ĐẶC TẢ LOGIC TỪNG PHÂN HỆ & MODULE

### 2.1. Phân hệ Hạ tầng Thực thi & Đánh giá Cơ sở (`cegr/core/`)

#### A. Sandbox an toàn & Biên dịch mã (`cegr/core/code_sandbox.py`)

- **Mục tiêu:** Biên dịch chuỗi code LLM trong môi trường cách ly (Cổng G1).
- **Quy tắc:**
  - **Kiểm tra AST (`check_code`):** cho phép import `math`, `numpy`, `typing`. Chặn `eval`, `exec`, `open`, `compile`, `globals`, `locals`, `getattr`, `setattr`, v.v. và mọi thuộc tính dunder bắt đầu bằng `__`. (Khác bản đã chạy trong báo cáo, xem Mục 6.1.)
  - **Namespace (`make_namespace`):**
    - `SAFE_BUILTINS` phải chứa `__import__` an toàn (chỉ nạp `math`, `numpy`, `typing`) để câu lệnh `import numpy as np` không gây `ImportError`.
    - `SAFE_NP` gồm: `np.linalg.norm`, `clip`, `exp`, `tanh`, `abs`, `sum`, `mean`, `min`, `max`, `asarray`, `array`, `dot`, `sqrt`, `atan2`, `cos`, `sin`.
  - **Kiểm tra thực thi:**
    - Miền rời rạc (`build_reward_table`): chạy trên toàn bộ bảng chuyển trạng thái–hành động, bỏ qua trạng thái kết thúc (`env.terminal[i] == True`), trả về `RewardTable(R=C.sum(-1), C=C, names=names)`.
    - Miền liên tục (`check_rollout_exec`): chạy trên mọi bước chuyển $(s, a, s')$ của tập quỹ đạo $P_0$.
    - Mọi giá trị ép về `float`, thỏa `math.isfinite(v)` và $\lvert v\rvert \le 10^4$.

#### B. Thuật toán TPE (`cegr/core/tpe_evaluator.py`)

- **Mục tiêu:** Lõi đánh giá TPE dùng chung cho `CARD`, `CARD_REFL` và G3 của `CEGR`.
- **Quy tắc:**
  - **Tổng chiết khấu:** $G(\tau) = \sum_{t=0}^{|\tau|-1} \gamma^t R(s_t, a_t, s_{t+1})$; mọi component đều nhân $\gamma^t$. Giá trị $\gamma$ cho TPE và DP lấy từ config, **cùng một giá trị**. 〔Mục 7.3〕
  - **Hai chế độ pass, không được lẫn lộn:**
    1. **`strict` (dùng cho `CARD`, `CARD_REFL`):** cặp $(\tau^+, \tau^-)$ đạt khi $R_b - R_w > 10^{-9}$. Cặp "đúng thứ tự nhưng chênh nhỏ" vẫn **pass**. Không dùng `eps_rel` trong chế độ này.
    2. **`margin` (dùng cho G3 của `CEGR`):** cặp đạt khi `classify_pair(...) is None`.
  - **`classify_pair(R_b, R_w, eps_rel=0.02)`:**
    - $\text{thr} = \max(10^{-4},\ \text{eps\_rel} \times \lvert R_b\rvert)$.
    - **Thứ tự kiểm tra bắt buộc (if–elif), để các nhãn loại trừ nhau:** (1) `hack` nếu $R_w \ge R_b - 10^{-9}$ (tức $R_b - R_w \le 10^{-9}$); chỉ khi không phải `hack` mới xét (2).
    - (2) `under_signal` nếu $R_b - R_w > 10^{-9}$ và $R_b - R_w < \text{thr}$. Vì $\text{thr} \ge 10^{-4} > 10^{-9}$ nên ba nhãn `hack` / `under_signal` / `None` phân hoạch toàn bộ trục $R_b - R_w$. Dùng $\lvert R_b\rvert$ nên đúng cả khi return âm (ví dụ $R_b = -5$ cho $\text{thr} = 0.1$).
    - `None`: $R_b - R_w \ge \text{thr}$.
    - Ghi chú: vùng gần hòa được coi là `hack` ở cổng ($\ge R_b - 10^{-9}$) nhưng Oracle chỉ coi là hacked khi $V_{\text{fail}} > V_{\text{succ}} + 10^{-6}$. Cổng chặt hơn Oracle là **có chủ ý**.
  - **`pair_margin` đa miền:**
    - Rời rạc: tính qua `traj_return(env, table, actions)` từ mảng chỉ số hành động `int`.
    - Liên tục: đánh giá lại `reward_fn(st.s, st.a, st.s2)` trên từng bước của `better_steps`/`worse_steps`. Không dùng return tĩnh đã đóng băng.
  - **Tập tham chiếu $P_0$:** 4 cặp tham chiếu $\{(T_3, T_2), (T_2, T_1), (T_1, T_0), (T_3, T_0)\}$ (3 cặp mốc liền kề + 1 cặp neo đầu–cuối; tiến độ thật của $T_0 \dots T_3$ = 0, 1, 2, 3 = chưa gì, nhặt chìa, mở cửa, tới goal), cộng tối đa 20 cặp nhiễu. **`CARD`, `CARD_REFL` và G3 của `CEGR` dùng cùng một $P_0$** (cả cặp nhiễu), sao cho khác biệt giữa chúng chỉ nằm ở CE pool, attacker và các cổng khác. 〔Mục 7.4〕

---

### 2.2. Phân hệ Bộ giải Toán học, RL Probe & Oracle (`cegr/solvers/`)

#### A. Quy hoạch động chính xác (`cegr/solvers/dp_solver.py`)

- **Mục tiêu:** Tính $V_{\text{succ}}$ (giá trị tối ưu của quỹ đạo thành công) và $V_{\text{fail}}$ (giá trị tối ưu của quỹ đạo không chạm goal) trên bảng trạng thái DoorKey.
- **Quy tắc:**
  - `succ = np.full(n, -np.inf)`, `fail = np.zeros(n)`.
  - Lặp ngược $t = T-1 \dots 0$:
    - $c_f = R + \gamma \cdot fail[nxt]$, đặt $c_f = -\infty$ tại các chuyển tiếp chạm đích (`gf == True`).
    - $c_s = R + \gamma \cdot \text{where}(gf, 0.0, succ[nxt])$.
  - Phán quyết hack: `hacked = (V_fail > V_succ + 1e-6)` tại chân trời được xét.
  - `dp_attack()`: truy vết $\tau_{\text{succ}}^*$ và $\tau_{\text{fail}}^*$, phân loại qua `classify_pair(V_succ, V_fail, eps_rel=0.02)`. Gọi `classify_pair` trực tiếp, không lọc thô trước, để bắt cả `hack` lẫn `under_signal`.

#### B. Bộ tấn công liên tục (`cegr/solvers/trajectory_attacker.py`)

- **Mục tiêu:** Tìm quỹ đạo cày điểm mà không hoàn thành nhiệm vụ trên Meta-World (G3).
- **Quy tắc:**
  - Hai nguồn tìm kiếm:
    1. **CEM:** tối ưu chuỗi hành động $a_t \sim \mathcal{N}(\mu_t, \sigma_t^2)$, clip về $[-1, 1]$, `score = r_cand - (1e4 if is_success else 0.0)`.
    2. **Perturbation / Buffer Mining:** khai thác quỹ đạo nhiễu lặp (đóng/mở gripper tại chỗ, rung sát vật thể).
  - **Phán quyết:** `classify_pair(ref_return, best_exploit_return, eps_rel=0.02)`, **không** lọc thô `best_exploit_return >= ref_return` trước đó.
  - Lưu đầy đủ `better_steps` và `worse_steps` vào CE.

#### C. RL Probes (`rl_probe.py`, `continuous_rl_probe.py`)

- **`rl_probe.py` (DoorKey, Tabular Q-learning):**
  - Chuẩn hóa $R_{\text{norm}} = R / \max(1.0, \max\lvert R\rvert)$; khởi tạo lạc quan $Q = 1.0$; giảm dần $\epsilon$ và $\alpha$.
  - `tail_success_rate` = trung bình tỷ lệ thành công của 25% điểm đánh giá cuối, lấy trung bình qua các seed.
  - **G5:** 1000 episodes × 2 seeds, pass khi $\ge 0.50$ (seed luồng 1000).
  - **Probe của Eureka và Oracle:** 2500 episodes × 3 seeds, $\gamma = 1.0$, pass/fitness theo `tail_success_rate` (seed luồng 1000 cho Eureka, luồng 3000 cho Oracle).
  - Phải ghi lại đường cong học theo **từng component** của reward tại 5 mốc 20/40/60/80/100% để phục vụ Eureka Reflection.
- **`continuous_rl_probe.py` (Meta-World):**
  - SAC hoặc PPO chuẩn (MLP 2 lớp ẩn); Short Probe ở G5 dùng 20.000–50.000 environment steps; Oracle dùng Full evaluation (seed 3000).
  - Ghi đường cong theo từng component tại 5 mốc như trên. Ngưỡng G5 liên tục lấy từ `metaworld_config.yaml`. 〔Mục 7.5〕
- **Trích xuất CE khi trượt G5:** nếu probe không đạt ngưỡng, rollout chính sách vừa học để lấy quỹ đạo thất bại thực tế làm CE đưa vào `CEPool`.

#### D. Bộ chấm Oracle độc lập (`cegr/solvers/verified_fix.py`)

- **Mục tiêu:** Chấm reward cuối của mọi phương pháp, post-hoc, seed 3000.
- **`verified_fix = True` khi cả ba điều kiện cùng đúng:**
  1. **Không bị hack (`no_hack_oracle`):**
     - DoorKey (`loose_dp`): DP chính xác ở **cả $T = 30$ và $T = 60$**; ở mỗi chân trời $V_{\text{fail}} \le V_{\text{succ}} + 10^{-6}$ và $V_{\text{succ}} - V_{\text{fail}} > 0.01 \cdot \lvert V_{\text{succ}}\rvert$.
     - Meta-World: attacker với ngân sách mở rộng tìm quỹ đạo thất bại có return cao nhất $R_{\text{exploit}}^{\max}$. Reward chỉ được coi là **không bị hack** khi $R_{\text{expert}} - R_{\text{exploit}}^{\max} > 0.01 \cdot \lvert R_{\text{expert}}\rvert$ (dạng hiệu tuyệt đối, đúng cả khi return âm; **không** dùng phép nhân 99% vì sai dấu khi $R_{\text{expert}} < 0$). Cùng dạng với margin 1% của DoorKey.
  2. **Có định hình tiến độ thực chất (`subgoal_shaping`):** không phải reward thưa thoái hóa; reward trên các mốc tăng dần $T_0 < T_1 < T_2 < T_3$.
  3. **Học được bằng RL độc lập (`rl_eval_success`):** DoorKey: 2500 episodes × 3 seeds, $\gamma = 1.0$, `tail_success_rate` $\ge 0.50$. Meta-World: Full SAC/PPO, `tail_success_rate` $\ge 0.50$.
- Có thể báo cáo thêm `tier1_safe_aligned` (đạt điều kiện 1 và 3, chấp nhận reward thưa thuần túy) như chỉ số phụ; **chỉ số chính là `verified_fix`** (tương đương `tier2_dense_vf`).

---

### 2.3. Phân hệ Baseline Đối sánh (`cegr/baselines/`)

#### A. EUREKA (`cegr/baselines/eureka/`)

Bài báo gốc: _Ma et al., ICLR 2024_ (bản triển khai này là bản rút gọn, xem Mục 6.2).

- **Đánh giá khởi tạo:** nếu $R_0$ qua G1 (chỉ G1, không G2/G3/G4), chạy RL probe (DoorKey: 2500 × 3; seed luồng 1000) để lấy đường cong theo component và `task_success_rate` làm fitness của $R_0$.
- **Tiến hóa:** $G = 2$ thế hệ, mỗi thế hệ $K = 2$ ứng viên độc lập sinh song song (tổng 4 mẫu). Nhiệt độ theo bảng 0.2: Thế hệ 1 dùng nhiệt độ sinh, Thế hệ 2 dùng nhiệt độ sửa. 〔Mục 7.1〕
- **RL bắt buộc:** mọi ứng viên hợp lệ (qua G1) đều chạy RL đầy đủ; cộng dồn vào `rl_episodes`. $R_0$ cũng tính vào `rl_episodes`.
- **Chọn cha mẹ cho Thế hệ 2:** chọn ứng viên có `task_success_rate` cao nhất **trong thế hệ 1**. Chỉ fallback về $R_0$ khi mọi ứng viên của thế hệ 1 đều lỗi G1 hoặc có success rate bằng 0.
- **Chọn $R_{\text{final}}$:** ứng viên hợp lệ có `task_success_rate` cao nhất trong **toàn bộ các ứng viên đã sinh** (Thế hệ 1 và 2). Hòa thì chọn ứng viên sinh sớm hơn. Chỉ trả về $R_0$ nếu không có ứng viên hợp lệ nào. 〔Mục 7.6〕
- **Reward Reflection (`reflection_prompt.py`):** với cá thể cha mẹ được chọn, trích giá trị trung bình **từng component** cùng `task_success_rate` tại đúng 5 mốc 20/40/60/80/100% quá trình huấn luyện.
- **Cấm:** G2, G3, G4, DP attacker, continuous attacker, `verified_fix.py`.

#### B. CARD & CARD_REFL (`cegr/baselines/card/`)

Bài báo gốc: _Sun et al., 2024_ (bản triển khai là biến thể TPE tĩnh, xem Mục 6.2).

- **Bộ đệm tĩnh:** chỉ gồm $P_0$ (đúng định nghĩa ở 2.1B) trong suốt các vòng. Không chèn CE từ attacker hay probe.
- **Cổng sử dụng:** G1 và TPE chế độ `strict`. Không G2, G4, G5.
- **Dừng sớm:** nếu mọi cặp trong $P_0$ thỏa $G(\tau^+) > G(\tau^-)$ (`strict`), kết luận PASS và dừng.
- **Ba khối feedback bắt buộc khi trượt (`feedback_generator.py`):**
  1. _Process Feedback:_ trạng thái thực thi và cảnh báo component thoái hóa (luôn 0 hoặc hằng số).
  2. _Trajectory Feedback:_ mean và std của tổng return và từng component trên nhóm $\tau^+$ so với $\tau^-$.
  3. _Preference Feedback:_ danh sách các cặp $(\tau^+, \tau^-) \in P_0$ bị xếp ngược ($G(\tau^+) \le G(\tau^-)$).
- **`CARD` vs `CARD_REFL`:** `CARD` chỉ gửi feedback của vòng ngay trước; `CARD_REFL` nối toàn bộ feedback văn bản từ vòng 0 (feedback chấm $R_0$) đến vòng $i-1$. Không có lần gọi LLM bổ sung. Cờ cấu hình: `use_reflection_memory`.
- **Reward cuối khi không PASS:** xem 2.4E.

---

### 2.4. Phân hệ Đề xuất CEGR (`cegr/synthesis/`)

#### A. Vòng lặp chính (`runner.py`)

1. **Architect:** sinh `anchors_v0` bằng LLM từ mô tả nhiệm vụ (không tính vào $N$).
2. Chấm $R_0$ một lần bằng GateKeeper để sinh feedback ban đầu (không tính vào $N$). 〔Mục 7.2〕
3. Lặp $i = 1 \dots N$:
   - Coder sinh $R_i$ từ ($R_{i-1}$ hoặc $R_0$ ở vòng 1, `anchors` hiện hành, feedback, kinh nghiệm do Assimilator rút ra).
   - GateKeeper kiểm tra $R_i$. PASS thì dừng.
   - Nếu FAIL: trích CE theo cổng, thêm vào `CEPool`, `Assimilator` cập nhật feedback và (nếu bật) `anchors`.
4. **Điều kiện dừng duy nhất:** PASS hoặc hết $N$ vòng. (Khác bản đã chạy: không còn dừng theo `STALL_K`, xem Mục 6.1.)
5. Các job được chạy theo wave để gộp batch cho vLLM; việc gộp batch không được thay đổi logic từng job.

#### B. GateKeeper 5 cổng (`gatekeeper.py`)

Kiểm tra từ rẻ đến đắt:

- **G1 (Exec):** biên dịch AST an toàn, không `NaN`/`Inf`, $\lvert R\rvert \le 10^4$. Lỗi thì dừng ngay, trả về lỗi biên dịch.
- **G2 (Static Anti-Hack, `static_antihack.py`):** (a) thiếu term goal, (b) hằng số dương vô điều kiện (survival bonus), (c) thưởng dương cho no-op, (d) shaping vượt độ lớn reward goal. **Chỉ trên Meta-World:** (e) thưởng thuận theo khoảng cách Euclid (`+norm(hand - goal)`) hoặc thưởng lớn cho vận tốc gây rung.
- **G3 (TPE + CE pool + attacker):** TPE chế độ `margin` trên $P_0$ ∪ toàn bộ `CEPool`, và vượt `dp_attack` (DoorKey) hoặc `trajectory_attacker` (Meta-World). G3 phải thực sự chạy trên $P_0$ ở cả hai miền, không được bị bỏ qua.
- **Quy tắc phối hợp G2–G3 (Merge Mode):** sau khi qua G1, chạy cả G2 và G3.
  - G2 FAIL và G3 FAIL: **ưu tiên CE của G3** đưa vào `CEPool` và `Assimilator`; feedback có thể nêu thêm vi phạm G2.
  - G2 FAIL và G3 PASS: trượt tại G2, không chạy G4/G5, không thêm CE (feedback là luật G2 bị vi phạm).
  - G2 PASS: xét G3 như bình thường.
- **G4 (Signal, `signal_checker.py`, seed 2000):** trên tập quỹ đạo kiểm tra G4 (các quỹ đạo có tiến độ thật phân biệt được), xét **mọi cặp** $(T_a \succ T_b)$ với tiến độ $a > b$: (a) mọi cặp thỏa `classify_pair(...) is None` (đúng thứ tự và chênh $\ge \text{thr}$); (b) phương sai reward theo giai đoạn khác 0; (c) Spearman(return, tiến độ thật) $\ge 0.75$.
  - **Trích CE khi trượt G4:** mọi cặp vi phạm đều được lưu vào `CEPool`: `hack` nếu đảo thứ tự hoặc hòa ($R_w \ge R_b - 10^{-9}$), `under_signal` nếu đúng thứ tự nhưng chênh $< \text{thr}$. Điều kiện (b) không gắn với cặp cụ thể và (c) mà không có cặp nào vi phạm thì chỉ sinh feedback, không sinh CE. 〔Mục 7.10〕
- **G5 (Short RL Probe):** theo 2.2C. Nếu trượt, rollout policy vừa học để trích CE.

#### C. Kho phản ví dụ (`ce_pool.py`)

- FIFO, dung lượng `POOL_MAX = 40`. Hai loại nhãn: `hack`, `under_signal`.
- **Nguồn CE:** `hack`: attacker (DP/CEM), cặp nhiễu của $P_0$, rollout của probe, cặp đảo thứ tự ở G4; `under_signal`: cặp đúng thứ tự nhưng chênh dưới thr (G3/G4).
- **Khử trùng lặp kép:** một CE bị coi là trùng nếu trùng **mã băm nội dung** (MD5 12 ký tự của `better_str|worse_str|type`) HOẶC trùng **khóa chữ ký** $(\text{type},\ \text{behavior\_signature},\ \text{progress\_pair})$, trong đó `behavior_signature` là hành động chiếm đa số của quỹ đạo khai thác (`pickup_loop`, `toggle_loop`, `gripper_spam`, ...) và `progress_pair` là cặp mức tiến độ của $(\tau^+, \tau^-)$. CE trùng bị bỏ, phần tử cũ giữ nguyên và không được đẩy lại vị trí FIFO. 〔Mục 7.7〕
- Lưu đầy đủ mảng hành động `better`/`worse` (rời rạc) và `better_steps`/`worse_steps` (liên tục) để `replay` trên mọi reward mới, bằng chính chế độ `margin` của TPE.

#### D. Assimilator (`assimilator.py`) và Dynamic Anchors (`dynamic_anchors.py`)

- **Anchors:**
  - `anchors_v0` (do Architect sinh, gồm khoảng 6 mệnh đề): chạm goal cho tổng return lớn nhất; term tiến độ phải là một lần hoặc dạng hiệu thế năng $\Phi(s') - \Phi(s)$; đứng yên hoặc xoay tại chỗ không được thưởng dương; phải nhặt chìa trước khi tương tác cửa.
  - Nâng `v0` lên `v1` khi `Assimilator` phát hiện lỗ hổng đặc tả từ CE. Chế độ `noAnchors` khóa ở `v0`.
- **Assimilator (dạng luật, không dùng LLM):**
  - Đầu vào: CE (nhãn `hack`/`under_signal`, chữ ký hành vi, bảng phân rã component trên $\tau^+$ vs $\tau^-$).
  - Rút luật kinh nghiệm theo nhãn, ví dụ `hack` + chữ ký `toggle` → _"chỉ thưởng mở cửa một lần hoặc dùng hiệu thế năng trạng thái cửa"_; `under_signal` → _"tăng hệ số định hình cho mốc tiến độ bị phẳng"_.
  - Cập nhật cả feedback cho Coder lẫn `anchors` (chế độ động).
  - Khi `noClass`, nhãn gộp thành `"generic"` nên chỉ sinh feedback chung.

#### E. Reward cuối khi không PASS (áp dụng cho `CARD`, `CARD_REFL`, `CEGR`)

- Nếu không vòng nào PASS: $R_{\text{final}}$ là ứng viên hợp lệ (qua G1) có **độ sâu cổng lớn nhất**. Độ sâu = số cổng liên tiếp đã qua tính từ G1 theo thứ tự G1→G2→G3→G4→G5, chỉ tính các cổng đang bật; trong Merge Mode, G2 FAIL thì độ sâu dừng ở 1 dù G3 PASS. Hòa độ sâu thì chọn ứng viên ở **vòng muộn nhất**. Nếu không có ứng viên nào qua G1 thì $R_{\text{final}} = R_0$. Quy tắc chỉ dựa vào kết quả cổng, **không** dùng Oracle. 〔Mục 7.6〕
- Quy tắc áp dụng thống nhất cho `CARD`, `CARD_REFL`, `CEGR` và 6 ablation. Với `CARD`/`CARD_REFL` (chỉ có G1 và TPE `strict`) độ sâu chỉ có ít mức nên quy tắc suy biến thành "ứng viên qua G1 ở vòng muộn nhất". Vì vậy quy tắc này chỉ thực sự tạo khác biệt cho `CEGR` (tránh mất ứng viên tốt ở vòng giữa khi vòng cuối sửa quá tay) và phải được nêu rõ khi diễn giải kết quả.

#### F. Sáu biến thể Ablation (`configs/ablations.yaml`)

Cùng quy trình chuẩn, tắt đúng một thành phần:

1. `CEGR_noPool`: không tích lũy `CEPool` (chỉ $P_0$ và CE của vòng hiện tại; attacker vẫn chạy).
2. `CEGR_noAnchors`: giữ `anchors_v0`, không nâng lên `v1`.
3. `CEGR_noClass`: gộp nhãn thành `"generic"`.
4. `CEGR_noG2`: bỏ G2.
5. `CEGR_noG4`: bỏ G4.
6. `CEGR_noG5`: bỏ G5.

---

## 3. CHỈ SỐ ĐO VÀ HẬU KIỂM

- **Verified Fix (VF):** theo 2.2D, tính trên $R_{\text{final}}$, post-hoc.
- **Hack rate (DP):** $V_{\text{fail}} > V_{\text{succ}} + 10^{-6}$ ở $T = 30$ hoặc $T = 60$. **Hack rate (RL)** và **Final RL success:** từ RL độc lập, seed 3000.
- **Vòng đến PASS:** job không PASS được tính là $N + 1$. Chỉ số này **bị kiểm duyệt phải**, không diễn giải như thời gian hội tụ.
- **Lỗi tái phát (`recurrence_count`):** post-hoc, không ảnh hưởng tìm kiếm. Là số CE đã từng gặp ở các vòng trước của cùng job mà $R_{\text{final}}$ còn vi phạm khi replay (bằng `margin`). Lấy trung bình mỗi job. Với phương pháp không có CE pool, tập "đã gặp" được dựng post-hoc từ các vi phạm của chính các ứng viên ở vòng trước bằng bộ kiểm hậu kiểm riêng (DP + $P_0$). 〔Mục 7.8〕
- **Thống kê:** bootstrap cụm ở cấp reward (5000 lần), vì các seed LLM của cùng một reward không độc lập.

---

## 4. KIẾN TRÚC ĐA MIỀN (DoorKey ↔ Meta-World)

| Thành phần            | DoorKey (rời rạc)                                                                          | Meta-World (liên tục)                                                                                           |
| :-------------------- | :----------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| State `s`             | `agent_pos`, `agent_dir`, `has_key`, `door_open`, `goal_pos`                               | `hand_pos`, `obj_pos`, `goal_pos`, `gripper_open`, `obj_grasped`, `in_target`                                   |
| G2 thêm               | (không)                                                                                    | thưởng thuận khoảng cách Euclid, thưởng vận tốc gây rung                                                        |
| G3 attacker           | `dp_attack` (DP chính xác)                                                                 | `trajectory_attacker` (CEM + perturbation)                                                                      |
| G5                    | Q-learning 1000 × 2                                                                        | SAC/PPO 20.000–50.000 steps                                                                                     |
| Oracle chống hack     | DP kép $T = 30, 60$, $V_{\text{succ}} - V_{\text{fail}} > 1\%\lvert V_{\text{succ}}\rvert$ | Attacker ngân sách mở rộng, $R_{\text{expert}} - R_{\text{exploit}}^{\max} > 1\%\lvert R_{\text{expert}}\rvert$ |
| Phạm vi đã có số liệu | DoorKey 5x5 (báo cáo)                                                                      | Chưa có số liệu                                                                                                 |

---

## 5. CHECKLIST KIỂM TRA MÃ NGUỒN (18 ĐIỂM)

- [ ] **1. Safe Import:** `import numpy as np`, `import math`, `from typing import ...` chạy được trong `code_sandbox.py` nhờ `__import__` an toàn.
- [ ] **2. Ngân sách:** Số lần gọi Coder ≤ 4 mỗi job với mọi phương pháp; Eureka đúng 2 × 2; lần gọi Architect được log riêng, không tính vào 4.
- [ ] **3. Không dừng sớm vì pool:** vòng lặp CEGR chỉ dừng khi PASS hoặc hết $N$; không còn logic `STALL_K`.
- [ ] **4. TPE hai chế độ:** `CARD`/`CARD_REFL` dùng `strict` (không ε); G3 của CEGR dùng `margin` qua `classify_pair`.
- [ ] **5. `classify_pair`:** $\text{thr} = \max(10^{-4}, 0.02 \lvert R_b\rvert)$; kiểm tra theo thứ tự if–elif (`hack` trước, `under_signal` yêu cầu $R_b - R_w > 10^{-9}$); `trajectory_attacker.py` và `dp_solver.py` gọi trực tiếp, không lọc thô trước.
- [ ] **6. $P_0$ đồng nhất:** CARD, CARD_REFL và G3 dùng cùng $P_0$ (gồm cặp nhiễu ≤ 20); CARD không chèn thêm CE nào vào $P_0$.
- [ ] **7. G2–G3 Merge:** khi cả hai fail, CE của G3 được ưu tiên đưa vào `CEPool`/`Assimilator`; G2 fail + G3 pass thì dừng tại G2 và không thêm CE.
- [ ] **8. G2 đủ luật:** có luật shaping không vượt goal; luật +norm chỉ bật trên Meta-World.
- [ ] **9. Dedup kép của CE pool:** theo hash nội dung **và** khóa chữ ký $(\text{type}, \text{signature}, \text{progress\_pair})$; CE trùng bị bỏ, không đẩy lại FIFO; lưu `better`/`worse` và `better_steps`/`worse_steps` để replay.
- [ ] **10. Assimilator & Anchors:** phân nhánh riêng cho `hack` vs `under_signal`, nâng `v0` → `v1` khi cần; `noClass` thật sự gộp thành `"generic"`; `noAnchors` thật sự khóa `v0`.
- [ ] **11. Oracle kép:** `verified_fix.py` kiểm DP ở cả $T = 30$ và $60$ với margin 1%; dùng seed 3000; RL 2500 × 3, $\gamma = 1.0$.
- [ ] **12. Eureka:** mọi ứng viên hợp lệ chạy RL (seed luồng 1000, không 3000); reflection đủ 5 mốc cho từng component; parent chọn trong thế hệ 1 với fallback; $R_{\text{final}}$ chọn trong mọi ứng viên đã sinh; không gọi G2/G3/G4/attacker.
- [ ] **13. CARD:** bộ đệm tĩnh; feedback đủ 3 khối; `CARD_REFL` nối text lịch sử nhưng không tốn lần gọi LLM.
- [ ] **14. Probe liên tục:** `continuous_rl_probe.py` dùng SAC/PPO thật, không dùng chính sách tuyến tính; xuất đường cong 5 mốc theo component; trượt G5 thì rollout trích CE.
- [ ] **15. Reward cuối:** khi không PASS, chọn ứng viên qua G1 có độ sâu cổng lớn nhất (hòa thì vòng muộn nhất), fallback $R_0$; Eureka theo 2.3A.
- [ ] **16. Cô lập seed và hậu kiểm:** ba luồng 1000/2000/3000 tách biệt; `recurrence_count` ghi log post-hoc cho mọi phương pháp; duck typing của `gt_score` hỗ trợ cả list `dict` lẫn list `Step`.
- [ ] **17. CE từ G4:** khi trượt G4, mọi cặp vi phạm được lưu vào `CEPool` (đảo/hòa → `hack`, chênh nhỏ → `under_signal`); trượt (b) hoặc (c) mà không có cặp vi phạm thì chỉ sinh feedback.
- [ ] **18. Oracle dạng hiệu:** Oracle Meta-World dùng $R_{\text{expert}} - R_{\text{exploit}}^{\max} > 0.01\lvert R_{\text{expert}}\rvert$, không dùng phép nhân 99%; $P_0$ gồm đúng 4 cặp $\{(T_3,T_2),(T_2,T_1),(T_1,T_0),(T_3,T_0)\}$ + ≤ 20 cặp nhiễu.

---

## 6. CÁC KHÁC BIỆT ĐÃ ĐƯỢC CHỦ Ý

### 6.1. Khác biệt giữa spec này và bản notebook đã sinh số liệu trong báo cáo

Số liệu trong báo cáo `CEGR` được sinh từ notebook; nếu code mới tuân thủ spec này thì các điểm sau làm số liệu **không còn tái lập chính xác** và cần chạy lại hoặc ghi chú khi so sánh:

| Điểm                       | Báo cáo (notebook)                                                     | Spec này                                                                             |
| :------------------------- | :--------------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| Điều kiện dừng             | PASS, hết $N$, hoặc pool không thêm phần tử mới sau `STALL_K = 2` vòng | Chỉ PASS hoặc hết $N$                                                                |
| Sandbox                    | Cấm import                                                             | Cho phép `math`, `numpy`, `typing` qua `__import__` an toàn                          |
| Khóa dedup CE              | Chữ ký hành vi                                                         | Hash nội dung **hoặc** khóa $(\text{type}, \text{signature}, \text{progress\_pair})$ |
| G2                         | Chỉ nêu luật cho DoorKey                                               | Thêm luật riêng Meta-World                                                           |
| Probe liên tục             | Không có                                                               | SAC/PPO                                                                              |
| Reward cuối khi không PASS | Chưa được báo cáo mô tả                                                | Độ sâu cổng lớn nhất, hòa thì vòng muộn nhất (2.4E)                                  |
| CE khi trượt G4            | Chưa được báo cáo mô tả                                                | Mọi cặp vi phạm đều vào pool (2.4B)                                                  |
| Eureka dừng sớm            | Có thể có (giả thuyết giải thích 3,8 lần gọi; **chưa xác minh**)       | Luôn đủ 2 × 2 = 4 mẫu sinh, không dừng sớm                                           |

### 6.2. Khác biệt so với bài báo gốc (theo mô tả trong tài liệu dự án và hiểu biết của người soạn; cần đối chiếu lại với bài gốc trước khi trích dẫn)

- **Eureka:** bài gốc dùng nhiều mẫu hơn mỗi vòng, nhiều vòng hơn, nhiệt độ cao hơn, nhiều mốc checkpoint hơn và bắt đầu từ reward rỗng. Ở đây $K = 2$, $G = 2$, 5 mốc, có $R_0$ cho sẵn, RL tabular. Phải gọi là _Eureka-style (simplified)_.
- **CARD:** bài gốc có thêm Evaluator LLM, trajectory lấy từ quá trình huấn luyện và huấn luyện RL sau khi qua TPE. Ở đây là TPE tĩnh trên $P_0$ cố định, không huấn luyện RL. Phải gọi là _CARD-style (static TPE)_.

### 6.3. Thiên lệch cần ghi trong báo cáo

- Hack rate (DP) cuối cùng đo bằng cùng DP mà attacker G3 dùng, nên thiên lệch có lợi cho CEGR.
- Fitness của Eureka (probe 2500 × 3 trên DoorKey) gần giống điều kiện RL của Oracle, nên có lợi cho Eureka về VF (khác seed).
- Chỉ DoorKey 5x5 một layout; Meta-World chưa có số liệu; RQ4/RQ5 chưa kiểm chứng.

---

## 7. CÁC ĐIỂM CHƯA ĐƯỢC BÁO CÁO XÁC NHẬN (CẦN CHỐT TRONG CONFIG/CODE)

| Mã   | Nội dung đã chọn trong spec                                                                                                                                                         | Cần xác nhận                                                                                                                                                                                                                                |
| :--- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 7.1  | Cùng bảng nhiệt độ cho mọi phương pháp: sinh 0.95, sửa 0.35; Eureka Thế hệ 1 = sinh, Thế hệ 2 = sửa                                                                                 | Báo cáo chỉ nêu "sinh 0.95, sửa 0.35"; chưa nêu cách gán cho Eureka. Phiên bản spec trước từng ghi 0.7 cho Eureka                                                                                                                           |
| 7.2  | $R_0$ được chấm một lần để sinh feedback vòng 1, không tính vào $N$                                                                                                                 | Báo cáo không nêu rõ vòng 1 có feedback từ $R_0$ hay không                                                                                                                                                                                  |
| 7.3  | $\gamma_{\text{TPE}} = \gamma_{\text{DP}}$ lấy từ config                                                                                                                            | Báo cáo chỉ nêu $\gamma = 1.0$ cho probe Oracle/Eureka                                                                                                                                                                                      |
| 7.4  | $P_0$ = 4 cặp $\{(T_3,T_2),(T_2,T_1),(T_1,T_0),(T_3,T_0)\}$ + ≤ 20 cặp nhiễu; CARD dùng chung                                                                                       | Đây là quyết định thiết kế (3 cặp liền kề + 1 cặp neo) khớp con số "4 cặp" của báo cáo; báo cáo không nêu cặp cụ thể                                                                                                                        |
| 7.5  | Ngưỡng G5 liên tục lấy từ `metaworld_config.yaml`                                                                                                                                   | Bản spec đầu từng ghi 0.40, bản sau bỏ; Oracle 0.50                                                                                                                                                                                         |
| 7.6  | $R_{\text{final}}$ khi không PASS: ứng viên hợp lệ có độ sâu cổng lớn nhất, hòa thì vòng muộn nhất (CARD/CARD_REFL/CEGR); Eureka chọn trong mọi ứng viên đã sinh (hòa chọn sớm hơn) | Cả hai tài liệu trước đều chưa định nghĩa; ảnh hưởng trực tiếp hack rate và VF. Quy tắc "độ sâu" chỉ có ý nghĩa cho CEGR nên có thể bị phản biện là ưu ái CEGR; cân nhắc báo cáo thêm kết quả với quy tắc "vòng cuối" như phân tích độ nhạy |
| 7.7  | Khóa chữ ký bổ sung `type` và `progress_pair`; CE trùng bị bỏ, không đẩy lại FIFO                                                                                                   | Báo cáo chỉ nêu "khóa chữ ký"; chưa nêu xử lý khi trùng                                                                                                                                                                                     |
| 7.8  | `recurrence_count` của baseline được dựng post-hoc bằng bộ kiểm hậu kiểm riêng                                                                                                      | Báo cáo có cột này cho cả Eureka/CARD nhưng không mô tả cách đếm                                                                                                                                                                            |
| 7.9  | Số lần gọi LLM của Eureka phải đúng 4 mẫu sinh                                                                                                                                      | Báo cáo ghi trung bình 3,8. Một giả thuyết hợp lý là bản cũ dừng sớm khi một ứng viên Thế hệ 1 qua ngưỡng probe, nhưng **chưa được xác minh**; cần kiểm tra log của notebook                                                                |
| 7.10 | G4 xét mọi cặp có tiến độ khác nhau; CE trượt G4 đi vào pool cùng quỹ đạo (seed 2000)                                                                                               | Notebook có thể chỉ xét cặp mốc liền kề. CE từ G4 vào pool làm một phần quỹ đạo seed 2000 trở thành tập hồi quy của G3, nên G4 không còn hoàn toàn "độc lập" với G3 sau vòng đó; cần chấp nhận hoặc tách luồng seed                         |
