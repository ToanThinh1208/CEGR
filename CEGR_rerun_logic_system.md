# CEGR_RERUN_LOGIC_SYSTEM.md
# Giao thức chạy lại thực nghiệm CEGR (ICAI-FAI 2026)

> Cách dùng: đưa file này cho trợ lý code (Claude Code, Cursor...) làm chỉ dẫn hệ thống, hoặc dùng làm checklist cho chính bạn.
> Mục tiêu: một lần chạy duy nhất, sạch, sao cho mọi bảng trong bài đều sinh ra từ CÙNG một tập dữ liệu và CÙNG một đoạn mã, để loại bỏ mâu thuẫn Bảng III / Bảng IV.

---

## 0. NGUYÊN TẮC BẤT BIẾN (agent phải tuân thủ)

1. **Một nguồn sự thật.** Mọi con số trong bài được sinh bởi script phân tích từ file log thô. Không chép tay số từ nơi này sang nơi khác.
2. **Cùng chạy, cùng harness.** CEGR và các ablation trên tập ablation phải chạy trong cùng một đợt, cùng phiên bản mã, cùng mô hình, cùng cấu hình. Ablation chỉ khác CEGR ở các cờ đã khai báo.
3. **Tập ablation là tập con khai báo trước của tập chính.** Không được chọn tay hay chọn sau khi đã thấy kết quả.
4. **Không sửa mã hay tham số sau khi thấy kết quả.** Nếu phải sửa (lỗi), ghi lại và chạy lại toàn bộ đợt bị ảnh hưởng.
5. **Không bịa số liệu.** Nếu thiếu dữ liệu, xuất cảnh báo và để trống, không điền ước lượng.
6. **Mọi tỷ lệ phải kèm mẫu số tường minh** (số reward, số mẫu hợp lệ).

---

## 1. THIẾT KẾ THỰC NGHIỆM

### 1.1 Tập dữ liệu

| Ký hiệu | Mô tả |
|---|---|
| `MAIN` | 244 base reward (giữ nguyên tập đã dùng) |
| `ABL` | Tập con của `MAIN`, chọn ngẫu nhiên có phân tầng bằng seed cố định (xem 1.2) |

- Tập `MAIN` và `ABL` dùng chung một danh sách `base_id`. Lưu `selection_manifest.json` gồm: seed chọn, thuật toán, danh sách `base_id`, thời điểm tạo, hash file.
- **Kích thước `ABL` theo ngân sách (chọn một mức, ghi rõ trong bài):**
  - Mức T1 (tối thiểu): giữ 12 reward cũ nhưng **chạy lại CEGR trong cùng đợt** và lặp 3 replicate.
  - Mức T2 (khuyến nghị): n = 24 reward.
  - Mức T3 (lý tưởng): n = 36 reward trở lên.
- Nếu 12 reward cũ nằm trong `MAIN`, đưa chúng vào `ABL` (để đối chiếu với kết quả cũ) rồi bổ sung ngẫu nhiên cho đủ n.

### 1.2 Quy tắc chọn `ABL`

```
seed_select = 20261031            # cố định, ghi vào manifest
bước 1: nếu có thể, phân tầng theo outcome của lần chạy MAIN (đã VF / chưa VF) và theo môi trường
bước 2: lấy mẫu ngẫu nhiên không hoàn lại, kích thước n, bằng seed_select
bước 3: ghi manifest, KHÔNG chọn lại
```
Mục đích: tránh lệch chọn (trước đây CEGR trên 12 reward đạt 33,3 / 55,6 / 11,1 so với 19,3 / 22,3 / 5,6 trên 244 reward).
Nếu bạn phân tầng theo outcome của lần chạy MAIN, phải nói rõ trong bài rằng tầng này làm thay đổi tỷ lệ cơ sở và không dùng ABL để ước lượng tỷ lệ tuyệt đối.

### 1.3 Các điều kiện (arm)

| Arm | Mô tả | Mục đích |
|---|---|---|
| `CEGR` | Pipeline đầy đủ | Baseline ghép cặp |
| `noPool` | Tắt CE pool (prompt và G3 replay) | Ablation |
| `noAnchors` | Tắt cập nhật anchor động, giữ 6 anchor đầu | Ablation |
| `noClass` | Gộp mọi nhãn vào `generic` → `_flat` | Ablation (đổi tên trong bài: "luật phản hồi theo nhãn vs luật phẳng") |
| `noG2` | Bỏ G2 (AST) | Ablation |
| `noG4` | Bỏ G4 | Ablation |
| `noG5` | Bỏ G5 | Ablation |
| `noFeedback` (MỚI) | Giữ G1–G5 làm bộ lọc, nhưng KHÔNG đưa CE, rule hay anchor mới vào prompt; mỗi vòng chỉ lấy mẫu lại cùng prompt gốc | Baseline "chỉ lấy mẫu lại cùng ngân sách": trả lời câu hỏi việc sửa có tác dụng không |
| `round0` (MỚI, không tốn compute) | Chỉ đánh giá VF của reward gốc / ứng viên ở vòng đầu, trước mọi phản hồi | Mốc "không sửa" |

> `noFeedback` cần một cờ mới (thay đổi mã nhỏ). Nếu không thể, bỏ arm này và nêu rõ đó là giới hạn trong bài. `round0` chỉ cần trích từ log, không cần mã mới.

### 1.4 Replicate và noise floor

- Mỗi (arm × môi trường × base_id) có 3 mẫu như cũ (`sample_id` ∈ {0,1,2}).
- **Replicate:** chạy `CEGR` ít nhất **R = 3 lần** độc lập (seed LLM khác nhau, `rep` ∈ {0,1,2}) trên toàn bộ `ABL`. Các ablation chạy ít nhất R = 1 (khuyến nghị R = 2 nếu đủ compute).
- Chênh lệch giữa các replicate của CEGR là **noise floor**. Mọi ΔVF của ablation phải được đọc so với noise floor này.
- Seed LLM: `llm_seed = hash(rep, base_id, sample_id)`; **dùng chung giữa các arm** trong cùng `rep` (common random numbers) để giảm phương sai ghép cặp. Ghi rõ rằng vLLM không đảm bảo tất định hoàn toàn khi gộp batch.
- Giữ nguyên các luồng seed đã dùng: CE generation = 1000, reward-signal = 2000, final VF = 3000.

### 1.5 Ước lượng ngân sách (điền số thật trước khi chạy)

```
Số job = (số arm) × (số môi trường = 3) × n_reward × 3 mẫu × R
Ví dụ T2: 9 arm × 3 × 24 × 3 × (R như trên) ≈ 1944 job nếu R = 1; nhân thêm cho replicate của CEGR
Mỗi job: tối đa 4 vòng × 3 ứng viên
```
Hãy đo thời gian trên 10 job thử, nhân ra tổng thời gian, rồi quyết định mức T1/T2/T3.
**Thứ tự ưu tiên nếu thiếu compute:** (1) CEGR ghép cặp lặp R lần trên ABL; (2) `noClass`; (3) `noFeedback`; (4) `noPool`, `noAnchors`; (5) `noG2`, `noG4`, `noG5`.

---

## 2. GHI LOG BẮT BUỘC (JSONL, mỗi ứng viên một dòng)

```json
{
  "run_id": "...", "code_commit": "...", "model": "Qwen2.5-Coder-7B-Instruct-AWQ",
  "env": "5x5|6x6|8x8", "arm": "CEGR", "rep": 0,
  "base_id": "...", "sample_id": 0, "round": 1, "round_index_base": 1,
  "cand_idx": 0, "llm_seed": 0, "temperature": 0.35,
  "gate_results": {"G1": "pass", "G2": "fail:static_flag_3", "G3": null, "G4": null, "G5": null},
  "first_failed_gate": "G2",
  "ce_label": "hack|under_signal|generic|none", "signature_key": "toggle",
  "pool_size_after": 3, "rules_issued": ["toggle"], "anchor_version": "v1",
  "g5_episodes_used": 0, "llm_tokens": 0,
  "passed_search": false
}
```

Và **một dòng kết quả cuối mỗi job** (`final.jsonl`):

```json
{
  "env": "...", "arm": "...", "rep": 0, "base_id": "...", "sample_id": 0,
  "valid": true, "invalid_reason": null,
  "returned_candidate_round": 4, "stop_reason": "pass|budget|no_new_ce",
  "n_llm_calls": 0, "n_candidates": 0, "k_g5_episodes_search": 0,
  "final_dp_T_ok": true, "final_dp_2T_ok": true,
  "final_subgoal_ok": true, "final_q_success": 0.0, "final_vf": 0,
  "dp_hack_flag": false
}
```

### Định nghĩa phải cố định và ghi vào bài

- `round_index_base`: vòng đầu tiên là 0 hay 1; ngân sách 4 vòng có gồm vòng 0 không. Thống nhất với ví dụ prompt ("ROUND 2/5") và sentinel.
- `rounds_to_pass`: **chỉ tính trên các job pass trong lúc tìm kiếm; không dùng sentinel khi báo cáo trung bình.** Báo cáo riêng: `P(pass tại vòng ≤ k)` (hàm tích lũy) và tỷ lệ job không pass.
- `final_vf` đánh giá trên ứng viên **trả về cuối cùng**, kể cả khi job không pass trong tìm kiếm. Bài cần nói rõ VF khác với "qua hết gate trong lúc tìm kiếm" (bảng cũ có nhiều ô Rounds = 5,00 nhưng VF > 0).
- `valid`: định nghĩa chính xác thế nào là mẫu hợp lệ. Mẫu không hợp lệ KHÔNG được loại âm thầm: báo cáo số mẫu hợp lệ trong mỗi ô (n_valid / n_total).
- `dp_hack_flag`: định nghĩa và mẫu số của "DP hack rate" (chia cho số job hay số ứng viên; có loại trừ với VF hay không).
- `n_llm_calls`: nêu rõ một lần gọi sinh 1 hay 3 ứng viên (n=3).
- `k_g5_episodes_search`: chỉ gồm probe G5 trong lúc tìm kiếm, tách khỏi chi phí đánh giá VF cuối (2.500 episode × 3 seed).

---

## 3. KIỂM TRA TỰ ĐỘNG TRƯỚC KHI PHÂN TÍCH (sanity checks, dừng nếu thất bại)

```
[ ] Mỗi (env, arm, rep, base_id) có đủ 3 sample_id; liệt kê mọi ô thiếu.
[ ] VF mỗi ô (aggregate theo mẫu) là bội của 1/(3 × n_valid_reward) khi mọi mẫu hợp lệ.
[ ] CEGR trên ABL tính từ lần chạy này; không dùng lại số cũ.
[ ] Với mọi arm: VF_ablation + ΔVF == VF_CEGR (ghép cặp), sai số làm tròn ≤ 0,1 pp.
[ ] Mẫu số của MAIN: kiểm tra n_job = 244 × 3 = 732 (hoặc số thực), VF% = n_vf / n_job.
[ ] Không job nào có final_vf = 1 mà final_dp_T_ok hoặc final_dp_2T_ok hoặc final_subgoal_ok = false.
[ ] final_q_success ≥ 0.5 với mọi job có final_vf = 1.
[ ] Với arm noG5: k_g5_episodes_search == 0. Với noG2: không có first_failed_gate == "G2".
[ ] Với noClass: ce_label ∈ {generic, none} và rules_issued chỉ chứa _flat.
[ ] Với noPool: pool_size_after == 0 xuyên suốt.
[ ] Tỷ lệ "đến được G5" > 0 cho CEGR (nếu ≈ 0, ghi chú vì noG5 gần như không can thiệp).
[ ] Đối chiếu: số kết quả CEGR của ABL trong lần chạy mới so với kết quả cũ (33,3 / 55,6 / 11,1) — ghi lại chênh lệch.
```

---

## 4. PHÂN TÍCH THỐNG KÊ (script `analyze.py`)

### 4.1 Đơn vị và quy tắc

- Đơn vị phân tích = **base reward**. VF của mỗi reward = trung bình `final_vf` trên các mẫu hợp lệ (và, nếu có R replicate, trung bình trên replicate).
- ΔVF = VF_CEGR − VF_ablation (pp, dương = có lợi cho CEGR), tính cặp theo `base_id`.
- Kiểm định hoán vị dấu **chính xác** khi n ≤ 20 (liệt kê 2^n tổ hợp); nếu n > 20 dùng ≥ 200.000 hoán vị Monte Carlo và ghi rõ.
- CI: bootstrap phần trăm trên base reward, 10.000 lần. Nếu hầu hết chênh lệch bằng 0, báo cáo rõ CI có thể suy biến.
- Hiệu chỉnh Holm.

### 4.2 Khai báo trước giả thuyết (pre-registration, ghi vào file này TRƯỚC khi chạy)

```
Họ giả thuyết chính (primary), Holm trên 3 so sánh:
  H1: CEGR vs noClass trong từng môi trường (5x5, 6x6, 8x8)
Họ thăm dò (exploratory), Holm riêng trên 18 so sánh như bài cũ; báo cáo p chưa hiệu chỉnh chỉ để minh bạch:
  noPool, noAnchors, noClass, noG2, noG4, noG5 × 3 môi trường
Bổ sung (không kiểm định): so sánh CEGR vs noFeedback và CEGR vs round0 (báo cáo hiệu ứng và CI)
```
Không thêm hay bớt giả thuyết sau khi thấy dữ liệu.

### 4.3 Mã tham khảo

```python
import numpy as np, pandas as pd

def reward_level(df, env, arm):
    d = df[(df.env == env) & (df.arm == arm) & (df.valid)]
    return d.groupby("base_id").final_vf.mean()           # trung bình mẫu rồi replicate

def paired_diff(df, env, arm_a="CEGR", arm_b="noClass"):
    a, b = reward_level(df, env, arm_a), reward_level(df, env, arm_b)
    idx = a.index.intersection(b.index)                   # chỉ các reward có ở cả hai arm
    return (a[idx] - b[idx]).to_numpy() * 100, list(idx)  # pp

def perm_p(d, n_mc=200_000, seed=0):
    d = np.asarray(d, float); n = len(d); obs = abs(d.mean())
    if n <= 20:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        stats = np.abs((signs * d).mean(axis=1))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1, 1], size=(n_mc, n))
        stats = np.abs((signs * d).mean(axis=1))
    return float((stats >= obs - 1e-12).mean())

def boot_ci(d, B=10_000, seed=0):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(B, len(d)), replace=True).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])

def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj
```

### 4.4 Noise floor

- Với R ≥ 2 replicate của CEGR: tính chênh lệch từng cặp replicate trên cùng reward (rep_i − rep_j), báo cáo trung bình và khoảng. Đây là mức nhiễu giữa hai lần chạy cùng cấu hình.
- Khi viết bài: chỉ coi một ΔVF là đáng chú ý nếu độ lớn vượt rõ noise floor, và vẫn phải nêu p sau Holm.

---

## 5. BẢNG VÀ HÌNH CẦN SINH TỰ ĐỘNG

1. **Bảng III (MAIN):** VF%, CI (nêu rõ bootstrap theo cụm reward hay không), DP hack rate (+ định nghĩa), final RL success, P(pass tại vòng ≤ k), LLM calls, K (chỉ G5 tìm kiếm). Ghi `n_job` và `n_reward`.
2. **Bảng IV (ABL, ghép cặp):** mỗi arm × env gồm `n_reward`, `n_valid_samples`, VF%, ΔVF so với CEGR **cùng đợt**, CI, p chưa hiệu chỉnh, p Holm (họ chính / họ thăm dò), Rounds (chỉ trên job pass) và K. Dòng CEGR là số của chính đợt này. Chú thích rõ khác với Bảng III.
3. **Bảng gate:** tỷ lệ ứng viên bị loại bởi từng gate và tỷ lệ đến được G5 (theo arm và env).
4. **Bảng nền tảng:** CEGR vs `noFeedback` vs `round0`.
5. **Bảng noise floor:** chênh lệch giữa các replicate của CEGR.
6. **Hình:** ΔVF với CI (như Hình 2), vẽ thêm vùng ±noise floor.
7. **Bảng đối chiếu Bảng III và Bảng IV:** VF của CEGR trên `ABL` lấy từ lần chạy MAIN (lọc theo base_id) so với lần chạy ghép cặp mới. Đây là bảng trả lời câu hỏi "vì sao 22,3% lại thành 55,6%".

---

## 6. QUY TẮC DIỄN GIẢI KẾT QUẢ (quyết định trước)

| Kết quả | Cách viết trong bài |
|---|---|
| Không có p Holm < 0,05 | "Không so sánh nào đạt ý nghĩa sau hiệu chỉnh; kết quả thăm dò." Không viết "không có hiệu ứng". |
| ΔVF của noClass lớn và vượt noise floor, p Holm họ chính < 0,05 | "Phản hồi theo nhãn có liên quan đến VF cao hơn trong pipeline này." Không viết "phân loại gây ra". |
| ΔVF của gate nhỏ hơn noise floor | "Chênh lệch nằm trong mức biến thiên giữa các lần chạy." |
| noG4/noG5 có tỷ lệ đến G5 ≈ 0 | "Các gate này hiếm khi được kích hoạt nên ablation gần như không can thiệp." |
| CEGR ≈ noFeedback | "Không có bằng chứng rằng phản hồi CE vượt việc lấy mẫu lại với cùng ngân sách." Hạ tuyên bố đóng góp tương ứng. |
| CEGR > noFeedback rõ | Đưa lên như kết quả chính, kèm CI. |
| VF trên ABL trong lần chạy mới khác lần chạy MAIN ở cùng reward | Nêu rõ, báo cáo như ước lượng biến thiên giữa các lần chạy. |
| 8×8 có ≤ 3 thành công trên toàn arm | Ghi "hiệu ứng sàn, không diễn giải". |

---

## 7. LỊCH LÀM VIỆC GỢI Ý (hạn nộp 31/10/2026)

| Giai đoạn | Việc | Cần compute |
|---|---|---|
| G0 (vài giờ) | Trích log cũ: lọc CEGR của lần chạy MAIN theo 12 base_id cũ; kiểm tra các sanity check; điền bảng đối chiếu (mục 5.7) | Không |
| G1 | Chạy CEGR × 3 replicate trên 12 reward cũ, cùng đợt với 6 ablation (T1); thêm `round0` | Trung bình |
| G2 | Mở rộng lên n = 24 (T2) và thêm `noFeedback` | Cao |
| G3 | Chạy `analyze.py`, sinh bảng/hình, viết lại các mục theo mục 6 | Không |

Nếu không kịp G1 trước hạn nộp: nộp bản sửa trình bày (Bảng IV đúng, mẫu số rõ, mô tả cách chọn 12 reward, thừa nhận giới hạn) và nêu việc chạy lại như hướng tiếp theo. Chưa xác minh được liệu ICAI-FAI cho phép cập nhật nội dung khoa học khi camera-ready (20/11); cần hỏi ban tổ chức nếu muốn dựa vào điều đó.

---

## 8. ĐIỀU AGENT KHÔNG ĐƯỢC LÀM

- Không làm tròn hay "làm đẹp" số liệu cho khớp văn bản.
- Không loại mẫu không hợp lệ mà không ghi lại.
- Không coi 3 mẫu của một reward là 3 quan sát độc lập.
- Không chọn lại `ABL` sau khi thấy kết quả.
- Không dùng Rounds-to-pass có sentinel làm bằng chứng hiệu quả.
- Không kết luận quan hệ nhân quả từ các ablation đan xen.
