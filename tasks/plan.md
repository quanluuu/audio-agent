# Implementation Plan: Voice Agent

## Overview
Xây theo [SPEC.md](../SPEC.md): trợ lý giọng nói. Bạn nói (push-to-talk) → chuỗi lọc DSP (band-pass → spectral subtraction → noise gate) → STT (`faster-whisper` local mặc định, Gemini tùy chọn qua `--stt`) → DeepSeek Flash trong graph LangGraph + tools (DSP, ghi chú, mở app/web có xác nhận bằng `interrupt()`) → in câu trả lời ra màn hình (agent không nói). Có thêm script đánh giá SNR + STOI. Task chi tiết nằm trong [todo.md](todo.md). Phân công và quy trình Git nằm trong [CONTRIBUTING.md](../CONTRIBUTING.md): **A (Windows)** làm Task 0, 3, 5–6b, 13–17; **B (Ubuntu)** làm Task 7–12 (7–10 đã xong).

## Architecture Decisions
- **Task 0 trước hết.** `SPEC.md` và `CONTRIBUTING.md` đang bị `.gitignore` chặn, nên B chưa có hợp đồng `Pipeline`. Phải đưa lên repo trước khi làm tiếp.
- **Rủi ro giọng nói làm trước agent (Phase 2).** `faster-whisper` chưa chắc cài được trên Python 3.13/Windows. Task 6 kiểm tra ngay đầu task.
- **Lát dọc "nói → thấy chữ" (Task 5–6).** Thu câu nói + monitor + STT + in transcript chạy end-to-end trước khi có agent.
- **STT là hàm `transcribe(wav_bytes) -> str` truyền vào.** Whisper và Gemini chỉ khác hàm tạo, nên Task 6b chỉ thêm backend và cờ, không đụng vòng chính.
- **Agent xây dần theo graph:** tool thuần (13, 14) → graph `llm ↔ tools` (15) → node `confirm` + `interrupt()` (16) → vòng chính (17). Mỗi bước test được bằng model giả.
- **Spike DeepSeek + LangGraph ở đầu Task 15**, trước khi viết graph: chốt tên model, đường import, tool calling thật.
- **Xác nhận nằm ở node riêng, không nằm trong tool.** Resume `interrupt()` chạy lại node từ đầu. Tool nằm sau `interrupt()` sẽ chạy 2 lần.
- **`actions.py` không phụ thuộc LangGraph.** Test bảo mật (allowlist, URL scheme, path traversal) chạy độc lập.
- **`Pipeline` là điểm hội tụ duy nhất.** Realtime, đánh giá và agent đều dùng cùng một `Pipeline`.
- **Phụ thuộc ngoài truyền vào dưới dạng tham số** (model Whisper, client Gemini, model chat, out, ask_yes_no, Popen, webbrowser). Import thư viện nặng hoặc chỉ có trên Windows (`faster_whisper`) nằm trong hàm, để test xanh trên Ubuntu.
- **TDD mỗi task:** viết test đỏ → code tối thiểu cho xanh → dọn.

## Dependency Graph
```
helpers ── BandPass ── Pipeline/Params/Meters ──┬── audio.py: stream + ghi WAV phiên            (xong)
                                                ├── NoiseGate, SS, smoothing, minstat           (xong, B)
                                                ├── scripts/evaluate.py (11) ── đánh giá (12)   (B)
                                                ├── monitor + thu câu nói (5) ── Whisper STT (6) ── Gemini STT + --stt (6b) ──┐
                                                └── tool DSP (13) ── graph llm↔tools (15) ──┐                                  │
actions.py + describe_action (14) ──────────────────────────────────── node confirm (16) ──┴── handle_turn + vòng chính (17) ◄┘
```

Thứ tự của A: `0 → 5 → 6 → 6b → 14 → 13 → 15 → 16 → 17`. Task 14 không phụ thuộc gì, có thể làm sớm hơn nếu bị chặn ở Phase 2.

## Task List

### Phase 0: Đồng bộ spec
- [ ] Task 0: Đưa SPEC lên repo (bỏ chặn trong `.gitignore`, PR cho B duyệt)

### Phase 1: Nền tảng + đường realtime
- [x] Task 1: Khung dự án và hàm đo
- [x] Task 2: BandPass + Pipeline tối thiểu
- [ ] Task 3: Chạy realtime micro → tai nghe + ghi WAV — code + test xong, chờ nghe thử

### Checkpoint 1: Nghe được trực tiếp, 0 underrun trong 2 phút

### Phase 2: Giọng nói
- [-] Task 4: ~~TTS giọng Việt~~ — bỏ, agent chỉ in chữ (SPEC Assumption 6)
- [x] Task 5: Monitor bật/tắt + thu câu nói theo lượt (+ cờ `--all-filters`)
- [x] Task 6: Whisper STT + vòng "nói → thấy chữ"
- [x] Task 6b: Gemini STT + cờ `--stt`

### Checkpoint 2: Nói → thấy đúng chữ (Whisper ≤ 5 s/câu 5 s)

### Phase 3: Bộ lọc khử ồn (B, xong)
- [x] Task 7: NoiseGate
- [x] Task 8: SpectralSubtractor, chế độ calibrate
- [x] Task 9: Làm mượt gain
- [x] Task 10: Ước lượng ồn minstat

### Checkpoint 3: Cả chuỗi lọc chạy realtime < 2 ms/khối, STT vẫn đúng khi bật quạt

### Phase 4: Đánh giá (B)
- [ ] Task 11: `scripts/evaluate.py`
- [ ] Task 12: Thu dữ liệu, chạy đánh giá, ghi README

### Checkpoint 4: Tiêu chí 8, 9 của spec đạt (hoặc được bạn duyệt chỉnh)

### Phase 5: Agent (LangGraph + DeepSeek)
- [x] Task 14: Hành động ngoài (`actions.py` + `describe_action`)
- [x] Task 13: Tool DSP cho agent
- [x] Task 15: Graph `llm ↔ tools` + spike DeepSeek
- [x] Task 16: Node `confirm` + `interrupt()`
- [x] Task 17: `handle_turn` + vòng hội thoại đầy đủ

### Checkpoint 5: Toàn bộ Success Criteria 1–9 trong SPEC.md
Duyệt 2026-11-12; còn mở tiêu chí 8, 9 (Task 11–12 của B) và `/code-simplify`.

## Risks and Mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| `faster-whisper` cài lỗi trên Python 3.13/Windows | Med | Cài thử đầu Task 6, báo bạn trước khi đổi hướng. |
| Whisper `small` nhận dạng tiếng Việt kém hoặc chậm > 5 s | Med | Đo ở Task 6. Kém thì thử `medium`, so với Gemini ở Task 6b. |
| Đường import LangGraph / tên model DeepSeek khác hiểu biết hiện tại | Med | Đầu Task 15 đọc docs chính thức đúng bản cài, ghim phiên bản. |
| `ChatDeepSeek` + tool calling có lỗi riêng | Med | Spike gọi thật ở Task 15 trước khi viết graph. Đổi gói phải hỏi bạn. |
| Resume `interrupt()` làm tool chạy 2 lần | Med | `confirm` không tác dụng phụ; test 2 tool hành động trong một `AIMessage` (Task 16). |
| SS làm STT kém đi (paper ghi nhận SS làm tăng WER) | Med | Checkpoint 3 so sánh STT khi bật/tắt bypass, trên cả Whisper và Gemini. |
| Callback audio bị underrun do Python/GIL | Med | Meters đếm underrun. Callback không gọi mạng, không STT. Whisper chạy ở luồng chính. |
| Nhận dạng sai dẫn tới hành động sai | Med | Xác nhận y/n, danh sách cho phép, không shell, có test bảo mật. |
| Hằng số minstat / ngưỡng STOI không hợp với dữ liệu thật | Med | Task 12 có bước chỉnh, mọi thay đổi tiêu chí phải được bạn duyệt. |
| Tổng trễ một lượt vài giây | Low | Chấp nhận. In "đang nghe chữ…" và "đang nghĩ…". |

## Open Questions
- Open Questions 1–7 trong SPEC.md vẫn mở. Câu 1 (tên model DeepSeek) chốt ở Task 15. Câu 6 (danh sách app) cần trả lời trước Task 14. Câu 7 (model Gemini STT) chốt ở Task 6b.
- Task 7–10 của B có vào `main` qua PR được duyệt không? Log chỉ có một commit thẳng trên `main`. Từ Task 11 nên đi đúng quy trình PR trong CONTRIBUTING.md.
