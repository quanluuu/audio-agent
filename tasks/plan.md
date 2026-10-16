# Implementation Plan: Voice Agent

## Overview
Xây theo [SPEC.md](../SPEC.md): trợ lý giọng nói. Bạn nói (push-to-talk) → chuỗi lọc DSP (band-pass → spectral subtraction → noise gate) → Gemini STT → Gemini chat + tools (DSP, ghi chú, mở app/web có xác nhận) → Gemini TTS đọc trả lời (hết hạn mức thì chuyển sang pyttsx3). Có thêm script đánh giá SNR + STOI. Task chi tiết nằm trong [todo.md](todo.md). Phân công và quy trình Git nằm trong [CONTRIBUTING.md](../CONTRIBUTING.md): **A (Windows)** làm Task 3–6b, 13–15; **B (Ubuntu)** làm Task 7–12.

## Architecture Decisions
- **Rủi ro giọng nói làm trước bộ lọc (Phase 2).** TTS tiếng Việt (giọng OneCore, pyttsx3) và chất lượng Gemini STT tiếng Việt là 2 điều chưa chắc nhất. Nếu một trong hai không được, hướng đề tài phải đổi. Vì vậy cần biết sớm, trước khi đầu tư vào DSP.
- **Lát dọc "nói → thấy chữ" (Task 5–6).** Thu câu nói + monitor + STT + in transcript chạy end-to-end khi mới có band-pass. Mỗi bộ lọc thêm sau đó vừa nghe được bằng `--no-agent`, vừa thấy được ảnh hưởng lên STT.
- **Mỗi bộ lọc là một lát dọc:** test → class trong `dsp.py` → nối vào `Pipeline` → nghe thử.
- **`Pipeline` là điểm hội tụ duy nhất.** Realtime, đánh giá và agent đều dùng cùng một `Pipeline`.
- **Đánh giá trước agent.** Kiểm chứng DSP và chỉnh hằng số trước khi giao cho LLM điều khiển.
- **Phụ thuộc ngoài truyền vào dưới dạng hàm** (transcribe, chat, speak, confirm, Popen, webbrowser). Lõi logic test được bằng hàm giả. Gemini, giọng TTS và app thật chỉ kiểm tra thủ công.
- **Hành động ngoài tách riêng `actions.py`**, có test bảo mật riêng (danh sách cho phép, URL scheme, path traversal, xác nhận).
- **TDD mỗi task:** viết test đỏ → code tối thiểu cho xanh → dọn.

## Dependency Graph
```
helpers ── BandPass ── Pipeline/Params/Meters ──┬── audio.py: stream + ghi WAV phiên          (xong)
                                                ├── audio.py: monitor + thu câu nói ── voice.transcribe ──┐
                                                ├── NoiseGate                                             │
                                                ├── SpectralSubtractor ─┬─ gain smoothing                 │
                                                │                       └─ minstat                        │
                                                ├── scripts/evaluate.py (cần cả 3 bộ lọc)                 │
                                                └── agent: tool DSP ──┐                                   │
voice.speak (spike giọng Việt) ───────────────────────────────────────┤                                   │
actions.py (open_app, open_url, search_web, save_note + confirm) ─────┼── agent: chat + handle_turn ◄─────┘
                                                                      ┘
```

## Task List

### Phase 1: Nền tảng + đường realtime
- [x] Task 1: Khung dự án và hàm đo
- [x] Task 2: BandPass + Pipeline tối thiểu
- [ ] Task 3: Chạy realtime micro → tai nghe + ghi WAV — code + test xong, chờ nghe thử

### Checkpoint 1: Nghe được band-pass trực tiếp, 0 underrun trong 2 phút

### Phase 2: Spike rủi ro giọng nói
- [ ] Task 4: TTS dự phòng pyttsx3 (spike giọng Việt)
- [ ] Task 5: Monitor bật/tắt + thu câu nói theo lượt
- [ ] Task 6: Gemini STT + vòng "nói → thấy chữ"
- [ ] Task 6b: Gemini TTS + chuyển sang pyttsx3 khi hết hạn mức

### Checkpoint 2: Nói một câu → thấy đúng chữ; máy đọc được câu tiếng Việt

### Phase 3: Bộ lọc khử ồn
- [ ] Task 7: NoiseGate
- [ ] Task 8: SpectralSubtractor, chế độ calibrate
- [ ] Task 9: Làm mượt gain
- [ ] Task 10: Ước lượng ồn minstat

### Checkpoint 3: Cả chuỗi lọc chạy realtime < 2 ms/khối, STT vẫn đúng khi bật quạt

### Phase 4: Đánh giá
- [ ] Task 11: `scripts/evaluate.py`
- [ ] Task 12: Thu dữ liệu, chạy đánh giá, ghi README

### Checkpoint 4: Tiêu chí 7, 8 của spec đạt (hoặc được bạn duyệt chỉnh)

### Phase 5: Agent
- [ ] Task 13: Tool DSP cho agent
- [ ] Task 14: Hành động ngoài có xác nhận (`actions.py`)
- [ ] Task 15: Vòng hội thoại Gemini đầy đủ

### Checkpoint 5: Toàn bộ Success Criteria trong SPEC.md

## Risks and Mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| pyttsx3 không thấy giọng Việt OneCore | Med | Giờ chỉ là đường dự phòng. Task 4 làm đầu Phase 2; sửa registry phải hỏi bạn trước. Không được thì khi Gemini TTS hết hạn mức agent chỉ in chữ. |
| Phát audio Gemini TTS khi stream song công đang mở bị tranh thiết bị | Med | Kiểm tra thủ công ở Task 6b. Lỗi thì phát qua chính stream (hàng đợi phát trong callback). |
| Gemini STT tiếng Việt kém, hoặc API audio khác hiểu biết của tôi | High | Task 6 đọc docs chính thức `google-genai` trước khi code, rồi thử ngay với giọng thật. |
| SS làm STT kém đi (paper ghi nhận SS làm tăng WER) | Med | Checkpoint 3 so sánh STT khi bật/tắt bypass. Open Question 5 của spec đề xuất đo WER. |
| Thiết bị âm thanh Windows không mở được 16 kHz hoặc độ trễ cao | Med | Đã kiểm tra: MME mặc định mở được 16 kHz, 0.2 ms/khối. Vẫn chờ nghe thử duplex ở Checkpoint 1. |
| Callback audio bị underrun do Python/GIL | Med | Meters đếm underrun. Callback không gọi mạng, không TTS. |
| Nhận dạng sai dẫn tới hành động sai | Med | Task 14: xác nhận y/n, danh sách cho phép, không shell, có test bảo mật. |
| Hằng số minstat / ngưỡng STOI không hợp với dữ liệu thật | Med | Task 12 có bước chỉnh, mọi thay đổi tiêu chí phải được bạn duyệt. |
| Tổng trễ một lượt vài giây | Low | Chấp nhận. In "đang nghĩ…". |

## Open Questions
- Open Questions 1–6 trong SPEC.md vẫn mở. Câu 2 có số đo ở Checkpoint 1. Câu 6 (danh sách app) cần trả lời trước Task 14.
- Repo GitHub private do A tạo (CONTRIBUTING.md mục 2). Mỗi task một nhánh + PR + người kia duyệt.
