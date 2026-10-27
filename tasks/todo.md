# Todo: Voice Agent

Mỗi task theo TDD: viết test đỏ → code tối thiểu cho xanh → dọn code. Lệnh test chung: `python -m pytest -q`.

## Phase 1: Nền tảng + đường realtime

### Task 1: Khung dự án và hàm đo
**Mô tả:** Tạo package `voiceagent`, `requirements.txt`, `.gitignore`, cài dependency. Viết 4 hàm hỗ trợ trong `dsp.py`.

**Acceptance:**
- [x] `pip install -r requirements.txt` thành công trên Python 3.13 (numpy, scipy, sounddevice, google-genai, pystoi, pytest).
- [x] `rms_db`, `snr_db`, `add_noise(x, snr_db, seed)`, `mix_at_snr(clean, noise, snr_db)` đạt SNR mục tiêu trong 0.1 dB. `mix_at_snr` cắt hoặc lặp ồn cho bằng độ dài giọng.
- [x] `.gitignore` loại `out/`, `data/`, `.env`.

**Verify:** `python -m pytest -q tests/test_dsp.py`

**Dependencies:** None
**Files:** `requirements.txt`, `.gitignore`, `voiceagent/__init__.py`, `voiceagent/dsp.py`, `tests/test_dsp.py`
**Scope:** S

### Task 2: BandPass + Pipeline tối thiểu
**Mô tả:** Class `BandPass` có state `zi`. `Params` (bản chụp bất biến), setter ở `Pipeline` kiểm tra miền, `Meters`, `Pipeline.process_block` chỉ có band-pass và `bypass`.

**Acceptance:**
- [x] BandPass: 1000 Hz giữ trong 1 dB; 50 Hz và 6000 Hz giảm ≥ 20 dB. Xử lý theo khối 256 mẫu giống xử lý cả tín hiệu (sai số < 1e-6). Dải sai ném `ValueError`.
- [x] Pipeline: `bypass` trả đúng đầu vào; đổi Params có hiệu lực từ khối kế tiếp; setter ngoài miền ném `ValueError`, Params giữ nguyên.
- [x] Meters cập nhật RMS vào/ra và thời gian xử lý; một khối trung bình < 2 ms.

**Verify:** `python -m pytest -q tests/test_dsp.py tests/test_pipeline.py`

**Dependencies:** Task 1
**Files:** `voiceagent/dsp.py`, `voiceagent/pipeline.py`, `tests/test_dsp.py`, `tests/test_pipeline.py`
**Scope:** M

### Task 3: Chạy realtime micro → tai nghe + ghi WAV
**Mô tả:** `audio.py` mở `sounddevice.Stream` song công 16 kHz, blocksize 256, gọi `Pipeline.process_block` trong callback, đếm underrun/overflow. `__main__.py` có `--list-devices`, `--input`, `--output`, `--no-agent`. Thoát bằng `quit`/Ctrl+C sẽ ghi `out/<thời gian>_raw.wav` và `_clean.wav`.

**Acceptance:**
- [x] Hàm ghi WAV (tách khỏi phần cứng) có unit test với `tmp_path`: ghi rồi đọc lại đúng độ dài và sample rate.
- [x] `python -m voiceagent --list-devices` in danh sách thiết bị.
- [x] Số thiết bị sai: in lỗi rõ ràng, thoát không traceback.

**Verify:**
- `python -m pytest -q`
- Thủ công (CHỜ BẠN): đeo tai nghe, chạy `python -m voiceagent --no-agent` 2 phút, nghe thấy giọng mình (đã band-pass). Meters báo 0 underrun. Ctrl+C sinh 2 file WAV.

**Dependencies:** Task 2
**Files:** `voiceagent/audio.py`, `voiceagent/__main__.py`, `tests/test_audio.py`
**Scope:** M

## Checkpoint 1: Sau Task 1–3
- [x] Toàn bộ test xanh (30/30)
- [ ] Nghe trực tiếp ổn định 2 phút, 0 underrun
- [ ] Ghi lại độ trễ cảm nhận được, trả lời Open Question 2 của spec
- [ ] Bạn duyệt trước khi sang Phase 2

## Phase 2: Spike rủi ro giọng nói

### Task 4: TTS dự phòng pyttsx3 (spike giọng Việt + `voice.speak` cục bộ)
**Mô tả:** Đường TTS offline, dùng khi Gemini TTS (Task 6b) hết hạn mức hoặc lỗi. Bạn cài language pack tiếng Việt (Settings → Time & Language → Speech → thêm giọng). Kiểm tra pyttsx3 có liệt kê giọng `vi-VN` không. Nếu không thấy, đề xuất cách vòng qua registry và **chờ bạn đồng ý** trước khi sửa. Sau đó viết `voice.py`: `pick_voice(voices)` chọn giọng `vi-VN`, `make_speaker()` trả hàm `speak(text)`. Không có giọng Việt thì cảnh báo một lần, `speak` chỉ in chữ.

**Acceptance:**
- [ ] `pick_voice` chọn đúng giọng `vi-VN` từ danh sách giọng giả; không có thì trả `None`.
- [ ] Không có giọng Việt: `speak` không gọi engine, chỉ in chữ.
- [ ] Thủ công: máy đọc được "Xin chào, mình là trợ lý giọng nói" bằng giọng Việt. Nếu không được: báo bạn. Dự án vẫn chạy được vì Gemini TTS là đường chính, nhưng khi hết hạn mức thì agent chỉ in chữ.

**Verify:** `python -m pytest -q tests/test_voice.py`; `python -c "from voiceagent.voice import make_speaker; make_speaker()('Xin chào')"`

**Dependencies:** None
**Files:** `requirements.txt` (thêm `pyttsx3`), `voiceagent/voice.py`, `tests/test_voice.py`
**Scope:** S

### Task 5: Monitor bật/tắt + thu câu nói theo lượt
**Mô tả:** Thêm `monitor` vào `Params` + `Pipeline.set_monitor`. Callback: monitor tắt thì ghi khối 0 ra tai nghe. Thêm bộ thu câu nói: `start_utterance()` / `stop_utterance() -> bytes | None`. Khi đang giữ lượt, callback chép khối đã khử ồn vào bộ đệm. Lúc dừng thì đóng gói thành WAV 16 kHz mono trong RAM. Câu < 0.3 s trả `None`. `--no-agent` bật monitor sẵn; chế độ thường thì tắt, trừ khi có `--monitor`.

**Acceptance:**
- [ ] Monitor tắt: tai nghe nhận khối 0. Monitor bật: nhận khối đã xử lý.
- [ ] Chỉ khối nằm giữa start và stop vào bộ đệm, và đó là khối đã khử ồn.
- [ ] WAV trong RAM đọc lại đúng 16 kHz, mono, đúng độ dài; câu < 0.3 s trả `None`.

**Verify:** `python -m pytest -q`

**Dependencies:** Task 3
**Files:** `voiceagent/pipeline.py`, `voiceagent/audio.py`, `voiceagent/__main__.py`, `tests/test_pipeline.py`, `tests/test_audio.py`
**Scope:** M

### Task 6: Gemini STT + vòng "nói → thấy chữ"
**Mô tả:** Đầu task, đọc docs chính thức `google-genai` về gửi audio inline. Viết `voice.transcribe(wav_bytes, client) -> str` và ghi transcript (`append_transcript(path, speaker, text)`). Sửa `__main__`: Enter trên dòng trống để bắt đầu/dừng lượt, in "🎙 đang nghe…", gửi STT, in `Bạn: …` và ghi transcript. Gõ chữ thì ghi `Bạn (gõ): …`. Chưa có chat. Thiếu `GEMINI_API_KEY`: báo rõ, vẫn chạy được phần DSP.

**Acceptance:**
- [ ] Transcript ghi đúng định dạng (test với file tạm).
- [ ] Thiếu key hoặc lỗi mạng: thông báo rõ, không traceback, stream không dừng.
- [ ] Thủ công: nói 5 câu tiếng Việt, chữ in ra đúng ý ít nhất 4/5 câu.

**Verify:** `python -m pytest -q`; thủ công: `python -m voiceagent`, nói 5 câu, mở `out/*_transcript.md`.

**Dependencies:** Task 5
**Files:** `voiceagent/voice.py`, `voiceagent/__main__.py`, `tests/test_voice.py`
**Scope:** M

### Task 6b: Gemini TTS + chuyển sang pyttsx3 khi hết hạn mức
**Mô tả:** Đầu task, đọc docs chính thức `google-genai` về TTS: tên model, cách chọn giọng, định dạng audio trả về. Viết `gemini_tts(text, client) -> (audio, sr)` và phát bằng `sounddevice`. Sửa `make_speaker` thành chuỗi Gemini TTS → pyttsx3 → in chữ, theo quy tắc trong SPEC mục voice. Hàm Gemini, pyttsx3 và play được truyền vào để test bằng hàm giả. Nhận diện "hết hạn mức" qua mã lỗi 429 / `RESOURCE_EXHAUSTED` của `google-genai`.

**Acceptance:**
- [ ] Gemini thành công: phát audio, không gọi pyttsx3.
- [ ] Lỗi hết hạn mức: câu đó đọc bằng pyttsx3; các câu sau không gọi Gemini nữa; thông báo chỉ in một lần.
- [ ] Lỗi khác: câu đó đọc bằng pyttsx3, câu sau vẫn thử Gemini. Không có key: không bao giờ gọi Gemini.
- [ ] Thủ công: câu tiếng Việt đọc bằng giọng Gemini trong lúc stream đang mở, không lỗi thiết bị. Đặt `GEMINI_TTS_MODEL` thành tên sai để giả lập lỗi, kiểm tra agent đọc bằng pyttsx3.

**Verify:** `python -m pytest -q tests/test_voice.py`; thủ công như trên.

**Dependencies:** Task 4, 6
**Files:** `voiceagent/voice.py`, `tests/test_voice.py`
**Scope:** S

## Checkpoint 2: Sau Task 4–6b
- [ ] Toàn bộ test xanh
- [ ] Máy đọc được tiếng Việt (Gemini TTS, và pyttsx3 khi dự phòng); nói → thấy đúng chữ
- [ ] Ghi chú chất lượng STT (đúng bao nhiêu câu, lỗi kiểu gì) để so sánh sau khi có bộ lọc
- [ ] Bạn duyệt trước khi sang Phase 3

## Phase 3: Bộ lọc khử ồn

### Task 7: NoiseGate
**Mô tả:** Class `NoiseGate`: RMS khối so ngưỡng, gain làm mượt (attack nhanh, release chậm). Nối vào Pipeline sau band-pass, thêm `set_noise_gate(enabled, threshold_db)`.

**Acceptance:**
- [x] Khối nhiễu -60 dBFS giảm ≥ 15 dB sau khi gain ổn định; khối sine -20 dBFS giữ trong 1 dB.
- [x] Bật/tắt có hiệu lực ở khối kế tiếp; ngưỡng ngoài miền ném `ValueError`.

**Verify:** `python -m pytest -q`; thủ công: `--no-agent`, im lặng thì nền gần như tắt.

**Dependencies:** Task 2
**Files:** `voiceagent/dsp.py`, `voiceagent/pipeline.py`, `tests/test_dsp.py`, `tests/test_pipeline.py`
**Scope:** S

### Task 8: SpectralSubtractor, chế độ calibrate
**Mô tả:** FFT 512, hop 256, sqrt-Hann, overlap-add. Gain `max(1 - alpha·|N|/|X|, beta)`. Phổ ồn lấy từ hiệu chuẩn. Pipeline có trạng thái hiệu chuẩn, tự hiệu chuẩn 1 s lúc khởi động. Đặt SS giữa band-pass và gate.

**Acceptance:**
- [x] `alpha=0`: đầu ra bằng đầu vào trễ 256 mẫu.
- [x] Sine 440 Hz + nhiễu trắng 5 dB, hiệu chuẩn 1 s nhiễu thuần: SNR tăng ≥ 5 dB.
- [x] Pipeline: trong lúc hiệu chuẩn không trừ ồn; xong thì Meters có mức ồn nền.

**Verify:** `python -m pytest -q`; thủ công: bật quạt, `--no-agent`, im lặng 1 s đầu, nghe ồn quạt giảm.

**Dependencies:** Task 2
**Files:** `voiceagent/dsp.py`, `voiceagent/pipeline.py`, `tests/test_dsp.py`, `tests/test_pipeline.py`
**Scope:** M

### Task 9: Làm mượt gain
**Mô tả:** Thêm `smoothing` vào SpectralSubtractor và setter: `G_s = smoothing·G_s_prev + (1 - smoothing)·G`.

**Acceptance:**
- [x] Trên nhiễu trắng thuần, độ lệch chuẩn theo thời gian của gain với `smoothing=0.5` thấp hơn `smoothing=0`.
- [x] `smoothing=0` cho kết quả giống hệt Task 8 (test cũ vẫn xanh).

**Verify:** `python -m pytest -q`

**Dependencies:** Task 8
**Files:** `voiceagent/dsp.py`, `voiceagent/pipeline.py`, `tests/test_dsp.py`
**Scope:** XS

### Task 10: Ước lượng ồn minstat
**Mô tả:** Thêm `noise_mode` (`calibrate` | `minstat`). Minstat: `P = 0.85·P + 0.15·|X|²`, ring buffer 96 khung, `|N| = sqrt(1.5·min)`. Chế độ minstat bỏ qua hiệu chuẩn.

**Acceptance:**
- [x] Nhiễu tăng +6 dB ở giây thứ 3: trong 2 s sau, ước lượng bám mức mới, sai số ≤ 3 dB.
- [x] Sine 1000 Hz bật/tắt mỗi 0.5 s: ước lượng ở bin 1000 Hz nằm trong 3 dB của nhiễu thật.
- [x] Sine bật liên tục 3 s: ước lượng ở bin đó tăng lên (ghi lại giới hạn đã biết).

**Verify:** `python -m pytest -q`; thủ công: `--no-agent` với minstat, nói ngay từ đầu, ồn vẫn giảm sau khoảng 1.5 s.

**Dependencies:** Task 8
**Files:** `voiceagent/dsp.py`, `voiceagent/pipeline.py`, `tests/test_dsp.py`, `tests/test_pipeline.py`
**Scope:** S

## Checkpoint 3: Sau Task 7–10
- [ ] Toàn bộ test xanh
- [ ] Cả chuỗi chạy realtime 2 phút, 0 underrun, < 2 ms/khối
- [ ] Bật quạt, nói 5 câu ở Task 6 khi bật lọc và khi bypass: so sánh STT, ghi kết quả
- [ ] Bạn duyệt trước khi sang Phase 4

## Phase 4: Đánh giá

### Task 11: scripts/evaluate.py
**Mô tả:** Trộn giọng + ồn ở từng SNR, chạy Pipeline theo khối 256, bù trễ 256 mẫu, tính SNR + STOI trước/sau cho 5 cấu hình trong spec. CLI đọc WAV, kiểm tra mono 16 kHz (sai thì báo lỗi, không resample), in bảng.

**Acceptance:**
- [ ] Cấu hình "không xử lý": STOI sau = STOI trước (sai số < 0.01); bỏ qua khi thiếu pystoi.
- [ ] File sai định dạng: lỗi rõ ràng, không traceback.
- [ ] Bảng in đủ 5 cấu hình × số mức SNR đã truyền.

**Verify:** `python -m pytest -q tests/test_evaluate.py`

**Dependencies:** Task 7, 9, 10
**Files:** `scripts/evaluate.py`, `tests/test_evaluate.py`
**Scope:** S

### Task 12: Thu dữ liệu và chạy đánh giá
**Mô tả:** Bạn thu `data/speech.wav` (20–30 s giọng) và `data/fan.wav` (≥ 30 s quạt). Chạy đánh giá, dán bảng vào README cùng hướng dẫn chạy ngắn.

**Acceptance:**
- [ ] Ở 5 dB, cả chuỗi: SNR tăng ≥ 5 dB và STOI giảm ≤ 0.02.
- [ ] Cấu hình 3 có STOI ≥ cấu hình 2 ở cả 3 mức SNR.
- [ ] Không đạt thì báo số liệu và đề xuất chỉnh hằng số. Chỉ sửa tiêu chí khi bạn đồng ý.

**Verify:** `python scripts/evaluate.py --clean data/speech.wav --noise data/fan.wav --snr 0 5 10`

**Dependencies:** Task 11
**Files:** `README.md` (có thể thêm `voiceagent/dsp.py` nếu chỉnh hằng số)
**Scope:** S

## Checkpoint 4: Sau Task 11–12
- [ ] Tiêu chí 7, 8 của spec đạt, hoặc bạn đã duyệt chỉnh
- [ ] Bạn duyệt trước khi sang Phase 5

## Phase 5: Agent

### Task 13: Tool DSP cho agent
**Mô tả:** 8 hàm tool trong `agent.py` (`get_status`, `calibrate_noise`, `set_bandpass`, `set_spectral_subtraction`, `set_noise_estimator`, `set_noise_gate`, `set_bypass`, `set_monitor`). Tạo bởi `make_dsp_tools(pipeline)`, mỗi hàm có type hint + docstring, trả dict. Lỗi tham số trả `{"error": ...}`.

**Acceptance:**
- [ ] Tham số hợp lệ đổi được Params; `get_status` trả Params + Meters.
- [ ] Tham số sai trả `{"error": ...}`, Params giữ nguyên, không ném exception.

**Verify:** `python -m pytest -q tests/test_agent.py`

**Dependencies:** Task 5, 10
**Files:** `voiceagent/agent.py`, `tests/test_agent.py`
**Scope:** S

### Task 14: Hành động ngoài có xác nhận (`actions.py`)
**Mô tả:** `make_action_tools(confirm, notes_dir)` trả 4 tool: `open_app`, `open_url`, `search_web`, `save_note`. Đi kèm `ALLOWED_APPS` theo Open Question 6 của spec. Mỗi tool kiểm tra đầu vào trước, sau đó hỏi `confirm`, đồng ý mới làm. `Popen` không qua shell; `webbrowser.open`; slug an toàn; không ghi đè.

**Acceptance:**
- [ ] `open_app`: ngoài danh sách thì trả lỗi, không gọi Popen, không hỏi confirm. Từ chối thì `cancelled`, không gọi gì. Đồng ý thì gọi đúng exe.
- [ ] `open_url` từ chối `file://`, `javascript:`, chuỗi không phải URL. `search_web` mã hóa đúng tiếng Việt và dấu cách.
- [ ] `save_note`: title `../../evil` vẫn nằm trong `notes/`; trùng tên thêm số, không ghi đè.

**Verify:** `python -m pytest -q tests/test_actions.py` (Popen và webbrowser thay bằng hàm giả qua monkeypatch)

**Dependencies:** None
**Files:** `voiceagent/actions.py`, `tests/test_actions.py`, `.gitignore` (thêm `notes/`)
**Scope:** S

### Task 15: Vòng hội thoại Gemini đầy đủ
**Mô tả:** System prompt theo spec. Chat Gemini với lịch sử chữ + tool DSP + tool hành động (automatic function calling). `handle_turn(text, chat, speak, log)` gửi câu tới chat, đọc trả lời, ghi transcript. `confirm` thật = đọc câu hỏi qua TTS + chờ gõ y/n. Nối vào `__main__`: lượt nói → STT → `handle_turn`. In "đang nghĩ…". In mỗi tool được gọi.

**Acceptance:**
- [ ] `handle_turn` với hàm giả: chữ rỗng thì không gọi chat; có chữ thì chat → speak đúng câu trả lời → ghi transcript; chat ném lỗi thì báo lỗi, phiên vẫn chạy.
- [ ] Thủ công: 6 kịch bản trong Success Criteria 4 của spec đều đạt.
- [ ] Thủ công: thiếu key, ngắt mạng, `quit`, Ctrl+C đều thoát hoặc báo lỗi sạch; WAV + transcript được ghi.

**Verify:** `python -m pytest -q`; thủ công: `python -m voiceagent`, chạy 6 kịch bản.

**Dependencies:** Task 6b, 13, 14
**Files:** `voiceagent/agent.py`, `voiceagent/__main__.py`, `tests/test_agent.py`
**Scope:** M

## Checkpoint 5: Hoàn thành
- [ ] Toàn bộ Success Criteria 1–8 trong SPEC.md đạt
- [ ] `/agent-skills:code-simplify` và ponytail review không còn vấn đề chặn
- [ ] Sẵn sàng `/agent-skills:ship`
