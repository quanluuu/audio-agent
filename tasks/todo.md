# Todo: Voice Agent

Mỗi task theo TDD: viết test đỏ → code tối thiểu cho xanh → dọn code. Lệnh test chung: `python -m pytest -q`.

Hướng đã chốt: STT (`faster-whisper` mặc định, Gemini tùy chọn) → DeepSeek Flash trong graph LangGraph → in câu trả lời ra màn hình. Agent không nói (không TTS), không dùng Gemini Live.

## Phase 0: Đồng bộ spec

### Task 0: Đưa SPEC lên repo
**Mô tả:** `.gitignore` đang chặn `SPEC.md` và `CONTRIBUTING.md`, nên B chưa có hợp đồng `Pipeline` và hướng mới. Bỏ 2 dòng đó khỏi `.gitignore` (giữ `CLAUDE.md`). Sửa `CONTRIBUTING.md`: bảng phân công thêm Task 16, 17; mục Bí mật đổi sang `DEEPSEEK_API_KEY` (và `GEMINI_API_KEY` khi `--stt gemini`). Nhánh `spec-deepseek-langgraph`, PR cho B duyệt.

**Acceptance:**
- [x] `git ls-files` có `SPEC.md`, `CONTRIBUTING.md`; không có `CLAUDE.md`. (commit `6577b9b`)
- [ ] B duyệt PR. Mục "Giao diện `Pipeline`" khớp với code B đã viết.

**Verify:** `git ls-files SPEC.md CONTRIBUTING.md`; `python -m pytest -q`

**Dependencies:** None
**Files:** `.gitignore`, `SPEC.md`, `CONTRIBUTING.md`, `tasks/plan.md`, `tasks/todo.md`
**Scope:** XS

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
- Thủ công (xong 2026-11-02): đeo tai nghe, chạy `python -m voiceagent --no-agent` 2 phút, nghe thấy giọng mình (đã band-pass). Meters báo 0 underrun. Ctrl+C sinh 2 file WAV.

**Dependencies:** Task 2
**Files:** `voiceagent/audio.py`, `voiceagent/__main__.py`, `tests/test_audio.py`
**Scope:** M

## Checkpoint 1: Sau Task 1–3
- [x] Toàn bộ test xanh (30/30)
- [x] Nghe trực tiếp ổn định 2 phút, 0 underrun — phiên 2026-11-02 dài 159.9 s (`out/20261102-160821_*.wav`): underrun 0, 0.28 ms/khối. Chỉ có band-pass (gate và SS tắt mặc định)
- [x] Ghi lại độ trễ cảm nhận được, trả lời Open Question 2 của spec — bạn nghe thử 2026-11-02: chấp nhận được (cảm nhận, chưa đo số ms)
- [ ] Bạn duyệt trước khi sang Phase 2

## Phase 2: Giọng nói (spike rủi ro trước)

### Task 4: ~~TTS giọng Việt~~ (bỏ)
Đã bỏ 2026-11-09: agent chỉ in chữ (SPEC Assumption 6). pyttsx3 không có giọng Việt SAPI; Piper chạy được nhưng giọng đọc không cần cho mục tiêu học. Code cũ giữ ở nhánh `task-04-vietnamese-tts`, không merge.

### Task 5: Monitor bật/tắt + thu câu nói theo lượt
**Mô tả:** Thêm `monitor` vào `Params` + `Pipeline.set_monitor`. Callback: monitor tắt thì ghi khối 0 ra tai nghe. Thêm bộ thu câu nói: `start_utterance()` / `stop_utterance() -> bytes | None`. Khi đang giữ lượt, callback chép khối đã khử ồn vào bộ đệm. Lúc dừng thì đóng gói thành WAV 16 kHz mono trong RAM. Câu < 0.3 s trả `None`. `--no-agent` bật monitor sẵn; chế độ thường thì tắt, trừ khi có `--monitor`. Task này sửa `pipeline.py` của B: làm sớm, merge nhanh. Gộp thêm: cờ `--all-filters` bật band-pass + SS (`minstat`, `smoothing=0.5`) + gate, tức cấu hình 5 của spec, vì `Params` mặc định tắt SS và gate nên trước đây không thử realtime được cả chuỗi. Bản đầu dùng `calibrate` + hiệu chuẩn 1 s: phiên 2026-11-02 có 1 s đầu không im lặng (cao hơn ồn ~9 dB), nên giọng bị trừ 6.5 dB. Chuyển sang `minstat`: chạy lại bản ghi đó offline, giọng chỉ giảm 4.4 dB, ồn giảm 29.2 dB.

**Acceptance:**
- [x] Monitor tắt: tai nghe nhận khối 0. Monitor bật: nhận khối đã xử lý.
- [x] Chỉ khối nằm giữa start và stop vào bộ đệm, và đó là khối đã khử ồn.
- [x] WAV trong RAM đọc lại đúng 16 kHz, mono, đúng độ dài; câu < 0.3 s trả `None`.
- [x] `enable_all_filters` bật cả chuỗi với `minstat`, `smoothing=0.5`, không hiệu chuẩn. Ồn trắng ổn định giảm ≥ 10 dB sau 1.5 s mà không cần 1 s im lặng.

**Verify:** `python -m pytest -q` (57/57 xanh); thủ công: `python -m voiceagent --no-agent --all-filters` bật quạt ≥ 2 phút, nghe ồn giảm, giọng vẫn rõ, Meters 0 underrun < 2 ms/khối (đóng mục Checkpoint 3).

**Dependencies:** Task 3
**Files:** `voiceagent/pipeline.py`, `voiceagent/audio.py`, `voiceagent/__main__.py`, `tests/test_pipeline.py`, `tests/test_audio.py`
**Scope:** M

### Task 6: Whisper STT + vòng "nói → thấy chữ"
**Mô tả:** Đầu task, cài `faster-whisper` trên Python 3.13/Windows và chạy thử model `small` với 1 file WAV float32. Cài lỗi (wheel `ctranslate2`, `av`) thì **báo bạn trước khi đổi hướng**. Viết `voice.whisper_transcribe(model) -> transcribe(wav_bytes) -> str`: đọc WAV thành mảng float32, gọi `model.transcribe(audio, language="vi")`, nối chữ các segment, strip. Model nạp một lần lúc khởi động (`WhisperModel(WHISPER_MODEL or "small", device="cpu", compute_type="int8")`), in "đang tải model Whisper…"; import `faster_whisper` nằm trong hàm. Viết `append_transcript(path, speaker, text)`. Sửa `__main__`: Enter trên dòng trống để bắt đầu/dừng lượt, in "🎙 đang nghe…" rồi "đang nghe chữ…", in `Bạn: …`, ghi transcript. Gõ chữ thì ghi `Bạn (gõ): …`. Chưa có chat. `--no-agent` không nạp Whisper. Sửa help `--no-agent` (bỏ chữ "Gemini").

**Acceptance:**
- [x] Model giả trả segment `" xin"`, `" chào"` → `"xin chào"`; không segment → `""`. Transcript ghi đúng định dạng (file tạm, `encoding="utf-8"`).
- [x] STT ném lỗi hoặc model không tải được: thông báo rõ, không traceback, stream không dừng, vẫn gõ chữ được.
- [x] Thủ công: nói 5 câu tiếng Việt, đúng ý ≥ 4/5. Câu 5 s có chữ trong ≤ 5 s (SPEC Success Criteria 6). Ghi thời gian thực đo. Kém thì thử `WHISPER_MODEL=medium` rồi báo bạn.

Spike 2026-11-09: faster-whisper 1.2.1 + ctranslate2 4.8.4 (4.8.5 mới 7 ngày, chưa ghim) + av 19.0.2. Lần đầu tải model `small` mất ~17 phút (mạng chậm). `av` 19 bỏ `metadata_errors` nên đưa file-like vào bị `TypeError`; chuyển sang đưa mảng numpy. Giọng Piper tổng hợp: nạp model ~3 s; câu 2.8 s ra chữ trong 4.4 s, gần đúng ("mưu" thay "mưa", "ổn" thay "ồn"). Thử giọng thật 2026-11-11 (`small`, không `--all-filters`, không quạt), mỗi câu ra chữ ~1 s:
  - "xin chào" → "Xin chào" ✓; "tôi vừa đi xem phim" → đúng ✓
  - "tôi tên là quân" → "tùy tên lạc quân" ✗
  - "hello bạn" → "hơi lúc mà" ✗ (lẫn tiếng Anh); "tôi 2k5" → "tối hài ca nằm" ✗ (tiếng lóng)
  - Đúng 2/5, chưa đạt ≥ 4/5. Câu 4–5 không phải tiếng Việt chuẩn; cần thử lại bằng câu trong Success Criteria 4.
Thử lại 2026-11-11 với câu Success Criteria 4: đúng ý 4/5, đạt.
  - "Thù đồ của Pháp là gì?", "cho mình nghe lại dọc mình": sai chính tả, vẫn hiểu ý ✓
  - "Còn ồn quá", "Tìm trên Google thời tiết Hà Nội" ✓
  - "Mở Notepad" → "Mở nút bắt" ✗: tên app tiếng Anh. Sau Task 14 thử `hotwords`/`initial_prompt` chứa tên app trong `ALLOWED_APPS`.

**Verify:** `python -m pytest -q`; thủ công: `python -m voiceagent`, nói 5 câu, mở `out/*_transcript.md`.

**Dependencies:** Task 5
**Files:** `requirements.txt` (thêm `faster-whisper==<bản đã cài>`), `voiceagent/voice.py`, `voiceagent/__main__.py`, `tests/test_voice.py`
**Scope:** M

### Task 6b: Gemini STT + cờ `--stt`
**Mô tả:** Đầu task, đọc docs chính thức `google-genai` về gửi audio inline, chọn model rẻ nhất nhận audio cho `GEMINI_STT_MODEL` (Open Question 7). Viết `gemini_transcribe(client, model)` với prompt "chép lại nguyên văn tiếng Việt, không thêm gì". Viết `make_transcriber(backend, env, load_whisper, make_gemini)`: `"whisper"` không bao giờ gọi Gemini; `"gemini"` có key thì dùng Gemini; `"gemini"` thiếu key hoặc không import được `google-genai` thì in một lần "Không dùng được Gemini STT, chuyển sang Whisper" và dùng Whisper. Thêm cờ `--stt whisper|gemini` (mặc định `whisper`).

**Acceptance:**
- [x] 3 nhánh của `make_transcriber` đúng như trên, với hàm Whisper/Gemini giả; thông báo fallback in đúng một lần.
- [x] Thủ công: 5 câu ở Task 6 chạy lại với `--stt gemini`. Ghi so sánh (đúng bao nhiêu câu, thời gian) với Whisper.

Docs 2026-11-11 (aistudio.google.com/docs/audio, /pricing): gửi WAV inline (`audio/wav`, ≤ 20 MB/request) qua `client.models.generate_content`; model mặc định `gemini-3.5-flash-lite`. `--stt gemini` thiếu key đã thử: fallback Whisper đúng. Thêm `load_env()` đọc `.env` lúc khởi động (trước đó key trong `.env` không được đọc).
So sánh 2026-11-11 (5 câu Success Criteria 4, không lọc, không quạt):
  - Whisper `small`: đúng ý 4/5, ~1 s/câu. Sai "Thù đồ", "dọc mình", mất tên app ("Mở nút bắt").
  - Gemini `gemini-3.5-flash-lite`: đúng ý 5/5, chậm hơn (~2.7 s/câu khi gọi thử). Chỉ sai "ổn" thay "ồn"; nhận đúng "NotePad".

**Verify:** `python -m pytest -q tests/test_voice.py`; thủ công như trên.

**Dependencies:** Task 6
**Files:** `voiceagent/voice.py`, `voiceagent/__main__.py`, `tests/test_voice.py`
**Scope:** S

## Checkpoint 2: Sau Task 5–6b
- [x] Toàn bộ test xanh (75/75)
- [x] Nói → thấy đúng chữ
- [x] Ghi chú chất lượng STT Whisper vs Gemini (đúng bao nhiêu câu, lỗi kiểu gì, thời gian) để so sánh sau khi có bộ lọc
- [x] Bạn duyệt trước khi sang Phase 5 (2026-11-11)

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
- [x] Toàn bộ test xanh (51/51 trên Windows)
- [x] Cả chuỗi chạy realtime 2 phút, 0 underrun, < 2 ms/khối — 2026-11-02 `--all-filters` (minstat), 158.0 s, 0 underrun, 0.44 ms/khối. Ồn giảm 35.4 dB, giọng giảm 4.7 dB (ước lượng theo phân vị 20%); phần giảm ở giọng để Task 12 chỉnh
- [x] (Sau Task 6) Bật quạt, nói 5 câu ở Task 6 khi bật lọc và khi bypass: so sánh STT, ghi kết quả — 2026-11-11, `--stt gemini`, bật quạt:
  - Không lọc: đúng ý 5/5 (1 lỗi chính tả "ổn"/"ồn"), "NotePad" đúng.
  - `--all-filters` (`out/20261111-010736_transcript.md`): đúng ý 4/5, "Mở Notepad" → "mở nút thắt".
  - Bộ lọc không giúp Gemini, có thể làm hỏng từ tiếng Anh (khớp ghi nhận của paper: SS làm tăng WER). Mẫu nhỏ (5 câu), chưa đủ kết luận; Task 12 xem lại độ méo giọng.
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

**Đề xuất chỉnh tham số (từ A, 2026-11-11, chờ B kiểm bằng STOI):** đổi mặc định `high_hz` 3800 → 7000 và `alpha` 2.0 → 1.0.

Lý do:
- Phiên `out/20261111-010736` (bật quạt, `--all-filters`, `--stt gemini`): "Mở Notepad" → "mở nút thắt"; tắt lọc thì Gemini nghe đúng "NotePad".
- Spectrogram raw vs clean: ồn giảm 45 dB, nhưng giọng mất 17 dB ở 4–8 kHz (band-pass cắt ở 3.8 kHz, kiểu điện thoại) và 4–5 dB ở 1–4 kHz. Trong khung có giọng, 81% bin bị cắt > 10 dB: SS `alpha=2` + gate xóa phụ âm bật/xát yếu (`t`, `p`, `d`, `s`), là các âm STT cần để phân biệt từ, nhất là từ tiếng Anh.
- Chạy lại bản raw offline qua `Pipeline` rồi cho Whisper `small` nhận dạng cả file (5 câu Success Criteria 4):

| Cấu hình | Ồn giảm (dB) | Giọng giảm (dB) | Đúng ý | Ghi chú |
|---|---|---|---|---|
| raw | 0 | 0 | 3/5 | |
| hiện tại: bp 100–3800, SS α=2, gate | 45.2 | 3.2 | 2/5 | hỏng câu 5 ("Điểm sinh Google thể tiếp hạ lụt") |
| bp 100–7000, SS α=2, gate | 44.7 | 3.1 | 3/5 | Whisper bịa câu ở đoạn im |
| **bp 100–7000, SS α=1, gate** | 33.0 | 1.9 | 3/5 | câu 3 và 5 đúng nguyên văn; bịa câu ở đoạn im |
| bp 100–7000, SS α=1, không gate | 13.7 | 1.6 | 3/5 | |
| chỉ bp 100–7000 | 0.6 | 1.0 | 3/5 | giống raw |

- `high_hz=7000` + `alpha=1` giữ độ đúng bằng raw, vẫn giảm ồn 33 dB, giọng chỉ mất 1.9 dB. Cấu hình hiện tại là cấu hình duy nhất kém hơn raw.
- "Thủ đô" và "Notepad" sai ở mọi cấu hình, kể cả raw: giới hạn của Whisper `small`, không phải do bộ lọc.
- Gate tạo đoạn im tuyệt đối, Whisper bịa câu ("Hãy subscribe cho kênh…"). Chạy theo lượt thì ít gặp hơn, nhưng cần để ý.
- Hạn chế: 1 phiên, nhận dạng cả file (không theo lượt), chỉ Whisper (không Gemini), mức ồn/giọng ước lượng theo phân vị năng lượng khung, chưa đo STOI. Script: `scratchpad/offline_stt.py` (ngoài repo); cần thì A đưa vào `scripts/`.

**Verify:** `python scripts/evaluate.py --clean data/speech.wav --noise data/fan.wav --snr 0 5 10`

**Dependencies:** Task 11
**Files:** `README.md` (có thể thêm `voiceagent/dsp.py` nếu chỉnh hằng số)
**Scope:** S

## Checkpoint 4: Sau Task 11–12
- [ ] Tiêu chí 8, 9 của spec đạt, hoặc bạn đã duyệt chỉnh
- [ ] Bạn duyệt trước khi sang Phase 5

## Phase 5: Agent (LangGraph + DeepSeek)

### Task 14: Hành động ngoài (`actions.py`)
**Mô tả:** 4 tool `open_app`, `open_url`, `search_web`, `save_note(notes_dir)` cùng `ALLOWED_APPS` (Open Question 6 của spec). Tool **không** nhận `confirm` và không import LangGraph; xác nhận nằm ở node `confirm` (Task 16). Thêm `describe_action(name, args) -> str`: kiểm tra hợp lệ rồi trả câu hỏi xác nhận, sai thì ném `ValueError`. Mỗi tool vẫn tự kiểm tra lại trước khi làm. `Popen([exe])` không qua shell; `webbrowser.open`; slug an toàn; không ghi đè. Docstring kiểu Google (`Args:`) để LangChain lấy mô tả tham số. Xuất `ACTION_TOOLS`.

**Acceptance:**
- [x] `open_app`: ngoài danh sách thì trả lỗi, không gọi Popen. Hợp lệ thì gọi đúng exe. `open_url` từ chối `file://`, `javascript:`, chuỗi không phải URL. `search_web` mã hóa đúng tiếng Việt và dấu cách.
- [x] `describe_action`: `open_app("cmd")`, `open_url("file:///C:/")` ném `ValueError`; `open_app("notepad")` trả câu hỏi có "Notepad".
- [x] `save_note`: title `../../evil` vẫn nằm trong `notes/`; trùng tên thêm số, không ghi đè.

`ALLOWED_APPS` dùng đường dẫn đầy đủ: `notepad`, `calculator` trong `%SystemRoot%\System32`; `paint` là app Store trên Win11, chạy qua alias `%LOCALAPPDATA%\Microsoft\WindowsApps\mspaint.exe`. Chưa mở app thật lần nào.

**Verify:** `python -m pytest -q tests/test_actions.py` (Popen và webbrowser thay bằng hàm giả qua monkeypatch)

**Dependencies:** None
**Files:** `voiceagent/actions.py`, `tests/test_actions.py`, `.gitignore` (kiểm tra có `notes/`)
**Scope:** S

### Task 13: Tool DSP cho agent
**Mô tả:** 8 hàm tool trong `agent.py` (`get_status`, `calibrate_noise`, `set_bandpass`, `set_spectral_subtraction`, `set_noise_estimator`, `set_noise_gate`, `set_bypass`, `set_monitor`). Tạo bởi `make_dsp_tools(pipeline)`, mỗi hàm có type hint + docstring kiểu Google (`Args:`), trả dict. Lỗi tham số trả `{"error": ...}`. Chưa cần LangGraph.

**Acceptance:**
- [x] Tham số hợp lệ đổi được Params; `get_status` trả Params + Meters.
- [x] Tham số sai trả `{"error": ...}`, Params giữ nguyên, không ném exception.

**Verify:** `python -m pytest -q tests/test_agent.py`

**Dependencies:** Task 5, 10
**Files:** `voiceagent/agent.py`, `tests/test_agent.py`
**Scope:** S

### Task 15: Graph LangGraph tối thiểu `llm ↔ tools` + spike DeepSeek
**Mô tả:** Đầu task: cài `langgraph` 1.x và `langchain-deepseek`, ghim phiên bản. Đọc docs chính thức LangGraph đúng bản đã cài (đường import `StateGraph`, `MessagesState`, `interrupt`, `Command`, `InMemorySaver`). Đọc docs DeepSeek (`api-docs.deepseek.com`) để chốt tên model (Open Question 1 của spec). Spike: gọi thật `ChatDeepSeek(...).bind_tools(dsp_tools)` với "tắt khử ồn đi", kiểm tra trả về tool call `set_bypass`. Lỗi tool calling thì báo bạn trước khi đổi gói. Sau đó viết `AgentState`, `SYSTEM_PROMPT`, node `llm`, node `tools` tự viết (tuần tự, `ToolMessage` cho mọi tool call, tên lạ / sai tham số → `{"error"}`, in `[tool] …`), `route_after_llm` (chưa có nhánh `confirm`), `build_graph(model, dsp_tools, action_tools, checkpointer)`.

**Acceptance:**
- [x] Model giả không trả tool call: graph kết thúc ở `END`, trả đúng câu.
- [x] Model giả gọi `set_bypass(enabled=True)`: Params đổi, `ToolMessage` đúng `tool_call_id`, model được gọi 2 lần. Tên tool lạ / thiếu tham số: `ToolMessage` chứa `error`, graph không ném lỗi.
- [x] `convert_to_openai_tool(set_bandpass)` có 3 tham số bắt buộc, mỗi tham số có mô tả. Hai lượt cùng `thread_id`: lượt 2 thấy message lượt 1.

Docs 2026-11-11: model `deepseek-flash` (V4.1-Flash-1017), base `https://api.deepseek.com`, thinking bật sẵn → tắt bằng `extra_body`. Bật thinking + tools thì phải gửi lại `reasoning_content` mỗi lượt, không thì 400. Giá off-peak $0.15/1M input, $0.6/1M output. Cài: langgraph 1.2.17, langchain-deepseek 1.1.1, langchain-core 1.7.1 (bản ≥ 2 tuần). Import trong bản đã cài khớp: `langgraph.graph` (`StateGraph`, `MessagesState`, `START`, `END`), `langgraph.types` (`interrupt`, `Command`), `langgraph.checkpoint.memory.InMemorySaver`. Spike gọi thật 2026-11-11 (`deepseek-flash`, thinking tắt), tool calling chạy:
  - "tắt khử ồn đi" → `get_status()` trước (đúng system prompt), thấy SS đang tắt nên hỏi lại có muốn bypass không. 2.1 s.
  - "Thủ đô của Pháp là gì?" → "Paris.", không gọi tool, 0.6 s.
  - "bỏ qua toàn bộ bộ lọc để mình so sánh với giọng gốc" → `set_bypass(enabled=True)` → `{"status": "ok"}`, Params đổi, trả lời đúng.
  - Hai lượt cùng `thread_id` nhớ ngữ cảnh ("còn ồn quá" → gợi ý bật lại trừ phổ).

**Verify:** `python -m pytest -q tests/test_agent.py`; spike thủ công với `DEEPSEEK_API_KEY`.

**Dependencies:** Task 13
**Files:** `requirements.txt`, `voiceagent/agent.py`, `tests/test_agent.py`
**Scope:** M

### Task 16: Node `confirm` + `interrupt()` cho hành động ngoài
**Mô tả:** Thêm trường `preset` vào `AgentState`, node `confirm`, nhánh `confirm` trong `route_after_llm` (có ít nhất một tool thuộc `ACTION_TOOLS`). `confirm` gọi `describe_action`: sai thì ghi `preset[id] = {"error"}` và không hỏi; hợp lệ thì `interrupt({"question": …})`, từ chối thì `preset[id] = {"status": "cancelled"}`. `confirm` **không có tác dụng phụ** (resume chạy lại node). `tools` dùng `preset` nếu có, rồi xóa `preset`.

**Acceptance:**
- [ ] `open_app("notepad")`: `invoke` trả `__interrupt__` có câu hỏi. Resume `True` → `Popen` gọi đúng 1 lần. Resume `False` → không gọi, `ToolMessage` chứa `cancelled`.
- [ ] `open_app("cmd")`: không interrupt, `ToolMessage` chứa `error`, không gọi `Popen`.
- [ ] Một `AIMessage` có 2 tool hành động: 2 interrupt; đồng ý cái đầu, từ chối cái sau → chỉ tool đầu chạy, đúng 1 lần.

**Verify:** `python -m pytest -q tests/test_agent.py`

**Dependencies:** Task 14, 15
**Files:** `voiceagent/agent.py`, `tests/test_agent.py`
**Scope:** S

### Task 17: `handle_turn` + vòng hội thoại đầy đủ
**Mô tả:** `handle_turn(text, graph, config, out, ask_yes_no, log)`: invoke graph, lặp khi còn `__interrupt__` (in câu hỏi qua `out`, `ask_yes_no()` đọc `y`/`n`, resume bằng `Command(resume=ok)`), in trả lời, ghi transcript. `recursion_limit=16`; bắt `GraphRecursionError` → in "Mình chưa làm được việc này". Bắt lỗi API (mạng, 401, 402, 429, 5xx, timeout) → in rõ, phiên vẫn chạy. Nối vào `__main__`: lượt nói → STT → "đang nghĩ…" → `handle_turn`. Thiếu `DEEPSEEK_API_KEY`: báo rõ, vẫn chạy DSP + STT + in transcript. `InMemorySaver`, `thread_id` = thời gian bắt đầu phiên.

**Acceptance:**
- [ ] Với graph/out/ask_yes_no giả: chữ rỗng thì không gọi graph; có chữ thì in đúng câu trả lời và ghi transcript; có interrupt thì hỏi rồi resume; graph ném lỗi → báo lỗi, phiên vẫn chạy; model luôn gọi tool → câu dự phòng.
- [ ] Thủ công: 6 kịch bản trong Success Criteria 4 của spec đều đạt.
- [ ] Thủ công: thiếu key, ngắt mạng, `quit`, Ctrl+C đều thoát hoặc báo lỗi sạch; WAV + transcript được ghi.

**Verify:** `python -m pytest -q`; thủ công: `python -m voiceagent`, chạy 6 kịch bản.

**Dependencies:** Task 6b, 16
**Files:** `voiceagent/agent.py`, `voiceagent/__main__.py`, `tests/test_agent.py`
**Scope:** M

## Checkpoint 5: Hoàn thành
- [ ] Toàn bộ Success Criteria 1–9 trong SPEC.md đạt
- [ ] `/agent-skills:code-simplify` và ponytail review không còn vấn đề chặn
- [ ] Sẵn sàng `/agent-skills:ship`
