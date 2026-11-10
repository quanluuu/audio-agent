# Spec: Voice Agent — Trợ lý giọng nói có khử ồn DSP

## Giả định
1. Chạy local trên Windows 11, một người dùng, Python 3.13.
2. Âm thanh mono, 16 kHz, float32 trong [-1, 1]. Micro và loa/tai nghe mặc định của hệ điều hành (chọn thiết bị khác qua cờ `--input`/`--output`).
3. Người dùng đeo **tai nghe**. Dùng loa ngoài mà bật monitor thì giọng lọt lại vào micro, gây hú. Không làm khử tiếng vang (echo cancellation).
4. LLM chat là **DeepSeek Flash**, gọi qua `ChatDeepSeek` (gói `langchain-deepseek`) bên trong một graph **LangGraph**. Key nằm trong biến môi trường `DEEPSEEK_API_KEY`. Tên model đọc từ `DEEPSEEK_MODEL`, mặc định `deepseek-v4-flash`. Tên này phải kiểm lại với docs chính thức ở đầu task chat. DeepSeek V4/V4.1 Flash **không nhận audio**: đầu vào chỉ có chữ và ảnh. Vì vậy STT dùng hệ khác (giả định 5).
5. **STT có 2 backend**, chọn bằng cờ `--stt whisper|gemini`:
   - `whisper` (mặc định): `faster-whisper` chạy local trên CPU, `compute_type="int8"`, `language="vi"`. Tên model đọc từ `WHISPER_MODEL`, mặc định `small` (~460 MB, tải về ở lần chạy đầu). Không cần key, chạy được khi không có mạng.
   - `gemini`: Gemini nhận audio qua `google-genai`, key nằm trong `GEMINI_API_KEY`. Tên model đọc từ `GEMINI_STT_MODEL`, mặc định `gemini-3.5-flash-lite` (rẻ nhất nhận audio, $0.30/1M token audio, 32 token/giây).
6. Agent **không nói**: chỉ in câu trả lời ra màn hình. Không có TTS (bỏ pyttsx3/Piper vì máy không có giọng Việt SAPI, và giọng đọc không phục vụ mục tiêu học DSP + agent).
7. Mỗi phiên ngắn (vài phút đến vài chục phút). Bản ghi giữ trong RAM rồi ghi WAV khi thoát.
8. Ồn nền tương đối ổn định (quạt, máy lạnh, tiếng rè). Ồn thay đổi nhanh (người nói chuyện, nhạc) nằm ngoài khả năng của DSP cổ điển.

## Mục tiêu
Xây một trợ lý giọng nói để học: bạn **nói chuyện trực tiếp** với agent. Giọng bạn được khử ồn bằng DSP cổ điển, ghi âm lại, chuyển thành chữ (Whisper local hoặc Gemini), rồi gửi cho DeepSeek để hiểu nội dung. Agent thực hiện yêu cầu (trả lời câu hỏi, ghi chú, chỉnh bộ lọc, mở app/web) và **in câu trả lời** ra màn hình.

Người dùng: chính tác giả, để hiểu:
1. DSP khử ồn hoạt động thế nào.
2. Xử lý âm thanh theo khối thời gian thực.
3. Khử ồn ảnh hưởng thế nào tới nhận dạng giọng nói. Đây là chủ đề của Moore et al. (2017).
4. LLM tool calling với hành động có xác nhận, xây bằng LangGraph (StateGraph, cạnh điều kiện, checkpointer, `interrupt()` cho human-in-the-loop).

Nguyên tắc kiến trúc:
- **LLM không nằm trong đường đi realtime của âm thanh.** Mỗi khối 16 ms được DSP xử lý trong luồng audio. LLM chỉ nhận cả câu nói đã ghi xong, và chỉ sửa tham số dùng chung qua tool.
- **Mọi hành động ra ngoài chương trình đều cần bạn xác nhận** (mở app, mở web, ghi file). Nhận dạng giọng nói có thể sai, nên agent phải hỏi lại trước khi làm.

```
               ┌──────────── monitor (tắt mặc định) ────────────► tai nghe
 micro ──► [audio callback: Pipeline.process_block] ──► bản ghi phiên (raw + clean WAV)
                     ▲                      │
              Params (có lock)              │ khi đang giữ lượt nói (Enter … Enter)
                     ▲                      ▼
                     │               câu nói đã khử ồn (WAV trong RAM)
                     │                      │
                     │              ① STT: audio → chữ
                     │                 (--stt whisper: faster-whisper local | gemini: Gemini API)
                     │                      │
           tool calls│              ② DeepSeek chat + tools ──► xác nhận y/n ──► app / web / notes/
                     └──────────────────────┤
                                            ▼
                                    ③ in câu trả lời ra màn hình
                                    + ghi transcript
```

STT và chat là **2 bước riêng**, vì DeepSeek không nhận audio. Tách riêng còn có lợi:
- Từng bước tách bạch, dễ hiểu.
- Có transcript chữ để lưu và để đo chất lượng nhận dạng.
- Lịch sử hội thoại chỉ gồm chữ, không phải gửi lại audio mỗi lượt.
- Đổi được backend STT để so sánh Whisper với Gemini trên cùng một câu nói đã khử ồn.

### Lượt nói (push-to-talk)
- Nhấn Enter trên dòng trống: bắt đầu ghi. Nhấn Enter lần nữa: dừng ghi và gửi đi.
- Gõ chữ rồi Enter: gửi thẳng câu chữ, bỏ qua bước STT. Dùng khi STT nghe sai hoặc khi không có micro.
- Lệnh `quit`: thoát.
- Trong lúc agent đang nghĩ hoặc đang nói, chương trình không nhận lượt nói mới.

### Module (xây theo thứ tự dsp → stream → voice → agent)

| Module | Việc | Phụ thuộc | Trạng thái |
|---|---|---|---|
| dsp | Bộ lọc thuần numpy/scipy, xử lý từng khối, có state giữa các khối | — | BandPass xong |
| stream | Pipeline + tham số dùng chung + đo chỉ số; nối micro/loa bằng `sounddevice`; ghi WAV phiên; thu câu nói theo lượt; monitor bật/tắt | dsp | Pipeline, ghi WAV xong |
| voice | STT (faster-whisper hoặc Gemini), transcript | stream | Chưa |
| agent | LangGraph StateGraph: DeepSeek chat + tools (DSP, ghi chú, mở app/web) + xác nhận bằng `interrupt()`, vòng hội thoại | voice | Chưa |

### dsp — các bộ lọc (thứ tự trong chuỗi)
1. **Band-pass** (`BandPass`): Butterworth IIR bậc 4, dạng `sos`, mặc định 100–3800 Hz (dải chứa giọng nói). Dùng `scipy.signal.sosfilt` với state `zi` giữ giữa các khối, để không có tiếng "click" ở ranh giới khối. Không dùng `sosfiltfilt` vì nó cần cả tín hiệu tương lai.
2. **Spectral subtraction** (`SpectralSubtractor`): FFT 512 điểm, hop 256 (chồng 50%), cửa sổ sqrt-Hann ở cả phân tích và tổng hợp, ghép lại bằng overlap-add. Độ trễ thêm: 256 mẫu (16 ms). Mỗi khung, với từng bin tần số k:
   - Gain thô: `G[k] = max(1 - alpha·|N[k]| / |X[k]|, beta)`. Cách viết này tương đương `|Y| = max(|X| - alpha·|N|, beta·|X|)`.
   - **Làm mượt gain theo thời gian** để giảm musical noise: `G_s[k] = smoothing·G_s_prev[k] + (1 - smoothing)·G[k]`. Đầu ra: `Y = G_s·X`, giữ nguyên pha của X.
   - Mặc định `alpha=2.0`, `beta=0.02`, `smoothing=0.5`. `smoothing=0` cho SS gốc, không làm mượt.
   - `|N|` lấy từ một trong 2 chế độ ước lượng ồn (`noise_mode`):
     - `calibrate` (mặc định): trung bình phổ biên độ trong lúc hiệu chuẩn, người dùng phải im lặng.
     - `minstat`: minimum statistics rút gọn (Martin, 2001). Làm mượt công suất `P[k] = 0.85·P[k] + 0.15·|X[k]|²`. Giữ P của 96 khung gần nhất (≈1.5 s) trong ring buffer. Lấy `P_min[k]` là giá trị nhỏ nhất trong buffer. `|N[k]| = sqrt(bias·P_min[k])`, với `bias=1.5` để bù vì giá trị min luôn thấp hơn trung bình. Ước lượng tự cập nhật liên tục, không cần im lặng. Nhược điểm: âm kéo dài liên tục hơn 1.5 s (tiếng còi, nốt nhạc ngân) bị coi là ồn và bị trừ.

   Nguồn: hai cải tiến này mượn từ khối SS trong Moore et al. (2017), *Computer Speech & Language* 46, 574–584, đã rút gọn. Paper dùng cepstral smoothing và gain MMSE. Spec thay bằng làm mượt gain theo thời gian cho dễ hiểu.
3. **Noise gate** (`NoiseGate`): tính RMS khối (dB). Dưới ngưỡng thì gain mục tiêu = `floor_gain` (mặc định 0.1), trên ngưỡng thì 1.0. Gain làm mượt theo hàm mũ (attack nhanh, release chậm) để tránh tiếng lụp bụp. Mặc định ngưỡng -45 dBFS.

Hàm hỗ trợ: `rms_db(x)`, `snr_db(clean, test)`, `add_noise(x, snr_db, seed)`, `mix_at_snr(clean, noise, snr_db)` (trộn ồn thật, cắt hoặc lặp ồn cho bằng độ dài giọng).

### stream — pipeline, thu âm, monitor
- `Params`: dataclass bất biến, là bản chụp mọi tham số (bật/tắt + giá trị từng bộ lọc, `bypass`, `monitor`). Setter nằm ở `Pipeline`, kiểm tra miền giá trị, ném `ValueError` trước khi đổi gì. Đọc/ghi qua `threading.Lock`.
- `Pipeline.process_block(block) -> block`: chạy chuỗi lọc theo Params hiện tại, cập nhật Meters. Không cấp phát lớn, không I/O, không print, không gọi mạng.
- Meters: RMS vào (dB), RMS ra (dB), mức ồn nền đã hiệu chuẩn (dB), SNR ước lượng (dB), thời gian xử lý khối trung bình (ms), số lần underrun/overflow.
- Hiệu chuẩn ồn (chế độ `calibrate`): 1 giây đầu sau khi khởi động (yêu cầu người dùng im lặng), hoặc khi agent gọi `calibrate_noise`. Trong lúc hiệu chuẩn, Pipeline cộng dồn phổ thay vì trừ. Chế độ `minstat` bỏ qua bước này.
- `audio.py`:
  - `sounddevice.Stream` song công, blocksize 256 mẫu, gọi `process_block` trong callback.
  - **Monitor**: tắt mặc định, nên tai nghe nhận khối 0. Bật bằng cờ `--monitor` hoặc tool `set_monitor` khi bạn yêu cầu.
  - **Thu câu nói**: khi đang giữ lượt, callback chép khối *đã khử ồn* vào bộ đệm câu nói. Hết lượt thì đóng gói thành WAV 16 kHz trong RAM (`io.BytesIO`). Câu nói dưới 0.3 s bị bỏ qua và báo "không nghe thấy gì".
  - Ghi cả phiên: `out/<thời gian>_raw.wav` và `out/<thời gian>_clean.wav` (float32).

### Giao diện `Pipeline` (hợp đồng giữa phần DSP và phần Agent)
Phần DSP viết các hàm này, phần Agent gọi chúng. Tên và chữ ký cố định từ bây giờ. Muốn đổi thì sửa SPEC trước, trong một PR được cả hai duyệt.

```python
class Pipeline:
    sr: int
    meters: Meters
    def params(self) -> Params: ...
    def process_block(self, block: np.ndarray) -> np.ndarray: ...
    # mọi setter ném ValueError khi ngoài miền và khi đó không đổi gì
    def set_bypass(self, enabled: bool) -> None: ...                                    # xong
    def set_bandpass(self, enabled: bool, low_hz: float, high_hz: float) -> None: ...   # xong
    def set_monitor(self, enabled: bool) -> None: ...                                   # Task 5
    def set_noise_gate(self, enabled: bool, threshold_db: float) -> None: ...           # Task 7
    def set_spectral_subtraction(self, enabled: bool, alpha: float, beta: float,
                                 smoothing: float) -> None: ...                         # Task 8, 9
    def calibrate_noise(self, seconds: float = 1.0) -> None: ...                        # Task 8
    def set_noise_estimator(self, mode: str) -> None: ...  # "calibrate" | "minstat"    # Task 10

@dataclass(frozen=True)
class Params:
    bypass: bool; monitor: bool
    bandpass_enabled: bool; low_hz: float; high_hz: float
    gate_enabled: bool; gate_threshold_db: float
    ss_enabled: bool; alpha: float; beta: float; smoothing: float; noise_mode: str

@dataclass
class Meters:
    input_db: float; output_db: float; noise_floor_db: float; snr_db: float
    process_ms: float; underruns: int; calibrating: bool
```

### voice — nghe
- `transcribe(wav_bytes) -> str`: chuyển câu nói thành chữ. Trả chuỗi rỗng nếu không có lời nói. `make_transcriber(backend, ...)` trả hàm này theo cờ `--stt`:
  - `whisper_transcribe`: model `faster-whisper` được nạp **một lần** lúc khởi động, có in "đang tải model Whisper…". Đọc WAV bằng `scipy.io.wavfile` thành mảng float32 16 kHz, gọi `model.transcribe(audio, language="vi")`, rồi nối chữ của các segment lại. Không đưa file-like vào: faster-whisper 1.2.1 giải mã file qua `av`, mà `av` 19 bỏ tham số `metadata_errors` nên lỗi.
  - `gemini_transcribe`: gửi audio inline kèm prompt "chép lại nguyên văn tiếng Việt, không thêm gì".
  - Chọn `gemini` mà thiếu `GEMINI_API_KEY`, hoặc không import được `google-genai`: in một lần "Không dùng được Gemini STT, chuyển sang Whisper", rồi dùng `whisper`.
  - STT lỗi giữa phiên (mạng, model): in lỗi, bỏ lượt đó, phiên vẫn chạy. Không tự đổi backend giữa phiên.
- Transcript: mỗi lượt ghi 2 dòng `Bạn: …` / `Agent: …` vào `out/<thời gian>_transcript.md`. Lượt gõ chữ ghi `Bạn (gõ): …`.

### agent — LangGraph StateGraph + DeepSeek
Agent là một `StateGraph` của LangGraph, tự khai báo node và cạnh. Không dùng `create_agent` hay `create_react_agent` có sẵn, để thấy rõ từng bước. Graph **chỉ lo phần chat**. STT và việc in câu trả lời nằm ở vòng chính, ngoài graph:

```
lượt nói ──► STT ──► graph.invoke(HumanMessage) ──► in chữ
                          │
              ┌───────────▼───────────┐
     START ──►│          llm          │◄──────────────┐
              └───┬────────┬────────┬─┘               │
        không có  │  chỉ   │ có tool│                 │
        tool call │  tool  │ hành   │                 │
                  ▼  DSP   │ động   ▼                 │
                 END       │     confirm ── interrupt() ──► vòng chính hỏi y/n
                           │        │       ◄── Command(resume=True/False)
                           ▼        ▼                 │
                          tools ◄───┘                 │
                            └─────────────────────────┘
```

**LLM:** `ChatDeepSeek(model=DEEPSEEK_MODEL, timeout=30, max_retries=1)` từ `langchain-deepseek`, key lấy từ `DEEPSEEK_API_KEY`. Gọi `.bind_tools(ALL_TOOLS)` một lần lúc khởi động. LangChain tự tạo schema từ type hint và docstring, nên không viết schema tay.

**State:** `AgentState(MessagesState)` thêm trường `preset: dict[str, dict]`. Trường này ánh xạ `tool_call_id` → kết quả đặt trước (bị từ chối hoặc không hợp lệ), do node `confirm` ghi.

**Node** (mỗi node là hàm thường, nằm trong `agent.py`):
- `llm`: gọi model với `[SystemMessage(SYSTEM_PROMPT)] + messages`. System prompt ghép lúc gọi, không lưu vào state. Kết quả là `AIMessage`.
- `route_after_llm` (cạnh điều kiện):
  - Không có `tool_calls` thì về `END`.
  - Có ít nhất một tool thuộc `ACTION_TOOLS` thì sang `confirm`.
  - Còn lại sang `tools`.
- `confirm`: với mỗi tool call hành động:
  1. Gọi `actions.describe_action(name, args)`. Hàm này kiểm tra hợp lệ rồi trả câu hỏi, ví dụ "Mình sẽ mở Notepad, đồng ý không?". Không hợp lệ (ví dụ `cmd` ngoài `ALLOWED_APPS`, URL `file://`) thì ném `ValueError`. Khi đó ghi `preset[id] = {"error": ...}` và **không hỏi**.
  2. Hợp lệ thì `approved = interrupt({"question": câu_hỏi})`. Từ chối thì ghi `preset[id] = {"status": "cancelled"}`.

  Node này **không có tác dụng phụ**. Đây là yêu cầu bắt buộc: khi resume, LangGraph chạy lại node từ đầu và trả lần lượt các giá trị resume cho từng `interrupt()`.
- `tools`: node tự viết, không dùng `ToolNode`. Chạy **tuần tự** từng tool call trong `AIMessage` cuối:
  - Có trong `preset` thì dùng kết quả đó, không gọi hàm.
  - Còn lại thì gọi hàm. Tên lạ, sai tham số, hoặc hàm ném `ValueError` thì kết quả là `{"error": "..."}`.
  - In `[tool] name(args) -> result`. Trả một `ToolMessage(content=json.dumps(result), tool_call_id=id)` cho **mọi** tool call, vì API bắt buộc mỗi tool call có đúng một tool message. Xóa `preset` sau khi dùng.
  - Xong thì quay lại `llm`.

  Xác nhận nằm ở node riêng chứ không nằm trong tool, vì tool nằm sau `interrupt()` sẽ bị chạy lại khi resume. Một `AIMessage` có 2 tool hành động thì cái đầu sẽ chạy 2 lần (ví dụ mở Notepad 2 lần).

**Bộ nhớ:** `InMemorySaver` làm checkpointer, `thread_id` là thời gian bắt đầu phiên. Lịch sử hội thoại nằm trong checkpointer, không tự giữ list `messages`. `interrupt()` bắt buộc phải có checkpointer.

**`handle_turn(text, graph, config, out, ask_yes_no, log)`** trong vòng chính:
1. `result = graph.invoke({"messages": [HumanMessage(text)]}, config)`.
2. Khi `result` còn `"__interrupt__"`: lấy `question`, gọi `out(question)`, rồi `ok = ask_yes_no()` (đọc `y`/`n` từ bàn phím). Sau đó `result = graph.invoke(Command(resume=ok), config)`.
3. Trả lời là `result["messages"][-1].content`. In `Agent: …` bằng `out` rồi ghi transcript.

**Giới hạn vòng:** `config["recursion_limit"] = 16`, khoảng 5 vòng llm → confirm → tools. Vượt giới hạn thì LangGraph ném `GraphRecursionError`. `handle_turn` bắt lỗi này và in "Mình chưa làm được việc này", để tránh lặp vô hạn và tốn tiền.

**Lỗi API** (mạng, 401 sai key, 402 hết số dư, 429, 5xx, timeout): `handle_turn` bắt lỗi, in thông báo rõ, lượt đó không có câu trả lời, phiên vẫn chạy. Lỗi xảy ra trong node `llm` nên checkpoint chỉ còn câu user chưa có trả lời. Lượt sau thêm câu user mới phía sau. API chấp nhận 2 câu user liền nhau, nên không cần dọn lịch sử.

**Tool:** hàm Python thường có type hint và docstring kiểu Google (có mục `Args:` để LangChain lấy mô tả tham số). Mỗi tool trả dict. Tool DSP tạo bằng `make_dsp_tools(pipeline)` (closure trên Pipeline). Lỗi tham số trả `{"error": "..."}` để LLM tự sửa, không làm sập chương trình.

`build_graph(model, dsp_tools, action_tools, checkpointer)` nhận model đã `bind_tools`. Test truyền vào model giả là một object có `.invoke(messages) -> AIMessage` trả theo kịch bản. Không gọi DeepSeek thật.

**Tool DSP** (không cần xác nhận, chỉ đổi tham số trong chương trình):

| Tool | Tác dụng |
|---|---|
| `get_status()` | Trả Params hiện tại + Meters |
| `calibrate_noise(seconds: float = 1.0)` | Đo lại phổ ồn nền |
| `set_bandpass(enabled: bool, low_hz: float, high_hz: float)` | Bật/tắt, chỉnh dải |
| `set_spectral_subtraction(enabled: bool, alpha: float, beta: float, smoothing: float)` | Bật/tắt, chỉnh độ mạnh và độ mượt gain |
| `set_noise_estimator(mode: str)` | `"calibrate"` hoặc `"minstat"` |
| `set_noise_gate(enabled: bool, threshold_db: float)` | Bật/tắt, chỉnh ngưỡng |
| `set_bypass(enabled: bool)` | Tắt khử ồn để so sánh |
| `set_monitor(enabled: bool)` | Bật/tắt nghe lại giọng mình qua tai nghe |

**Tool hành động ngoài** (luôn hỏi xác nhận trước khi làm):

| Tool | Tác dụng | Giới hạn an toàn |
|---|---|---|
| `open_app(name: str)` | Mở ứng dụng Windows | Chỉ tên có trong `ALLOWED_APPS` (dict tên → đường dẫn exe, khai báo trong `actions.py`; mặc định `notepad`, `calculator`, `paint`). Chạy bằng `subprocess.Popen([exe])`, không qua shell. |
| `open_url(url: str)` | Mở trang web | Chỉ `http://` hoặc `https://`. Mở bằng `webbrowser.open`. |
| `search_web(query: str)` | Tìm Google | Ghép URL tìm kiếm có mã hóa `urllib.parse.quote_plus`, rồi mở như `open_url`. |
| `save_note(title: str, content: str)` | Ghi ghi chú/tóm tắt ra file | Chỉ ghi vào `notes/`. Tên file `<ngày>_<slug>.md`, slug chỉ giữ chữ, số, gạch ngang (chặn `../`). Trùng tên thì thêm số, không ghi đè. |

**Xác nhận:** làm bằng `interrupt()` ở node `confirm` (xem trên). Tool hành động trong `actions.py` **không** nhận `confirm`, và không import LangGraph. `actions.py` có:
- `describe_action(name, args) -> str`: kiểm tra hợp lệ và trả câu hỏi xác nhận, sai thì ném `ValueError`.
- `ACTION_TOOLS`: list các tool hành động.

Mỗi tool hành động vẫn tự kiểm tra lại allowlist/URL/đường dẫn trước khi làm, phòng trường hợp bị gọi mà không qua `confirm`. Bạn từ chối thì tool không được gọi, và LLM nhận `{"status": "cancelled"}`.

**Hỏi đáp chung và tóm tắt:** không cần tool. Agent trả lời từ kiến thức của model và lịch sử hội thoại. "Tóm tắt rồi lưu lại" thì agent tự tóm tắt, sau đó gọi `save_note`. Agent không truy cập internet để trả lời. `search_web` chỉ mở trình duyệt cho bạn xem.

System prompt ngắn:
- Bạn là trợ lý giọng nói tiếng Việt. Câu trả lời hiện trên terminal, nên giữ ngắn (1–3 câu), không dùng markdown hay danh sách.
- Giải thích từng bộ lọc và miền tham số an toàn. Gọi `get_status` trước khi chỉnh bộ lọc.
- Người dùng than "rè kim loại / lách tách": tăng `smoothing` hoặc giảm `alpha`. Than "ồn thay đổi" hoặc không muốn im lặng để hiệu chuẩn: chuyển sang `minstat`.
- Câu nghe không rõ hoặc mơ hồ thì hỏi lại, không đoán rồi hành động.

Ví dụ phiên:
```
[Enter] 🎙 đang nghe… [Enter]
Bạn: mở notepad giúp mình
Agent: Mình sẽ mở Notepad, đồng ý không? (y/n) y
[tool] open_app(name="notepad") -> {"status": "ok"}
Agent: Đã mở Notepad.

[Enter] 🎙 đang nghe… [Enter]
Bạn: còn nghe tiếng quạt rõ quá
[tool] get_status() -> noise_floor_db=-52, alpha=2.0
[tool] set_spectral_subtraction(enabled=True, alpha=3.0, beta=0.02, smoothing=0.5)
Agent: Mình tăng mức trừ ồn từ 2 lên 3. Nếu giọng nghe rè thì báo mình giảm lại.
```

Ngoài phạm vi: GUI, khử ồn bằng deep learning (Whisper chỉ dùng cho STT), khử vang, nhiều kênh, VAD/tự phát hiện lượt nói, agent nói (TTS, kể cả offline hay qua API), streaming câu trả lời, chế độ thinking của DeepSeek, STT trong graph, nhiều agent/subgraph, checkpointer lưu đĩa (SQLite) hay nhớ qua nhiều phiên, LangSmith/LangGraph Studio, Gmail/Calendar/OAuth, chạy lệnh shell tùy ý, agent tự truy cập web để trả lời, đóng gói phân phối.

## Tech Stack
- Python 3.13
- numpy, scipy: DSP, ghi WAV (đã cài)
- sounddevice: micro/loa (đã cài)
- langgraph: StateGraph, `interrupt`, `Command`, `InMemorySaver` (cần cài, bản 1.x; ghim phiên bản lúc cài)
- langchain-deepseek: `ChatDeepSeek`, kéo theo `langchain-core` và `openai` (cần cài)
- faster-whisper: STT local, backend mặc định (cần cài; kéo theo ctranslate2, av)
- google-genai: Gemini STT, backend tùy chọn (đã cài)
- pystoi: chỉ số STOI, chỉ dùng trong script đánh giá (đã cài)
- pytest (đã cài)
- Thư viện chuẩn: `webbrowser`, `subprocess`, `urllib.parse`, `io`, `wave`

## Commands
```
Cài:            pip install -r requirements.txt
Xem thiết bị:   python -m voiceagent --list-devices
Chạy đầy đủ:    python -m voiceagent [--input 1] [--output 3] [--monitor] [--stt whisper|gemini]
Chỉ DSP:        python -m voiceagent --no-agent      (không cần API key, không nạp Whisper, monitor bật sẵn)
Biến môi trường: DEEPSEEK_API_KEY (bắt buộc cho chat), DEEPSEEK_MODEL, WHISPER_MODEL,
                 GEMINI_API_KEY + GEMINI_STT_MODEL (chỉ khi --stt gemini)
Đánh giá:       python scripts/evaluate.py --clean data/speech.wav --noise data/fan.wav --snr 0 5 10
Test:           python -m pytest -q
```
Trong phiên: Enter trên dòng trống để bắt đầu/dừng nói; gõ chữ + Enter để gửi chữ; `quit` để thoát (ghi WAV + transcript).

Dữ liệu đánh giá do bạn tự thu, dạng WAV mono 16 kHz:
- `data/speech.wav`: 20–30 s giọng nói, phòng yên tĩnh, micro gần miệng.
- `data/fan.wav`: ≥ 30 s chỉ có tiếng quạt, không có giọng.

Có thể thu bằng `python -m voiceagent --no-agent` rồi lấy file `_raw.wav`. File sai định dạng sẽ bị script báo lỗi rõ ràng, script không tự resample.

`evaluate.py` trộn giọng với ồn ở từng mức SNR, rồi chạy **cùng `Pipeline` như lúc realtime**, đưa vào từng khối 256 mẫu. Trước khi so với giọng sạch, script bù độ trễ 256 mẫu. Kết quả in ra một bảng: mỗi dòng là một cặp (mức SNR, cấu hình), mỗi cột là SNR và STOI trước/sau. Các cấu hình gồm:
1. Không xử lý
2. Chỉ SS `calibrate`, `smoothing=0`
3. Chỉ SS `calibrate`, `smoothing=0.5`
4. Chỉ SS `minstat`, `smoothing=0.5`
5. Cả chuỗi (band-pass + SS `minstat` + gate)

Với cấu hình `calibrate`, script hiệu chuẩn trên 1 s ồn thuần đặt trước đoạn giọng.

## Project Structure
```
voiceagent/
  __init__.py
  __main__.py   argparse, khởi động stream, vòng hội thoại, dọn dẹp khi thoát
  dsp.py        BandPass, SpectralSubtractor, NoiseGate, rms_db, snr_db, add_noise, mix_at_snr
  pipeline.py   Params, Meters, Pipeline (không import sounddevice)
  audio.py      sounddevice Stream, monitor, thu câu nói, ghi WAV
  voice.py      make_transcriber (whisper | gemini), ghi transcript
  actions.py    ALLOWED_APPS, open_app, open_url, search_web, save_note, describe_action, ACTION_TOOLS (không import LangGraph)
  agent.py      make_dsp_tools, SYSTEM_PROMPT, AgentState, node llm/confirm/tools, build_graph, handle_turn
scripts/
  evaluate.py   đánh giá offline SNR + STOI (import pystoi chỉ ở đây)
data/           WAV bạn tự thu cho đánh giá (không commit)
notes/          ghi chú agent lưu (không commit)
out/            WAV + transcript sinh ra (không commit)
tests/
  test_dsp.py, test_pipeline.py, test_audio.py, test_voice.py,
  test_actions.py, test_agent.py, test_evaluate.py
requirements.txt
```
`dsp.py`, `pipeline.py`, `actions.py` và phần lõi `agent.py` không phụ thuộc phần cứng hay mạng. Client DeepSeek, model Whisper, Gemini, `subprocess` và `webbrowser` được truyền vào dưới dạng tham số, nên test thay được bằng hàm giả.

## Code Style
```python
class BandPass:
    """Butterworth band-pass, causal, keeps filter state across blocks."""

    def __init__(self, sr: int, low_hz: float, high_hz: float, order: int = 4):
        if not 0 < low_hz < high_hz < sr / 2:
            raise ValueError(f"need 0 < low < high < {sr / 2} Hz, got {low_hz}, {high_hz}")
        self.sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2))

    def process(self, block: np.ndarray) -> np.ndarray:
        out, self.zi = signal.sosfilt(self.sos, block, zi=self.zi)
        return out
```
- snake_case, type hint cho hàm public, docstring một dòng.
- Bộ lọc có state là class có `process(block) -> block`. Hàm không state là hàm thường.
- Input sai ném `ValueError` với thông điệp rõ ràng. Tool agent bắt lỗi đó và trả `{"error": ...}`.
- Phụ thuộc bên ngoài (DeepSeek, STT, mở app) truyền vào dưới dạng tham số hàm, không gọi trực tiếp sâu bên trong logic.
- Comment giải thích *vì sao* (ý tưởng DSP), không lặp lại code. Đây là dự án học, nên mỗi bộ lọc có một comment ngắn nêu nguyên lý.

## Testing Strategy
pytest, tín hiệu tổng hợp (sine + nhiễu trắng có seed). Không cần micro, mạng, API key hay model Whisper.

- **dsp**
  - BandPass: 1000 Hz giữ trong 1 dB; 50 Hz và 6000 Hz giảm ≥ 20 dB. Xử lý theo khối cho kết quả giống xử lý cả tín hiệu một lần (sai số < 1e-6). Dải sai ném `ValueError`. *(đã có)*
  - SpectralSubtractor:
    - Tắt trừ ồn (`alpha=0`) thì đầu ra bằng đầu vào trễ 256 mẫu (kiểm tra overlap-add tái tạo đúng).
    - Sine 440 Hz + nhiễu trắng SNR 5 dB, chế độ `calibrate` trên 1 s nhiễu thuần: SNR đầu ra tăng ≥ 5 dB.
    - Làm mượt gain: trên đoạn chỉ có nhiễu trắng, độ lệch chuẩn theo thời gian của gain với `smoothing=0.5` thấp hơn `smoothing=0`.
  - Minimum statistics:
    - Nhiễu tăng +6 dB ở giây thứ 3: trong vòng 2 s sau đó, ước lượng ồn bám theo mức mới, sai số ≤ 3 dB.
    - Sine 1000 Hz bật/tắt mỗi 0.5 s: ước lượng ồn ở bin 1000 Hz nằm trong 3 dB của mức nhiễu thật.
    - Sine bật liên tục 3 s: ước lượng ở bin 1000 Hz tăng lên (ghi lại giới hạn đã biết).
  - NoiseGate: khối nhiễu -60 dBFS giảm ≥ 15 dB sau khi gain ổn định; khối sine -20 dBFS giữ trong 1 dB.
  - `snr_db` / `add_noise` / `mix_at_snr`: đạt SNR mục tiêu trong 0.1 dB. *(đã có)*
- **pipeline**: đổi Params có hiệu lực ở khối kế tiếp; `bypass` trả đúng đầu vào; Meters cập nhật; setter ném `ValueError` khi ngoài miền; < 2 ms/khối. *(đã có)*
- **audio**:
  - Monitor tắt: tai nghe nhận khối 0. Monitor bật: tai nghe nhận khối đã xử lý.
  - Chỉ những khối nằm giữa bắt đầu và kết thúc lượt mới vào bộ đệm câu nói. Bộ đệm chứa khối đã khử ồn.
  - Đóng gói WAV trong RAM rồi đọc lại: đúng 16 kHz, mono, đúng độ dài. Câu < 0.3 s trả `None`.
  - Ghi WAV phiên, CLI liệt kê thiết bị, lỗi thiết bị báo rõ. *(đã có)*
- **voice**:
  - Transcript ghi đúng định dạng 2 dòng mỗi lượt.
  - STT với model Whisper giả: segment `" xin"`, `" chào"` cho ra `"xin chào"`. Không có segment thì trả `""`.
  - `make_transcriber` (Whisper và Gemini đều là hàm giả): `"gemini"` có key thì dùng Gemini. `"gemini"` thiếu key thì dùng Whisper, thông báo in đúng một lần. `"whisper"` thì không bao giờ gọi Gemini.
- **actions** (`subprocess.Popen` và `webbrowser.open` được thay bằng hàm giả qua `monkeypatch`):
  - `open_app`: tên ngoài `ALLOWED_APPS` thì trả lỗi và không gọi gì. Hợp lệ thì gọi đúng exe.
  - `describe_action`: `open_app("cmd")` và `open_url("file:///C:/")` ném `ValueError`. `open_app("notepad")` trả câu hỏi có chữ "Notepad".
  - `open_url`: từ chối `file://`, `javascript:`, chuỗi không phải URL.
  - `search_web`: query có dấu cách và ký tự tiếng Việt được mã hóa đúng.
  - `save_note`: ghi vào thư mục tạm; title `../../evil` vẫn nằm trong `notes/`; trùng tên không ghi đè.
- **agent**:
  - Tool DSP gọi với một Pipeline thật: hợp lệ thì đổi Params; sai thì trả `{"error": ...}` và Params giữ nguyên.
  - Schema: `convert_to_openai_tool(set_bandpass)` có 3 tham số bắt buộc, và mỗi tham số có mô tả lấy từ docstring.
  - Graph với model giả (trả `AIMessage` theo kịch bản), `InMemorySaver`, Pipeline thật, `Popen`/`webbrowser.open` giả:
    - Không có tool call: kết thúc ở `END`, trả đúng câu trả lời.
    - `set_bypass(enabled=True)`: không có interrupt. Params đổi. `ToolMessage` có đúng `tool_call_id`. Model được gọi 2 lần.
    - `open_app("notepad")`: `invoke` trả `__interrupt__` có câu hỏi. Resume `True` thì `Popen` được gọi **đúng 1 lần**. Resume `False` thì `Popen` không được gọi, và `ToolMessage` chứa `cancelled`.
    - `open_app("cmd")`: không có interrupt, `ToolMessage` chứa `error`, `Popen` không được gọi.
    - Một `AIMessage` có 2 tool hành động: có 2 interrupt. Đồng ý cái đầu, từ chối cái sau: chỉ tool đầu chạy, và chạy đúng 1 lần.
    - Tên tool lạ hoặc thiếu tham số: `ToolMessage` chứa `error`, graph không ném lỗi.
    - Model luôn trả tool call: `handle_turn` bắt `GraphRecursionError` và in câu dự phòng.
    - Hai lượt cùng `thread_id`: lượt 2 thấy message của lượt 1.
  - `handle_turn` với transcribe/graph/out/ask_yes_no giả: STT trả rỗng thì không gọi chat; có chữ thì chat rồi in đúng câu trả lời và ghi transcript; chat ném lỗi mạng thì báo lỗi, phiên vẫn chạy.
- **evaluate**: với cấu hình "không xử lý", STOI sau bằng STOI trước (sai số < 0.01), chứng minh bù trễ đúng. Bỏ qua khi thiếu pystoi.
- **Kiểm tra thủ công** (không tự động): xem Success Criteria 4–6.

## Boundaries
- Always: viết test trước (TDD); chạy `python -m pytest -q` trước khi báo xong một task; giữ `dsp.py`/`pipeline.py` không có I/O, mạng, print; giữ audio callback không chặn (không lock lâu, không gọi mạng, không STT); hỏi xác nhận trước mọi hành động ngoài.
- Ask first: thêm dependency ngoài danh sách trên; đổi thứ tự chuỗi lọc; đổi sample rate hay blocksize; thêm tool mới cho agent; thêm app vào `ALLOWED_APPS`; đổi model mặc định (DeepSeek, Whisper); bật thinking mode của DeepSeek.
- Never: commit API key, file WAV, `data/`, `notes/`, `out/`, model Whisper đã tải; dùng SNR làm thước đo duy nhất (phải xem kèm STOI); gọi DeepSeek, Gemini, nạp model Whisper thật, mở app hay mở trình duyệt thật trong test; chạy lệnh qua shell hay chạy exe ngoài `ALLOWED_APPS`; ghi file ngoài `notes/` và `out/`; tự ý clip hay resample mà không báo.

## Success Criteria
1. `python -m pytest -q` qua toàn bộ test ở trên, không cần micro hay mạng.
2. `--no-agent` chạy ≥ 2 phút không underrun (Meters báo 0), xử lý trung bình < 2 ms/khối.
3. Với nhiễu quạt thật, bật cả 3 bộ lọc nghe rõ giảm ồn khi im lặng; giọng vẫn hiểu được. Thoát thì ghi `_raw.wav`, `_clean.wav` và `_transcript.md`.
4. Hội thoại giọng nói (thủ công, bật quạt). Mỗi câu nói được nhận dạng đúng ý và agent in câu trả lời đúng:
   - "Thủ đô của Pháp là gì?" → trả lời đúng.
   - "Còn ồn quá" / "Cho mình nghe lại giọng mình" → gọi đúng tool DSP / `set_monitor`.
   - "Mở Notepad" → hỏi xác nhận; `y` mới mở, `n` thì không.
   - "Tìm trên Google thời tiết Hà Nội" → hỏi xác nhận rồi mở trình duyệt.
   - "Tóm tắt cuộc nói chuyện rồi lưu lại" → hỏi xác nhận rồi tạo file trong `notes/`.
   - "Mở cmd" (không có trong danh sách) → agent từ chối, không chạy gì.
5. Các lỗi sau đều in thông báo rõ ràng, và phiên vẫn chạy được ở mức tối đa có thể:
   - Thiếu `DEEPSEEK_API_KEY`: vẫn chạy DSP và STT, in transcript, nhưng không có câu trả lời.
   - DeepSeek lỗi mạng, hết số dư (402) hoặc 429: bỏ lượt đó, lượt sau thử lại.
   - `--stt gemini` mà thiếu `GEMINI_API_KEY`: chuyển sang Whisper.
   - Sai số thiết bị.
   - Whisper chưa tải được model (lần đầu không có mạng): báo lỗi, vẫn gõ chữ được.
6. Với cùng một câu nói đã khử ồn, `--stt whisper` (`small`, CPU) trả chữ trong ≤ 5 s cho câu 5 s. Đây là kiểm tra thủ công, có ghi lại thời gian thực đo.
7. `quit` hoặc Ctrl+C: dừng stream, ghi WAV + transcript, thoát sạch không traceback.
8. `scripts/evaluate.py` in bảng đủ 5 cấu hình × 3 mức SNR. Với ồn quạt ở 5 dB, cấu hình 5 (cả chuỗi) phải đạt cả hai điều kiện:
   - SNR tăng ≥ 5 dB.
   - STOI giảm không quá 0.02 so với không xử lý.

   Bảng kết quả được dán vào README. Lưu ý từ paper: SS một kênh thường *không* làm STOI tăng, và có thể làm nhận dạng giọng nói kém đi.
9. Cấu hình 3 có STOI ≥ cấu hình 2 ở cả 3 mức SNR, tức làm mượt gain không gây hại.

## Rủi ro
| Rủi ro | Mức | Cách xử lý |
|---|---|---|
| Whisper `small` nhận dạng tiếng Việt kém (sai dấu, sai từ) | Med | Thử ngay với giọng thật ở task STT. Kém thì đổi `WHISPER_MODEL=medium`, hoặc dùng `--stt gemini`. |
| `faster-whisper` cài lỗi trên Python 3.13 / Windows (wheel `ctranslate2`, `av`) | Med | Cài và chạy thử ở đầu task STT, trước khi viết code. Lỗi thì báo bạn trước khi đổi hướng. |
| Tên model / tool calling của DeepSeek khác hiểu biết hiện tại (nguồn ngoài nói `deepseek-v4-flash` có thể đã trỏ sang V4.1 Flash) | Med | Task chat đọc docs chính thức `api-docs.deepseek.com` trước khi code, rồi gọi thử 1 tool. |
| API LangGraph đổi giữa các bản (`create_react_agent` đã chuyển sang `langchain.agents`; import `InMemorySaver`, `interrupt`, `Command` có thể đổi chỗ) | Med | Đầu task agent đọc docs chính thức LangGraph đúng bản đã cài, rồi ghim phiên bản trong `requirements.txt`. |
| `ChatDeepSeek` + tool calling có lỗi riêng (ví dụ `reasoning_content`, `tool_choice`), chưa tìm thấy tài liệu | Med | Spike: gọi thật 1 tool với 1 câu tiếng Việt trước khi viết graph. Lỗi thì thử `ChatOpenAI(base_url=...)` (cần hỏi bạn, vì là dependency khác). |
| Nhầm lẫn khi resume `interrupt()`: node chạy lại từ đầu | Med | Node `confirm` không có tác dụng phụ. Có test 2 tool hành động trong một `AIMessage` để bắt lỗi chạy 2 lần. |
| SS làm STT nhận dạng *kém* đi (paper ghi nhận SS làm tăng WER) | Med | Tool `set_bypass` cho phép so sánh ngay trong phiên. Open Question 5 đề xuất đo thêm, giờ có thể đo trên cả Whisper và Gemini. |
| Tổng độ trễ một lượt (STT + chat) vài giây | Med | Chấp nhận cho mục đích học. In "đang nghe chữ…" và "đang nghĩ…" để biết chương trình không bị treo. |
| Nhận dạng sai dẫn tới hành động sai | Med | Luôn xác nhận y/n bằng bàn phím, có danh sách cho phép, không có shell. |

## Open Questions
1. Model DeepSeek: bạn nhắc tới `deepseek-v4.1-flash-1017`. Trên API chính thức của DeepSeek, model này có thể chỉ gọi được qua alias `deepseek-v4-flash`. Mặc định dùng alias đó, và chốt tên ở task chat sau khi đọc docs. Bạn có cần ghim đúng phiên bản 1017 không? Nếu cần, có thể phải gọi qua nhà cung cấp khác (Cloudflare, Google Cloud).
2. ~~Độ trễ micro → tai nghe khi bật monitor (thường 50–150 ms trên Windows) có chấp nhận được không?~~ Đã trả lời 2026-11-02: chấp nhận được, theo cảm nhận khi nghe thử `--no-agent`. Chưa đo số ms.
3. Có cần báo cáo/đồ thị (spectrogram trước/sau) cho bài tập không? Spec hiện không có `matplotlib`.
4. Ngưỡng "STOI giảm ≤ 0.02" và các hằng số minstat (0.85, 96 khung, bias 1.5) là giá trị khởi đầu tôi chọn, không lấy từ paper. Có thể cần chỉnh sau khi chạy `evaluate.py` lần đầu.
5. Muốn đo luôn ảnh hưởng của khử ồn lên nhận dạng giọng nói không? Cách làm: chạy STT trên giọng ồn và giọng đã khử ồn, rồi tính WER so với câu đọc mẫu. Đây đúng là thí nghiệm của paper. Có 2 backend thì so được cả Whisper với Gemini. Với Whisper thì chạy offline, không tốn tiền. Việc này cần thêm script.
6. Danh sách `ALLOWED_APPS` mặc định (`notepad`, `calculator`, `paint`) có đủ không? Bạn muốn thêm app nào, kèm đường dẫn exe?
7. ~~Model Gemini cho STT~~ — chốt 2026-11-11: `gemini-3.5-flash-lite` (Stable, $0.30/1M token audio). Có `gemini-3.5-transcribe` chuyên STT nhưng đắt hơn ($2.00/1M).
