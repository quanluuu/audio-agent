# Làm việc chung

Dự án có 2 người: **A** dùng Windows, **B** dùng Ubuntu. Đọc [SPEC.md](SPEC.md) để biết làm gì. Đọc [tasks/todo.md](tasks/todo.md) để biết từng task. File này nói cách làm chung.

## 1. Phân công: chia theo tầng

| | A (Windows): Giọng nói + Agent | B (Ubuntu): DSP + Đánh giá |
|---|---|---|
| Task | 3 (nghe thử), 4, 5, 6, 6b, 14, 13, 15 | 7, 8, 9, 10, 11, 12 |
| File sở hữu | `audio.py`, `voice.py`, `actions.py`, `agent.py`, `__main__.py` + test tương ứng | `dsp.py`, `pipeline.py`, `scripts/evaluate.py` + test tương ứng |
| Vì sao | Mở app Windows, thiết bị âm thanh đều cần Windows | Bộ lọc là numpy thuần, chạy và test được trên mọi máy |

**Thứ tự để hai người không chờ nhau:**
```
A:  4 ─► 5 ─► 6 ─► 6b ─► 14 ────────► 13 ─► 15
B:  7 ─► 8 ─► 9 ─► 10 ─► 11 ─► 12
                      └── 13 cần 10 xong ──┘
```
- Task 13 (tool DSP) chỉ bắt đầu khi Task 10 đã vào `main`.
- Task 5 của A sửa nhẹ `pipeline.py` (thêm `monitor`). A nên làm Task 5 sớm và merge nhanh, để B ít phải gộp xung đột.

**File dùng chung** (`SPEC.md`, `tasks/`, `requirements.txt`, `README.md`, `.gitignore`): ai sửa cũng được, nhưng ghi rõ trong PR.

**Hợp đồng giữa 2 tầng:** mục "Giao diện `Pipeline`" trong SPEC.md. B viết đúng chữ ký đó, A gọi đúng chữ ký đó. Muốn đổi thì mở PR sửa SPEC trước, chờ người kia duyệt.

## 2. Khởi tạo repo (chỉ A làm, một lần)

1. Trên github.com: **New repository** → đặt tên → **Private** → *không* tick thêm README, .gitignore hay license (vì code đã có sẵn) → **Create repository**.
2. Trong thư mục dự án (PowerShell hoặc Git Bash):
   ```
   git config --global user.name "Tên của bạn"
   git config --global user.email "email-dang-ky-github@example.com"
   git init -b main
   git add .
   git status        # kiểm tra không có out/, data/, .venv/, file WAV
   git commit -m "chore: initial project skeleton (phase 1)"
   git remote add origin https://github.com/<tài-khoản>/<tên-repo>.git
   git push -u origin main
   ```
   Lần push đầu, Git for Windows mở trình duyệt để đăng nhập GitHub.
3. Mời B: repo → **Settings** → **Collaborators** → **Add people** → nhập username của B. B nhận lời mời qua email.

Gói GitHub Free không bật được bảo vệ nhánh cho repo private. Quy tắc "không push thẳng lên `main`" vì vậy là quy ước giữa hai người, GitHub không tự chặn.

## 3. Cài đặt máy (mỗi người, một lần)

Cả hai dùng **Python 3.13** và môi trường ảo `.venv` riêng trong thư mục dự án. Không cài thư viện vào Python hệ thống.

**A — Windows (PowerShell):**
```
git clone https://github.com/<tài-khoản>/<tên-repo>.git   # bỏ qua nếu đã ở trong thư mục dự án
cd <tên-repo>
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```
`Activate.ps1` báo lỗi "running scripts is disabled": chạy `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` một lần rồi thử lại.

**B — Ubuntu:**
```
python3 --version                     # cần 3.13; nếu thấp hơn, cài python3.13 từ PPA deadsnakes
sudo apt install git libportaudio2 python3.13-venv   # sounddevice trên Linux cần libportaudio2
ssh-keygen -t ed25519 -C "email-dang-ky-github@example.com"
cat ~/.ssh/id_ed25519.pub             # dán vào GitHub → Settings → SSH and GPG keys → New SSH key
git clone git@github.com:<tài-khoản>/<tên-repo>.git
cd <tên-repo>
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

Mỗi lần mở terminal mới, kích hoạt lại `.venv` trước khi chạy code.

## 4. Quy trình mỗi task

```
git switch main
git pull                                  # lấy code mới nhất
git switch -c task-07-noise-gate          # nhánh mới cho task
# ... TDD: viết test đỏ → code cho xanh → dọn ...
python -m pytest -q                       # phải xanh hết trước khi push
git add voiceagent/dsp.py tests/test_dsp.py tasks/todo.md    # add từng file, tránh `git add .`
git commit -m "feat(dsp): add NoiseGate with attack/release smoothing"
git push -u origin task-07-noise-gate
```
Sau đó lên GitHub → **Compare & pull request** → điền mẫu có sẵn → **Reviewers**: chọn người kia.

**Người duyệt:**
```
git fetch
git switch task-07-noise-gate
python -m pytest -q                       # chạy trên máy mình = kiểm tra luôn trên hệ điều hành kia
```
Xem diff trên GitHub, để comment nếu chưa hiểu hoặc thấy lỗi. Ổn thì **Approve**. Người mở PR bấm **Squash and merge**, rồi **Delete branch**.

**Nhánh bị cũ so với `main`** (người kia vừa merge):
```
git switch main && git pull
git switch task-07-noise-gate
git merge main
```
Có xung đột: mở file bị báo, tìm các dòng `<<<<<<<` / `=======` / `>>>>>>>`, giữ phần đúng, xóa các dòng đánh dấu. Sau đó chạy `python -m pytest -q`, rồi `git add <file>` và `git commit`. Không chắc phần nào đúng thì hỏi người kia, đừng đoán.

## 5. Quy ước

**Nhánh:** `task-<số 2 chữ số>-<mô-tả-ngắn>`, ví dụ `task-06-whisper-stt`. Sửa SPEC: `spec-<mô-tả>`. Một nhánh chỉ làm một task.

**Commit:** [Conventional Commits](https://www.conventionalcommits.org/), tiếng Anh, ngắn, nói *ý định*:
```
feat(dsp): add spectral subtraction with calibrate mode
fix(audio): count output underflow as underrun
test(actions): reject file:// urls
docs(spec): add Pipeline interface contract
chore: pin dependency versions
```
Scope là tên module: `dsp`, `pipeline`, `audio`, `voice`, `actions`, `agent`, `eval`, `spec`.

**Pull Request:** nhỏ, mỗi PR một task, dưới khoảng 300 dòng thay đổi. Test xanh trên máy người mở PR *và* người duyệt. Không ai tự merge PR chưa được duyệt.

**Code** (chi tiết ở mục Code Style của SPEC.md):
- Tên biến, hàm, comment, commit: tiếng Anh. Chữ hiện cho người dùng và tài liệu: tiếng Việt.
- Type hint cho hàm public, docstring một dòng, comment giải thích *vì sao*.
- TDD: test viết trước, nằm trong `tests/test_<module>.py`.
- Thư viện mới: bàn với người kia trước, rồi thêm vào `requirements.txt` kèm phiên bản cố định (`tên==x.y.z`).

**Chạy được trên cả 2 hệ điều hành:**
- Đường dẫn dùng `pathlib.Path`, không viết cứng `\` hay `/`.
- Mở file văn bản luôn ghi `encoding="utf-8"`. Mặc định của Windows không phải UTF-8, tiếng Việt sẽ lỗi.
- Thư viện chỉ có trên Windows (exe trong `ALLOWED_APPS`) chỉ import *bên trong hàm*. Test thay bằng hàm giả. Nhờ vậy `python -m pytest -q` vẫn xanh trên Ubuntu.
- Test không được cần micro, loa, mạng hay API key.
- Line ending do `.gitattributes` lo (LF cho mọi file văn bản), không cần chỉnh gì thêm.

**Bí mật:** `GEMINI_API_KEY` đặt trong biến môi trường hoặc file `.env` (đã nằm trong `.gitignore`). Không bao giờ dán key vào code, commit hay PR. Lỡ commit key thì báo người kia và **tạo key mới ngay**. Xóa commit không đủ, vì key đã lên GitHub.
