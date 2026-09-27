# BTVN#3: Hệ Thống Flight Booking Agent với LangGraph & LangChain

Hệ thống tác tử thông minh (Agentic System) hỗ trợ nghiệp vụ tra cứu và đặt vé máy bay, được triển khai trên nền tảng LangGraph và LangChain. Dự án hiện thực hóa các nguyên lý cốt lõi của Harness Engineering: **Ràng buộc dữ liệu (Data Constraints)**, **Tiêu chí hoàn thành độc lập bằng Code (Deterministic Verification)**, **Kiểm soát quyền truy cập theo vai trò (RBAC)**, **Phát hiện vòng lặp vô ích (Loop Detection)** và **Cơ chế dừng an toàn / Bàn giao cho con người (Safe Failure & Handoff)**.

---

## 1. Cấu Trúc Thư Mục Dự Án

```text
booking_Agent/
├── agents/                         # Triển khai 3 mẫu kiến trúc Agent
│   ├── __init__.py                 # Khởi tạo package agents
│   ├── react_agent.py              # Mô hình phản xạ ReAct tuần tự
│   ├── plan_execute.py             # Mô hình Lập kế hoạch rồi Thực thi (Plan-then-Execute)
│   └── hybrid_agent.py             # Mô hình Lai (Hybrid / Substep ReAct)
├── lib/                            # Các module nền tảng và khung an toàn (Harness)
│   ├── __init__.py                 # Khởi tạo package lib
│   ├── state.py                    # Định nghĩa TypedDict AgentState & Pydantic TicketBooking
│   ├── tools_booking.py            # Cơ sở dữ liệu mô phỏng & công cụ tra cứu, đặt vé
│   ├── guards.py                   # Rào chắn an toàn (Check quyền, Loop detector, Verify code)
│   ├── handoff.py                  # Đóng gói ngữ cảnh bàn giao có cấu trúc (Handoff payload)
│   └── model.py                    # Quản lý LLM Provider (Hỗ trợ MockLLM offline & Gemini API)
├── benchmark.py                    # Script đánh giá hiệu năng tự động trên 6 kịch bản
├── main.py                         # Điểm chạy tương tác dòng lệnh (CLI trace)
├── requirements.txt                # Danh sách các thư viện phụ thuộc
└── README.md                       # Tài liệu hướng dẫn dự án
```

---

## 2. Các Đặc Tính Kỹ Thuật Nổi Bật (Lớp Harness)

1. **Ràng buộc dữ liệu (Constraints as Data):** Dữ liệu nghiệp vụ được chuẩn hóa qua schema `TicketBooking` và hàm `validate_booking_data()`, kiểm soát định dạng mã sân bay (3 ký tự), tính hợp lệ của ngày bay (không đặt ngày trong quá khứ) trước khi cho phép gọi công cụ.
2. **Kiểm soát quyền hạn (RBAC):** Hàm `check_permission()` phân tầng người dùng rõ rệt: tài khoản cấp `guest` chỉ được phép tra cứu (`search_flights`); chỉ tài khoản `member` hoặc `admin` mới được kích hoạt thao tác đặt vé (`book_ticket`).
3. **Phát hiện vòng lặp (Loop Detection):** Lớp `LoopDetector` sử dụng cơ chế cửa sổ trượt (Sliding Window) để lưu vết bộ đôi `(tool_name, args)`. Nếu phát hiện tần suất lặp vượt ngưỡng `repeat_k`, hệ thống lập tức ngắt chu trình để tránh lãng phí tài nguyên token.
4. **Tiêu chí hoàn thành kiểm bằng code (Deterministic Verification):** Thay vì tin tưởng nội dung phản hồi dạng chuỗi văn bản của LLM, hàm `verify_completion_code()` trực tiếp xác thực trạng thái `booking_info` trong state, yêu cầu phải có `booking_id` hợp lệ và trạng thái `confirmed`.
5. **Cơ chế dừng an toàn & Bàn giao (Handoff Payload):** Khi gặp ngoại lệ nghiệp vụ, lỗi phân quyền hoặc vi phạm dữ liệu, agent tạo cấu trúc payload chuẩn hóa gồm: lý do dừng (`stop_reason`), các bước đã thử (`da_thu`), ảnh chụp trạng thái (`trang_thai`) và câu hỏi cần giải quyết cho con người (`cau_hoi_cho_nguoi`).
6. **Môi trường kép (Dual Mode):** Hỗ trợ chuyển đổi linh hoạt giữa chế độ Offline (`MockLLM` - kiểm thử logic và benchmark tức thì mà không cần mạng) và chế độ trực tuyến (`Gemini API` qua model `gemini-2.5-flash`).

---

## 3. Cài Đặt & Cấu Hình

### Yêu cầu tiên quyết
- Python 3.10 trở lên.

### Các bước cài đặt
1. **Khởi tạo và kích hoạt môi trường ảo:**
   ```bash
   # Trên Windows
   python -m venv .venv
   .\.venv\Scripts\activate

   # Trên Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Cài đặt các gói phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Cấu hình môi trường (Tùy chọn):**
   - **Mặc định:** Hệ thống tự động kích hoạt `MockLLM` ở chế độ offline.
   - **Chạy trực tuyến với Google Gemini API:** Tạo file `.env` tại thư mục gốc của dự án với nội dung:
     ```env
     BOOKING_LLM_MODE=api
     GEMINI_API_KEY=AIzaSy...your_gemini_api_key...
     ```

---

## 4. Hướng Dẫn Sử Dụng

### 4.1. Chạy tương tác CLI (`main.py`)
Xem trực tiếp vết suy luận (trace), chuỗi gọi tool và quyết định của các router rào chắn:

- **Chạy ReAct Agent với quyền Member đặt vé thành công:**
   ```bash
   python main.py --agent react --role member
   ```

- **Kiểm tra cơ chế chặn quyền người dùng Guest (kích hoạt Handoff):**
   ```bash
   python main.py --agent react --role guest
   ```

- **Chạy Plan-then-Execute Agent:**
   ```bash
   python main.py --agent plan --role member
   ```

- **Chạy Hybrid Agent kiểm tra kịch bản hết vé:**
   ```bash
   python main.py --agent hybrid --role member --query "Đặt vé từ SGN đến DAD ngày 2026-10-15"
   ```

---

### 4.2. Chạy Benchmark so sánh hiệu năng (`benchmark.py`)
Tự động kiểm thử và đo lường 6 kịch bản thực tế trên cả 3 mô hình thiết kế:
```bash
python benchmark.py
```

Danh sách 6 test case kiểm thử:
- **TC1:** Đặt vé thành công (Quyền member, thông tin chuyến hợp lệ).
- **TC2:** Chặn quyền Guest (Tài khoản guest cố gắng đặt vé).
- **TC3:** Chuyến bay hết vé (Tuyến bay hết ghế trống).
- **TC4:** Không tìm thấy chuyến (Tuyến bay không tồn tại trong hệ thống).
- **TC5:** Dữ liệu sai ngày (Ngày bay nằm trong quá khứ).
- **TC6:** Yêu cầu vòng lặp (Yêu cầu tìm kiếm lặp đi lặp lại liên tục).

---

## 5. Kết Quả Đo Đạc Thực Nghiệm (Benchmark)

Bảng tổng hợp thu thập từ lần đo thực nghiệm tự động với 6 kịch bản kiểm thử:

| Mẫu thiết kế | Tỷ lệ thành công | Tỷ lệ bàn giao (Handoff) | Thời gian thực thi TB |
| :--- | :---: | :---: | :---: |
| **ReAct** | **16.7%** (1/6) | **83.3%** (5/6) | **0.006s** |
| **Plan-Execute** | **16.7%** (1/6) | **50.0%** (3/6) | **0.003s** |
| **Hybrid** | **16.7%** (1/6) | **83.3%** (5/6) | **0.002s** |

### Chi tiết các ca kiểm thử:
- **ReAct Agent:**
  - **TC1:** Hoàn thành: `True` | Bàn giao: `False` (5 bước, 0.017s)
  - **TC2 - TC6:** Hoàn thành: `False` | Bàn giao: `True` (Chặn quyền Guest tại 3 bước; phát hiện hết chỗ/không tìm thấy chuyến tại 5-7 bước; chặn sai ngày ngay bước 1; bắt lặp tại 5 bước).
- **Plan-Execute Agent:**
  - **TC1:** Hoàn thành: `True` | Bàn giao: `False` (2 bước, 0.004s)
  - **TC2, TC5, TC6:** Bàn giao: `True` (Chặn quyền, sai ngày, bắt lặp kế hoạch thành công).
  - **TC3, TC4:** Hoàn thành: `False` | Bàn giao: `False` (Do kế hoạch được lập tĩnh ban đầu gồm `[search, book]`, khi bước search báo hết vé/không tìm thấy, chu trình không tự động rẽ nhánh sang Handoff mà chỉ dừng lại mà không tạo vé).
- **Hybrid Agent:**
  - **TC1:** Hoàn thành: `True` | Bàn giao: `False` (2 bước, 0.003s)
  - **TC2 - TC6:** Hoàn thành: `False` | Bàn giao: `True` (Bắt lỗi và kích hoạt handoff triệt để ở tất cả các tình huống bất thường: chặn hết vé/không tìm thấy ngay tại bước 1, chặn quyền tại bước 2, chặn sai ngày tại bước 1, bắt lặp tại bước 3).

### Nhận xét & Đánh giá hiệu quả kiến trúc:
1. **Tính chính xác và An toàn (16.7% Success Rate):**
   - Tập dữ liệu gồm 1 ca hợp lệ (TC1) và 5 ca bất thường/vi phạm quy tắc (TC2–TC6). Do đó, tỷ lệ hoàn thành chính xác 1/6 (16.7%) trên cả 3 agent chứng minh rằng hàm kiểm tra hoàn thành bằng code (`verify_completion_code`) hoạt động tuyệt đối chính xác: **không có bất kỳ ca vi phạm nào bị gắn nhãn hoàn thành sai**.
2. **Khả năng phản ứng & Bàn giao (Handoff Rate):**
   - **Hybrid Agent (83.3% handoff, 0.002s TB):** Đạt hiệu năng tối ưu nhất. Nhờ kế hoạch sơ bộ nhưng từng substep lại có cơ chế rẽ nhánh phản xạ, Hybrid Agent vừa tối ưu số bước xử lý (chỉ 1–3 bước mỗi case) vừa kích hoạt bàn giao an toàn cho toàn bộ 5 ca lỗi.
   - **ReAct Agent (83.3% handoff, 0.006s TB):** An toàn cao tương đương Hybrid nhưng tốn nhiều bước tương tác qua lại hơn (5–7 bước cho các ca tra cứu và hết vé), dẫn đến thời gian thực thi dài hơn.
   - **Plan-Execute Agent (50.0% handoff, 0.003s TB):** Tốc độ nhanh do ít bước, nhưng bộc lộ điểm yếu cố hữu của mô hình lập kế hoạch tĩnh: khi kết quả thực tế của bước tra cứu thay đổi (hết vé, không thấy chuyến), agent không tự thích ứng để kích hoạt bàn giao ngoại lệ.
