# BTVN#3: Hệ Thống Flight Booking Agent với LangGraph & LangChain

Hệ thống tác tử thông minh (Agentic System) hỗ trợ nghiệp vụ tra cứu và đặt vé máy bay, được triển khai trên nền tảng LangGraph và LangChain. Dự án áp dụng các nguyên lý cốt lõi của môn học: Ràng buộc là dữ liệu (Data Constraints), Tiêu chí hoàn thành độc lập bằng Code (Deterministic Verification), Kiểm soát quyền truy cập (Permission Control), Bắt lặp hành vi (Loop Detection) và Cơ chế dừng an toàn / Bàn giao cho con người (Safe Failure & Handoff).

---

## 1. Cấu Trúc Thư Mục Dự Án

```text
booking_Agent/
├── agents/                       # Triển khai 3 mẫu kiến trúc Agent
│   ├── __init__.py               # Khởi tạo package agents
│   ├── react_agent.py            # Mô hình phản xạ ReAct tuần tự
│   ├── plan_execute.py           # Mô hình Lập kế hoạch rồi Thực thi (Plan-then-Execute)
│   └── hybrid_agent.py           # Mô hình Lai (Hybrid / Substep ReAct)
├── lib/                          # Các module nền tảng và khung an toàn (Harness)
│   ├── __init__.py               # Khởi tạo package lib
│   ├── state.py                  # Định nghĩa TypedDict AgentState & Pydantic TicketBooking
│   ├── tools_booking.py          # Cơ sở dữ liệu mô phỏng & công cụ tra cứu, đặt vé
│   ├── guards.py                 # Rào chắn an toàn (Check quyền, Loop detector, Verify code)
│   ├── handoff.py                # Đóng gói ngữ cảnh bàn giao có cấu trúc (Handoff payload)
│   └── model.py                  # Quản lý LLM Provider (Hỗ trợ MockLLM offline & Gemini API)
├── benchmark.py                  # Script đánh giá hiệu năng tự động trên 6 kịch bản
├── main.py                       # Điểm chạy tương tác dòng lệnh (CLI trace)
├── requirements.txt              # Danh sách các thư viện phụ thuộc
└── README.md                     # Tài liệu hướng dẫn dự án
```
---

## 2. Các Đặc Tính Kỹ Thuật Nổi Bật

1. Ràng buộc là dữ liệu (Constraints as Data): Dữ liệu nghiệp vụ được chuẩn hóa thông qua Pydantic schema (TicketBooking) và LangGraph State (AgentState), kiểm soát chặt chẽ trạng thái trước khi ghi nhận giao dịch.
2. Kiểm soát quyền hạn (RBAC): Tài khoản cấp guest chỉ được quyền tra cứu chuyến bay; quyền member/admin mới được kích hoạt công cụ book_ticket.
3. Phát hiện vòng lặp vô ích (Loop Detection): LoopDetector theo dõi dấu vết hành động kèm tham số qua cửa sổ trượt (Sliding Window), chủ động ngắt để bảo vệ tài nguyên khi phát hiện gọi lặp lặp lại.
4. Tiêu chí hoàn thành kiểm bằng code: Không dựa vào câu chữ tự sinh của LLM mà trực tiếp kiểm tra sự tồn tại của booking_id hợp lệ và trạng thái confirmed từ cơ sở dữ liệu.
5. Bàn giao an toàn (Handoff Mechanism): Đóng gói toàn bộ lý do ngắt, các bước đã thử, snapshot trạng thái và câu hỏi cần giải quyết cho con người khi vi phạm quyền hoặc lỗi nghiệp vụ.
6. Môi trường kép (Dual Mode): Chạy kiểm thử linh hoạt ở chế độ Offline (MockLLM) không phụ thuộc Internet/API Key, hoặc kích hoạt chế độ API (Google Gemini) để đo lường thực tế.

---

## 3. Cài Đặt & Cấu Hình

### Yêu cầu tiên quyết
- Python 3.10 trở lên.

### Các bước cài đặt
1. Khởi tạo và kích hoạt môi trường ảo:
   # Trên Windows
   python -m venv .venv
   .\.venv\Scripts\activate

   # Trên Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate

2. Cài đặt các gói phụ thuộc:
   pip install -r requirements.txt

3. Cấu hình môi trường (Tùy chọn):
   - Nếu chạy Offline hoàn toàn: Mặc định hệ thống tự kích hoạt MockLLM.
   - Nếu muốn chạy với Google Gemini API: Tạo file .env tại thư mục gốc:
     BOOKING_LLM_MODE=api
     GEMINI_API_KEY=AIzaSy...your_gemini_api_key...

---

## 4. Hướng Dẫn Sử Dụng

### 4.1. Chạy tương tác CLI (main.py)

Cung cấp giao diện xem trực tiếp vết suy luận (trace) và các bước kích hoạt nút/công cụ:

- Chạy ReAct Agent với quyền thành viên đặt vé thành công:
  python main.py --agent react --role member

- Kiểm tra cơ chế chặn quyền người dùng Guest (kích hoạt Handoff):
  python main.py --agent react --role guest

- Chạy Plan-then-Execute Agent:
  python main.py --agent plan --role member

- Chạy Hybrid Agent kiểm tra kịch bản hết vé:
  python main.py --agent hybrid --role member --query "Đặt vé từ SGN đến DAD ngày 2026-10-15"

---

### 4.2. Chạy Benchmark so sánh hiệu năng (benchmark.py)

Tự động thực thi đánh giá 6 kịch bản kiểm thử (TC1: Đặt vé thành công, TC2: Chặn quyền Guest, TC3: Hết vé, TC4: Không tìm thấy chuyến, TC5: Sai ngày bay, TC6: Yêu cầu gây lặp) trên cả 3 kiến trúc:

python benchmark.py

---

## 5. Kết Quả Đo Đạc Thực Nghiệm (Benchmark)

Bảng tổng hợp thu thập từ quá trình chạy tự động 6 kịch bản:

| Mẫu thiết kế | Tỷ lệ thành công | Tỷ lệ bàn giao (Handoff) | Thời gian thực thi TB |
| :--- | :---: | :---: | :---: |
| **ReAct** | **16.7%** | **83.3%** | 0.005s |
| **Plan-then-Execute** | **33.3%** | **16.7%** | 0.003s |
| **Hybrid (Lai)** | **33.3%** | **66.7%** | 0.003s |

### Nhận xét kết quả:
- Tỷ lệ thành công thể hiện tính an toàn: Do tập kiểm thử có tới 5/6 ca là bất thường/vi phạm (TC2-TC6), một Agent vận hành đúng chuẩn bảo mật bắt buộc phải từ chối hoàn thành và chỉ cho phép duy nhất ca TC1 thành công (tương ứng tỷ lệ chuẩn 1/6 ≈ 16.7%).
- ReAct Agent thể hiện độ an toàn cao nhất khi từ chối hoàn thành toàn bộ các trường hợp lỗi dữ liệu/vi phạm quyền và kích hoạt bàn giao cho con người đạt 83.3%.
- Plan-then-Execute Agent có ưu thế về tốc độ do lập kế hoạch một lần, nhưng thiếu linh hoạt khi gặp lỗi ngoại lệ giữa chừng (dẫn đến tỷ lệ bàn giao thấp).
- Hybrid Agent cân bằng giữa việc giữ khung lộ trình và thích ứng động ở các bước thực thi con (bắt được ca hết vé và chuyển giao xử lý an toàn).
