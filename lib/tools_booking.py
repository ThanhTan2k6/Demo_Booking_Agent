from langchain_core.tools import tool

# Mock Database vé chuyến bay
FLIGHT_DB = {
    ("SGN", "HAN", "2026-10-15"): {"seats": 2, "price": 1500000},
    ("SGN", "DAD", "2026-10-15"): {"seats": 0, "price": 1000000},  # Hết vé để kiểm thử dừng/lặp
}

def reset_db():
    """Reset lại dữ liệu ghế trống trước mỗi lượt chạy test."""
    global FLIGHT_DB
    FLIGHT_DB = {
        ("SGN", "HAN", "2026-10-15"): {"seats": 2, "price": 1500000},
        ("SGN", "DAD", "2026-10-15"): {"seats": 0, "price": 1000000},
    }

@tool
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Tra cứu chuyến bay theo nơi đi (SGN, HAN,...), nơi đến và ngày."""
    key = (origin.upper(), destination.upper(), date)
    if key in FLIGHT_DB:
        info = FLIGHT_DB[key]
        return {"status": "available", "seats": info["seats"], "price": info["price"]}
    return {"status": "not_found", "error": f"Không có chuyến từ {origin} đến {destination} ngày {date}"}

@tool
def book_ticket(origin: str, destination: str, date: str, customer_id: str = "KH_01") -> dict:
    """Đặt vé chuyến bay sau khi đã kiểm tra chỗ."""
    key = (origin.upper(), destination.upper(), date)
    if key in FLIGHT_DB and FLIGHT_DB[key]["seats"] > 0:
        FLIGHT_DB[key]["seats"] -= 1
        return {"status": "confirmed", "booking_id": f"BK_{origin.upper()}_{destination.upper()}_88"}
    return {"status": "failed", "error": "Hết ghế hoặc không tìm thấy chuyến."}

# Danh sách tools cung cấp cho Agent
tools = [search_flights, book_ticket]