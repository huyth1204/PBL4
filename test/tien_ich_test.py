"""Tiện ích dùng chung cho các test tự viết (repo không dùng pytest).

`BoQua` (skip) khác hẳn PASS: test không chạy được vì thiếu dữ liệu/mô hình
KHÔNG được tính là đạt, và bộ test sẽ in cảnh báo rõ ràng để không ai tưởng là xanh.

Đặt ở module riêng (không để trong test/__init__.py) vì khi chạy
`python test/test_mo_rong.py`, `import test` sẽ trúng gói `test` của thư viện
chuẩn Python chứ không phải thư mục test/ của repo.
"""


class BoQua(Exception):
    """Ném ra khi test không chạy được do thiếu dữ liệu/mô hình (không phải lỗi code)."""


def dung_utf8():
    """Ép stdout/stderr sang UTF-8.

    Console Windows mặc định cp1252 -> in tiếng Việt là UnicodeEncodeError (kể cả khi
    chuyển hướng ra file hoặc bắt qua pipe). Gọi sớm ở mọi điểm vào.
    """
    import sys
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def chay_tests(globals_dict, ten=""):
    """Chạy mọi hàm `test_*` trong `globals_dict`, in PASS/FAIL/SKIP.

    Trả về True nếu không có FAIL. SKIP không làm hỏng bộ test nhưng có cảnh báo.
    """
    dung_utf8()
    tests = [(n, f) for n, f in sorted(globals_dict.items())
             if n.startswith("test_") and callable(f)]
    dem = {"pass": 0, "skip": 0, "fail": 0}
    for n, f in tests:
        try:
            f()
        except BoQua as e:
            print(f"  [SKIP] {n}: {e}")
            dem["skip"] += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {n}: {type(e).__name__}: {e}")
            dem["fail"] += 1
        else:
            print(f"  [PASS] {n}")
            dem["pass"] += 1
    dau = f"{ten}: " if ten else ""
    phu = []
    if dem["skip"]:
        phu.append(f"{dem['skip']} bỏ qua (không kiểm chứng)")
    if dem["fail"]:
        phu.append(f"{dem['fail']} lỗi")
    print(f"\n{dau}{dem['pass']}/{len(tests)} đạt"
          + (", " + ", ".join(phu) if phu else "") + ".")
    if dem["skip"]:
        print("!! Có test BỎ QUA: kết quả trên KHÔNG chứng minh phần đó đúng. "
              "Bổ sung dữ liệu/mô hình rồi chạy lại.")
    return dem["fail"] == 0
