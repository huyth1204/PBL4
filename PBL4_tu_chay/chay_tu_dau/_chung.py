"""Hàm dùng chung cho 4 tệp chạy thí nghiệm từ đầu (sinh dữ liệu → huấn luyện → chấm → in kết quả).

Mọi kết quả ghi vào ket_qua_tu_chay/<thí nghiệm>/ (chế độ --thu ghi vào ket_qua_tu_chay/<thí nghiệm>_thu/), không đụng
tới results/ (kết quả gốc để so). Bước nào đã xong thì lần sau bỏ qua, nên dừng giữa chừng rồi chạy lại vẫn được.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
os.chdir(GOC)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass
MOI_TRUONG = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def thu_muc(ten, thu):
    """Thư mục kết quả của một thí nghiệm; tự thêm .gitignore để git bỏ qua cả ket_qua_tu_chay/."""
    goc = GOC / "ket_qua_tu_chay"
    goc.mkdir(exist_ok=True)
    (goc / ".gitignore").write_text("*\n", encoding="utf-8")
    d = goc / (ten + ("_thu" if thu else ""))
    (d / "nhat_ky").mkdir(parents=True, exist_ok=True)
    return d


def kiem_dau_vao(*tep):
    thieu = [t for t in tep if not (GOC / t).exists()]
    if thieu:
        raise SystemExit("Thiếu tệp đầu vào: " + ", ".join(thieu) + "\nChạy từ thư mục gốc dự án (có src/, data/, dataset/).")


def tieu_de(ten, cau_hoi, thu):
    print("=" * 100)
    print(ten + ("   [CHẠY THỬ --thu: số liệu nhỏ, chỉ để kiểm tra máy]" if thu else ""))
    print("Câu hỏi:", cau_hoi)
    print("=" * 100, flush=True)


def buoc(ten, lenh, nhat_ky, xong=None, don_neu_do_dang=None):
    """Chạy `python -m <lenh...>`, in ra màn hình đồng thời ghi nhật ký.

    xong: tệp đánh dấu bước đã hoàn tất (có thì bỏ qua). don_neu_do_dang: thư mục của bước bị dừng giữa chừng,
    xoá đi trước khi chạy lại (các công cụ của nhóm từ chối ghi vào thư mục đã có dữ liệu).
    """
    if xong is not None and Path(xong).exists():
        print(f"[bỏ qua] {ten}: đã có {Path(xong).relative_to(GOC)}", flush=True)
        return
    if don_neu_do_dang is not None and Path(don_neu_do_dang).exists():
        shutil.rmtree(don_neu_do_dang)
    print(f"\n>>> {ten}\n    nhật ký: {Path(nhat_ky).relative_to(GOC)}", flush=True)
    t0 = time.time()
    with open(nhat_ky, "w", encoding="utf-8") as f:
        lenh = [str(x.relative_to(GOC)) if isinstance(x, Path) and x.is_relative_to(GOC) else str(x) for x in lenh]
        p = subprocess.Popen([sys.executable, "-W", "ignore", "-m", *lenh], cwd=GOC, env=MOI_TRUONG,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for dong in p.stdout:
            f.write(dong)
            f.flush()
            print("    " + dong.rstrip()[:160], flush=True)
        p.wait()
    if p.returncode != 0:
        raise SystemExit(f"LỖI ở bước '{ten}' (mã {p.returncode}). Xem nhật ký: {nhat_ky}")
    phut = (time.time() - t0) / 60
    print(f"<<< xong {ten} sau {phut:.1f} phút", flush=True)


def doc(p):
    return json.load(open(p, encoding="utf-8"))


def bang(cot, hang):
    rong = [max(len(str(c)), *(len(str(h[i])) for h in hang)) + 2 for i, c in enumerate(cot)]
    dong = lambda xs: "".join(str(x).ljust(w) if i == 0 else str(x).rjust(w) for i, (x, w) in enumerate(zip(xs, rong)))
    print(dong(cot))
    print("-" * sum(rong))
    for h in hang:
        print(dong(h))
    print(flush=True)


def so(x, n=2):
    return "—" if x is None else f"{x:.{n}f}".replace(".", ",")


def ktc(diem, khoang, n=2):
    return f"{so(diem, n)} [{so(khoang[0], n)}; {so(khoang[1], n)}]"
