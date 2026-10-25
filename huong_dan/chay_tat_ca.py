"""
Chạy tất cả — chạy lần lượt bước 0 → 9 mà không cần bấm gì, lưu mọi hình vào huong_dan/ket_qua/ và mở một trang tổng hợp
(bang_dieu_khien.html) trong trình duyệt. Trang tự làm mới trong lúc chạy: bước nào xong hiện ngay hình và bảng kiểm tra.
Chạy trong PyCharm: chuột phải tệp → Run 'chay_tat_ca'. Mất khoảng 3–6 phút trên máy thường.
Chỉ chạy vài bước: Run → Edit Configurations → Parameters, ví dụ  --buoc 3 5 7
"""
import argparse
import base64
import html
import importlib
import inspect
import io
import json
import sys
import time
import traceback
import webbrowser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huong_dan.tien_ich import GOC, plt  # noqa: E402

KQ = GOC / "huong_dan" / "ket_qua"
TRANG = KQ / "bang_dieu_khien.html"
NHAN = ("File dự án:", "File test:", "Mô hình:", "Dữ liệu:", "Tài liệu:", "Sẽ thấy:", "Chạy trong PyCharm:", "Luật:", "Chuỗi:")

CSS = """
body{margin:0;background:#fafaf7;color:#0b0b0b;font-family:"Times New Roman",Times,serif;font-size:17px;line-height:1.45}
main{max-width:1180px;margin:0 auto;padding:18px 16px 60px}
h1{font-size:26px;margin:6px 0 2px}h2{font-size:21px;margin:0}
.phu{color:#52514e;margin:0 0 14px}
.luoi{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:8px;margin:12px 0 22px}
.o{display:block;padding:8px 10px;border-radius:8px;background:#fff;border:1px solid #e4e3de;text-decoration:none;color:#0b0b0b}
.o b{display:block;font-size:15px}.o span{font-size:14px;color:#52514e}
.dat{border-left:6px solid #1baf7a}.loi{border-left:6px solid #d62728}.cho{border-left:6px solid #c9c8c3;opacity:.65}
.dang{border-left:6px solid #eda100}
section{background:#fff;border:1px solid #e4e3de;border-radius:10px;padding:16px 18px;margin:0 0 18px}
.dau{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.nhan{font-size:15px;font-weight:bold;padding:2px 10px;border-radius:12px;color:#fff}
.nhan.d{background:#1baf7a}.nhan.l{background:#d62728}
.mo p{margin:4px 0}.mo b{font-weight:bold}
img{display:block;max-width:100%;height:auto;margin:12px auto 4px;border:1px solid #eeede8;border-radius:6px}
table{border-collapse:collapse;width:100%;margin-top:10px;font-size:15.5px}
td,th{padding:5px 8px;text-align:left;vertical-align:top}th{background:#f3f2ee}
tr.d{background:#eaf7f1}tr.l{background:#fdecea}td.k{font-family:"DejaVu Sans",Arial,sans-serif;font-weight:bold;width:24px}
tr.d td.k{color:#1baf7a}tr.l td.k{color:#d62728}td.t{color:#8a8984;text-align:right;white-space:nowrap}
td.c{color:#52514e}pre{white-space:pre-wrap;background:#fdecea;padding:10px;border-radius:6px;font-size:13px}
.bd a{display:inline-block;margin:6px 10px 0 0;padding:8px 14px;border-radius:8px;background:#2a78d6;color:#fff;text-decoration:none}
"""


def tach_mo_ta(mod):
    dong = (inspect.getdoc(mod) or "").splitlines()
    tieu_de = dong[0] if dong else mod.__name__
    doan = []
    for d in dong[1:]:
        d = d.rstrip()
        if not d:
            continue
        if d.startswith("  ") and doan:
            doan[-1] += "<br>" + html.escape(d.strip())
            continue
        e = html.escape(d)
        for n in NHAN:
            if d.startswith(n):
                e = f"<b>{html.escape(n)}</b>{html.escape(d[len(n):])}"
        doan.append(e)
    return tieu_de, doan


def luu_hinh(ma):
    anh = []
    for k, so in enumerate(plt.get_fignums()):
        fig = plt.figure(so)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
        (KQ / f"{ma}_hinh{k + 1}.png").write_bytes(buf.getvalue())
        anh.append(base64.b64encode(buf.getvalue()).decode())
    plt.close("all")
    return anh


def chay_buoc(tep):
    mod = importlib.import_module(f"huong_dan.{tep.stem}")
    tieu_de, mo_ta = tach_mo_ta(mod)
    plt.close("all")
    t0, loi = time.time(), ""
    try:
        tham = {"mo_trinh_duyet": False} if "mo_trinh_duyet" in inspect.signature(mod.chay).parameters else {}
        kq = mod.chay(**tham)
    except Exception:  # noqa: BLE001
        kq, loi = [], traceback.format_exc()
    return {"ma": tep.stem, "tieu_de": tieu_de, "mo_ta": mo_ta, "kiem_tra": kq, "loi": loi,
            "giay": time.time() - t0, "anh": luu_hinh(tep.stem)}


def thoi_gian(k):
    return f"{k['giay']:.1f} s" if k.get("giay", 0) >= 0.1 else ""


def the_buoc(b):
    so = sum(k["dat"] for k in b["kiem_tra"])
    tot = not b["loi"] and so == len(b["kiem_tra"])
    hang = "".join(
        f"<tr class='{'d' if k['dat'] else 'l'}'><td class='k'>{'✔' if k['dat'] else '✘'}</td>"
        f"<td>{html.escape(k['ten'])}</td><td class='c'>{html.escape(str(k.get('chi_tiet', '')))}</td>"
        f"<td class='t'>{thoi_gian(k)}</td></tr>" for k in b["kiem_tra"])
    bang = f"<table><tr><th></th><th>Kiểm tra</th><th>Chi tiết</th><th></th></tr>{hang}</table>" if hang else ""
    anh = "".join(f"<img src='data:image/png;base64,{a}' alt='{b['ma']} hình {i + 1}'>" for i, a in enumerate(b["anh"]))
    ban_do = ""
    if b["ma"].startswith("buoc_9"):
        ban_do = "<div class='bd'>" + "".join(
            f"<a href='ban_do_{c}.html' target='_blank'>Mở bản đồ: {t}</a>"
            for c, t in (("sach", "dữ liệu sạch"), ("nhieu", "số đo nhiễu σ = 40%"), ("mua", "giờ mưa thật"))
            if (KQ / f"ban_do_{c}.html").exists()) + "</div>"
    loi = f"<pre>{html.escape(b['loi'])}</pre>" if b["loi"] else ""
    return (f"<section id='{b['ma']}'><div class='dau'><h2>{html.escape(b['tieu_de'])}</h2>"
            f"<span class='nhan {'d' if tot else 'l'}'>{so}/{len(b['kiem_tra'])} đạt · {b['giay']:.0f} s</span></div>"
            f"<div class='mo'>{''.join(f'<p>{d}</p>' for d in b['mo_ta'])}</div>{ban_do}{loi}{anh}{bang}</section>")


def ghi_trang(xong, con, dang_chay):
    o = []
    for b in xong:
        so = sum(k["dat"] for k in b["kiem_tra"])
        tot = not b["loi"] and so == len(b["kiem_tra"])
        o.append(f"<a class='o {'dat' if tot else 'loi'}' href='#{b['ma']}'><b>{html.escape(b['tieu_de'].split('—')[0])}</b>"
                 f"<span>{html.escape(b['tieu_de'].split('—')[-1].strip())}<br>{so}/{len(b['kiem_tra'])} đạt · {b['giay']:.0f} s</span></a>")
    for i, t in enumerate(con):
        lop = "dang" if i == 0 and dang_chay else "cho"
        o.append(f"<div class='o {lop}'><b>{t.stem.split('_')[0].replace('buoc', 'Bước')} {t.stem.split('_')[1]}</b>"
                 f"<span>{'đang chạy…' if lop == 'dang' else 'chờ'}</span></div>")
    tong = sum(len(b["kiem_tra"]) for b in xong)
    dat = sum(k["dat"] for b in xong for k in b["kiem_tra"])
    lam_moi = "<meta http-equiv='refresh' content='5'>" if dang_chay else ""
    trang_thai = (f"Đang chạy… ({len(xong)}/{len(xong) + len(con)} bước xong, trang tự làm mới 5 giây một lần)"
                  if dang_chay else f"Xong: {dat}/{tong} kiểm tra đạt, {sum(b['giay'] for b in xong):.0f} giây")
    TRANG.write_text(
        f"<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
        f"initial-scale=1'>{lam_moi}<title>Kết quả hướng dẫn PBL4</title><style>{CSS}</style></head><body><main>"
        f"<h1>DM + GNN định tuyến SAGSIN — kết quả chạy trên máy bạn</h1>"
        f"<p class='phu'>{trang_thai}. Mỗi ô là một tệp huong_dan/buoc_*.py; bấm vào ô để tới phần của bước đó.</p>"
        f"<div class='luoi'>{''.join(o)}</div>{''.join(the_buoc(b) for b in xong)}</main></body></html>",
        encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buoc", type=int, nargs="*", help="chỉ chạy các bước này, ví dụ --buoc 3 5 7")
    ap.add_argument("--khong-mo", action="store_true", help="không tự mở trình duyệt")
    a = ap.parse_args()
    KQ.mkdir(parents=True, exist_ok=True)
    tep = sorted((GOC / "huong_dan").glob("buoc_*.py"), key=lambda p: int(p.stem.split("_")[1]))
    if a.buoc:
        tep = [t for t in tep if int(t.stem.split("_")[1]) in a.buoc]
    xong = []
    ghi_trang(xong, tep, True)
    if not a.khong_mo:
        webbrowser.open(TRANG.as_uri())
    for i, t in enumerate(tep):
        print(f"[{i + 1}/{len(tep)}] {t.stem} …", flush=True)
        b = chay_buoc(t)
        xong.append(b)
        so = sum(k["dat"] for k in b["kiem_tra"])
        print(f"        {so}/{len(b['kiem_tra'])} đạt, {b['giay']:.0f} s" + (" — LỖI, xem trang kết quả" if b["loi"] else ""), flush=True)
        ghi_trang(xong, tep[i + 1:], i + 1 < len(tep))
    tom = [{"buoc": b["ma"], "dat": sum(k["dat"] for k in b["kiem_tra"]), "tong": len(b["kiem_tra"]), "giay": round(b["giay"], 1),
            "loi": bool(b["loi"]), "kiem_tra": [{k_: v for k_, v in k.items() if k_ != "giay"} for k in b["kiem_tra"]]} for b in xong]
    (KQ / "tong_ket.json").write_text(json.dumps(tom, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Trang kết quả: {TRANG}")
    return all(not b["loi"] and all(k["dat"] for k in b["kiem_tra"]) for b in xong)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
