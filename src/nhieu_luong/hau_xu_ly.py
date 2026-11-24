"""
hau_xu_ly.py — Thí nghiệm 14 (results/bao_da_tuyen/TIEU_CHI_14.md): hai bước hậu xử lý sau DM, chỉ dùng thông tin có lúc
quyết định (bảng rủi ro, dung lượng / tải nền quan sát ở o, nhóm rủi ro của cạnh), chỉ chọn trong kho của DM.

  va_het_tuyen  (nhược điểm 1, phần "không còn tuyến sống")  GIỮ tuyến chính DM đã chọn, chọn lại các tuyến dự phòng để
                mất DV ước lượng trên bảng rủi ro TƯƠNG QUAN theo nhóm (TheGioi.tu_bang_nhom) nhỏ nhất. Tuyến chính không
                đổi nên trễ tuyến chính và tải tuyến chính không đổi.
  giam_tre      (nhược điểm 2, trễ tuyến chính)  giữ nguyên TẬP tuyến của mỗi luồng, đổi tuyến nào làm tuyến chính nếu trễ
                giảm mà mất DV ước lượng không tăng và tuyến chính mới không rủi ro hơn theo bảng (an toàn trước, trễ sau).
Chẩn đoán (chỉ để phân tích, dùng thế giới CHẤM — không phải phương pháp):
  san_a_kho     phần "không còn tuyến sống" nhỏ nhất kho cho phép, từng luồng chọn riêng.
  diem_hong_chung  tỉ lệ luồng (theo λ) có mọi tuyến cùng đi qua một nhóm rủi ro (trừ nút nguồn / đích).
"""

from __future__ import annotations

import numpy as np


def _mat(H, lam):
    return (H * lam).sum(-1) / lam.sum()


def va_het_tuyen(tgw, a, so_vong=2):
    """TKCB chỉ trên các lựa chọn có CÙNG tuyến chính với lời giải hiện tại, mục tiêu = mất DV ước lượng (λ-trọng) trên
    tgw. Chỉ nhận thay đổi làm giảm hẳn; luồng lớn trước. Trả lựa chọn mới."""
    bt = tgw.bt
    a = np.array(a, np.int64).copy()
    for _ in range(so_vong):
        doi = False
        for k in np.argsort(-bt.lam, kind="stable"):
            O = bt.opts[k]
            chi = np.flatnonzero(O[:, 0] == O[a[k], 0])
            if len(chi) < 2:
                continue
            H, _, lam = tgw.Hd_lua_chon(a, k, chi=chi)
            m = _mat(H, lam)
            j, cu = int(np.argmin(m)), int(np.flatnonzero(chi == a[k])[0])
            if m[j] < m[cu] - 1e-12:
                a[k], doi = int(chi[j]), True
        if not doi:
            break
    return a


def _cung_tap(bt, k):
    """{tập tuyến (frozenset): [chỉ số lựa chọn]} của luồng k — các lựa chọn chỉ khác nhau ở tuyến nào là tuyến chính."""
    if not hasattr(bt, "_cung_tap"):
        bt._cung_tap = {}
    if k not in bt._cung_tap:
        d = {}
        for j, o in enumerate(bt.opts[k]):
            d.setdefault(frozenset(int(i) for i in o), []).append(j)
        bt._cung_tap[k] = d
    return bt._cung_tap[k]


def rui_ro_tuyen(bt, k):
    """[số kho] xác suất hỏng của từng tuyến trong kho luồng k theo bảng rủi ro (cạnh độc lập): 1 − Π(1 − p)."""
    return -np.expm1(bt.inc[k].astype(float) @ np.log1p(-np.clip(bt.risk, 0, 1 - 1e-9)))


def giam_tre(tgw, a, so_vong=1):
    """Mỗi luồng (lớn trước): thử đổi tuyến chính sang tuyến khác trong cùng tập; nhận phương án có trễ tuyến chính
    (λ-trọng, mọi luồng) nhỏ nhất trong các phương án KHÔNG làm tăng mất DV ước lượng trên tgw VÀ có tuyến chính mới
    không rủi ro hơn tuyến chính cũ theo bảng (an toàn trước, trễ sau)."""
    bt = tgw.bt
    a = np.array(a, np.int64).copy()
    for _ in range(so_vong):
        doi = False
        for k in np.argsort(-bt.lam, kind="stable"):
            chi = np.array(_cung_tap(bt, k)[frozenset(int(i) for i in bt.opts[k][a[k]])], np.int64)
            if len(chi) < 2:
                continue
            H, d, lam = tgw.Hd_lua_chon(a, k, chi=chi)
            m, tr = _mat(H, lam), _mat(d, lam)
            cu = int(np.flatnonzero(chi == a[k])[0])
            p = rui_ro_tuyen(bt, k)[bt.opts[k][chi, 0]]
            duoc = np.flatnonzero((m <= m[cu] + 1e-12) & (tr < tr[cu] - 1e-9) & (p <= p[cu] + 1e-12))
            if len(duoc):
                a[k], doi = int(chi[duoc[np.argmin(tr[duoc])]]), True
        if not doi:
            break
    return a


def san_a_kho(tgw):
    """Chẩn đoán (thế giới chấm): % lưu lượng (λ-trọng) không còn tuyến sống nếu MỖI luồng chọn tập ít cùng chết nhất
    trong kho. Phần (a) không phụ thuộc luồng khác nên cận này đạt được đồng thời."""
    bt = tgw.bt
    v = np.array([(~tgw.alive[k][:, bt.opts[k]].any(-1)).mean(0).min() for k in range(bt.K)])
    return float((v * bt.lam).sum() / bt.lam.sum() * 100)


def diem_hong_chung(bt, nhom, nut, dau_cuoi, a):
    """% lưu lượng (λ-trọng) có mọi tuyến của tập cùng qua ít nhất một nhóm rủi ro, không kể nhóm của nút nguồn / đích.
    nhom [U, 3] nhóm của từng cạnh; nut [số nhóm] nút của nhóm nút (−1 = nhóm mưa); dau_cuoi [K, 2]."""
    if nhom is None:
        return float("nan")
    nhom = np.asarray(nhom, np.int64)
    co = np.zeros(bt.K, bool)
    for k in range(bt.K):
        bo = {int(j) for j in np.flatnonzero(np.isin(nut, dau_cuoi[k]) & (nut >= 0))}
        chung = None
        for i in bt.opts[k][a[k]]:
            g = {int(x) for x in nhom[bt.inc[k][i]].ravel() if x >= 0} - bo
            chung = g if chung is None else chung & g
        co[k] = bool(chung)
    return float((co * bt.lam).sum() / bt.lam.sum() * 100)
