"""
truc_quan_dm.py — Viết tiếp visualization.py: vẽ tuyến do mô hình DM chọn trên bản đồ thật, cạnh tuyến tối ưu.

Ba chế độ:
  sach  : dữ liệu sạch — tuyến tối ưu (nhãn Dijkstra) và tuyến DM (results/dmx0_4lop.pt)
  nhieu : số đo trễ có nhiễu σ — thêm tuyến Dijkstra chạy trên số đo (bị nhiễu đánh lừa) và tuyến DM học với nhiễu
  mua   : một giờ mưa thật (ERA5) — tô các liên kết đang có mưa, điểm đang mưa, và luật chia đôi chọn Dijkstra hay DM

    python truc_quan_dm.py                                  # chế độ sạch, mẫu 5 của dataset/test.npz
    python truc_quan_dm.py --che-do nhieu --sigma 0.4 --sample 120
    python truc_quan_dm.py --che-do mua --sample 60

Bản đồ nhúng sẵn đường bờ biển (data/topojson) nên xem được cả khi không có mạng; tự mở trong trình duyệt.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC))
import visualization as vz  # noqa: E402

MAU = {"toi_uu": "#ff66cc", "dm": "#2a78d6", "dijkstra_do": "#eb6834", "mua": "#7fb3e6", "diem_mua": "#1f5fbf"}


def _tuyen_geo(coords, p):
    return vz.split_geo_line(coords[p, 1].tolist(), coords[p, 0].tolist())


def them_tuyen(fig, coords, p, ten, mau, rong=4, kieu="solid", hover=""):
    rong += 1
    lon, lat = _tuyen_geo(coords, p)
    fig.add_trace(go.Scattergeo(lon=lon, lat=lat, mode="lines", name=ten, line=dict(width=rong, color=mau, dash=kieu),
                                hoverinfo="text", text=hover))
    return len(fig.data) - 1


def tre(W, p):
    return float(sum(W[u, v] for u, v in zip(p[:-1], p[1:])))


def nhung_topojson(html: Path):
    """Nhúng bản đồ nền vào trang để plotly không phải tải từ Internet."""
    topo = GOC / "data" / "topojson" / "world_50m.json"
    if not topo.exists():
        return
    s = html.read_text(encoding="utf-8")
    chen = "<script>window.PlotlyGeoAssets = {topojson: {'world_50m': " + topo.read_text(encoding="utf-8") + "}};</script>"
    html.write_text(s.replace("<head>", "<head>" + chen, 1), encoding="utf-8")


def tu_chon_mau(a, d):
    """Chọn một mẫu cho thấy rõ điều cần minh họa (không đổi kết quả, chỉ để hình dễ hiểu)."""
    from test.danh_gia_quy_mo import chay_mo_hinh, load
    from test.so_sanh_dijkstra_nhieu import dijkstra, lam_nhieu
    if a.che_do == "sach":
        return next(k for k in range(len(d)) if len(d.paths[k]) >= 4 and
                    any(str(d.layer[n]) == "space" for n in d.paths[k]))
    m = load(GOC / "results" / "quy_mo" / "dm_nhieu.pt")
    if a.che_do == "nhieu":
        Wd = lam_nhieu(d.W, a.sigma, a.seed)
        ung = [k for k in range(0, len(d), 3) if tre(d.W[d.sid[k]], dijkstra(Wd[d.sid[k]], d.paths[k][0], d.paths[k][-1],
                                                                              d.can_relay)) > 1.1 * tre(d.W[d.sid[k]], d.paths[k])]
        dn = copy.copy(d)
        dn.W = Wd
        rt, ok = chay_mo_hinh(m, dn, np.array(ung), 1.0, 2, 200)
        return next((k for k, r in zip(ung, rt) if r == d.paths[k]), ung[0] if ung else 0)
    from src.ngau_nhien.kich_ban import MangGoc
    from src.ngau_nhien.thoi_tiet import ThoiTiet
    from test.luat_chia_doi import delta_matran, nhieu_lk
    tt = ThoiTiet(str(GOC / "data" / "thoi_tiet_vn.npz"))
    h = a.gio if a.gio is not None else int(np.argmax((tt.precip[:, :13] > 2.5).sum(1)))
    g = MangGoc(d.W, d.kind, d.node_order, tt)
    rng = np.random.default_rng(a.seed)
    A, Dm = d.W.copy(), {}
    for s in np.unique(d.sid):
        D, mua = delta_matran(g, d, tt, s, h)
        Dm[s] = D
        A[s] = nhieu_lk(d.W[s], np.minimum(np.sqrt(a.sigma0 ** 2 + D ** 2), 0.8), rng)
    ung = []
    for k in range(len(d)):
        p = dijkstra(A[d.sid[k]], d.paths[k][0], d.paths[k][-1], d.can_relay)
        if np.sqrt(a.sigma0 ** 2 + Dm[d.sid[k]][p[:-1], p[1:]] ** 2).max() > 0.075:
            ung.append((k, p))
    dn = copy.copy(d)
    dn.W = A
    rt, ok = chay_mo_hinh(m, dn, np.array([k for k, _ in ung]), 1.0, 2, 200)
    for (k, p), r in zip(ung, rt):
        if r != p and tre(d.W[d.sid[k]], r) < tre(d.W[d.sid[k]], p):
            return k
    return ung[0][0] if ung else 0


def khung_nhin(fig, coords, cac_tuyen, data):
    """Thu khung về Việt Nam và các nút trên tuyến (khung của visualization.py rộng 70° kinh độ)."""
    lop = vz.infer_layers(data)
    chon = sorted(set(np.flatnonzero(lop != "space")) | {n for p in cac_tuyen for n in p})
    lon, lat = coords[chon, 1], coords[chon, 0]
    lat_lo, lat_hi = float(lat.min()) - 2.5, float(lat.max()) + 2.5
    nua = max((float(lon.max()) - float(lon.min())) / 2 + 3, (lat_hi - lat_lo) * 0.9)
    tam = (float(lon.max()) + float(lon.min())) / 2
    fig.update_geos(lonaxis_range=[tam - nua, tam + nua], lataxis_range=[lat_lo, lat_hi],
                    center=dict(lon=tam, lat=(lat_lo + lat_hi) / 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--che-do", choices=["sach", "nhieu", "mua"], default="sach")
    ap.add_argument("--data", default="dataset/test.npz")
    ap.add_argument("--sample", type=int, default=None, help="mặc định: tự tìm một mẫu minh họa rõ cho chế độ đã chọn")
    ap.add_argument("--sigma", type=float, default=0.4, help="chế độ nhieu: mức nhiễu đo")
    ap.add_argument("--sigma0", type=float, default=0.02, help="chế độ mua: chất lượng đo chung của mạng")
    ap.add_argument("--gio", type=int, default=None, help="chế độ mua: chỉ số giờ trong data/thoi_tiet_vn.npz")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default=None)
    ap.add_argument("--khong-mo", action="store_true", help="không tự mở trình duyệt")
    a = ap.parse_args()

    from src.ai.data_v2 import SagsinData
    from test.danh_gia_quy_mo import chay_mo_hinh, load
    from test.so_sanh_dijkstra_nhieu import dijkstra, lam_nhieu

    duong_dan = (GOC / a.data).resolve()
    data = vz.load_npz(duong_dan)
    d = SagsinData(duong_dan)
    assert len(d) == len(data["y_paths_indices"]), "dataset có đường độ dài < 2, không khớp chỉ số mẫu"
    if a.sample is None:
        a.sample = tu_chon_mau(a, d)
    split_info = vz.load_split_info(GOC / "dataset" / "split_info.json")
    tle = GOC / "data" / "tle" / "starlink.txt"
    fig, sid, sid_goc, p_nhan, thoi_diem = vz.build_figure(data, a.sample, None, tle, split_info, 42, 20)
    coords, _ = vz.get_coordinates(data, sid, sid_goc, tle, 42, 20)
    ten = data["node_order"].astype(str)
    k = a.sample
    W = d.W[sid]
    so_cu = len(fig.data)
    moi, tieu_de, ghi_chu = [], "", []

    if a.che_do == "sach":
        m = load(GOC / "results" / "dmx0_4lop.pt")
        rt, ok = chay_mo_hinh(m, d, np.array([k]), 1.0, 2, 200)
        p_dm = rt[0]
        moi.append(them_tuyen(fig, coords, p_dm, f"Tuyến DM ({tre(W, p_dm):.2f} ms)", MAU["dm"], 3,
                              hover=" → ".join(ten[p_dm])))
        tieu_de = f"DM so với tối ưu — tối ưu {tre(W, p_nhan):.2f} ms, DM {tre(W, p_dm):.2f} ms" + \
                  (" (trùng nhau)" if p_dm == p_nhan else "")
        ghi_chu = ["Hồng: tuyến tối ưu (Dijkstra)", "Xanh: tuyến DM chọn"]

    elif a.che_do == "nhieu":
        m = load(GOC / "results" / "quy_mo" / "dm_nhieu.pt")
        Wd = lam_nhieu(d.W, a.sigma, a.seed)
        dn = copy.copy(d)
        dn.W = Wd
        rt, ok = chay_mo_hinh(m, dn, np.array([k]), 1.0, 2, 200)
        p_dj = dijkstra(Wd[sid], p_nhan[0], p_nhan[-1], d.can_relay)
        moi.append(them_tuyen(fig, coords, p_dj, f"Dijkstra trên số đo ({tre(W, p_dj):.2f} ms thật)", MAU["dijkstra_do"], 3,
                              "dash", hover=" → ".join(ten[p_dj])))
        moi.append(them_tuyen(fig, coords, rt[0], f"DM học với nhiễu ({tre(W, rt[0]):.2f} ms thật)", MAU["dm"], 3,
                              hover=" → ".join(ten[rt[0]])))
        tieu_de = (f"Số đo nhiễu σ = {a.sigma:.0%}: tối ưu {tre(W, p_nhan):.2f} ms | Dijkstra trên số đo "
                   f"{tre(W, p_dj):.2f} ms | DM {tre(W, rt[0]):.2f} ms")
        ghi_chu = ["Hồng: tối ưu trên trễ thật", "Cam đứt: Dijkstra tin số đo", "Xanh: DM học với nhiễu"]

    else:
        from src.ngau_nhien.kich_ban import MangGoc
        from src.ngau_nhien.thoi_tiet import ThoiTiet
        from test.luat_chia_doi import delta_matran, nhieu_lk, uoc_luong_sigma0
        tt = ThoiTiet(str(GOC / "data" / "thoi_tiet_vn.npz"))
        h = a.gio if a.gio is not None else int(np.argmax((tt.precip[:, :13] > 2.5).sum(1)))
        g = MangGoc(d.W[sid:sid + 1], d.kind, d.node_order, tt)
        dd = copy.copy(d)
        dd.W = d.W[sid:sid + 1]
        D, mua = delta_matran(g, dd, tt, 0, h)
        rng = np.random.default_rng(a.seed)
        S = np.minimum(np.sqrt(a.sigma0 ** 2 + D ** 2), 0.8)
        A = nhieu_lk(W, S, rng)
        s_hat = uoc_luong_sigma0(A, nhieu_lk(W, S, rng), np.triu(W > 0, 1) & ~mua)
        p_dj = dijkstra(A, p_nhan[0], p_nhan[-1], d.can_relay)
        sig = np.sqrt(s_hat ** 2 + D[p_dj[:-1], p_dj[1:]] ** 2)
        dung_dijkstra = s_hat <= 0.075 and sig.max() <= 0.075
        iu, iv = np.nonzero(np.triu(mua, 1))
        lon, lat = [], []
        for u, v in zip(iu, iv):
            lo, la = _tuyen_geo(coords, [u, v])
            lon += lo + [None]
            lat += la + [None]
        fig.add_trace(go.Scattergeo(lon=lon, lat=lat, mode="lines", name=f"Liên kết đang có mưa ({len(iu)})",
                                    line=dict(width=1.5, color=MAU["mua"]), hoverinfo="skip"))
        moi.append(len(fig.data) - 1)
        dat = [i for i, n in enumerate(ten) if n in tt.names and tt.precip[h, tt.idx(n)] > 0.1]
        fig.add_trace(go.Scattergeo(lon=coords[dat, 1], lat=coords[dat, 0], mode="markers", name="Điểm đang mưa",
                                    marker=dict(size=[8 + 3 * min(tt.precip[h, tt.idx(ten[i])], 12) for i in dat],
                                                color=MAU["diem_mua"], opacity=0.45),
                                    text=[f"{ten[i]}: {tt.precip[h, tt.idx(ten[i])]:.1f} mm/h" for i in dat],
                                    hoverinfo="text"))
        moi.append(len(fig.data) - 1)
        if dung_dijkstra:
            moi.append(them_tuyen(fig, coords, p_dj, f"Luật chọn Dijkstra ({tre(W, p_dj):.2f} ms)", MAU["dijkstra_do"], 3,
                                  hover=" → ".join(ten[p_dj])))
            quyet = "Dijkstra (tuyến không bị mưa làm nhiễu)"
        else:
            m = load(GOC / "results" / "quy_mo" / "dm_nhieu.pt")
            dn = copy.copy(d)
            dn.W = d.W.copy()
            dn.W[sid] = A
            rt, ok = chay_mo_hinh(m, dn, np.array([k]), 1.0, 2, 200)
            moi.append(them_tuyen(fig, coords, p_dj, f"Dijkstra đi qua mưa ({tre(W, p_dj):.2f} ms) → bị loại",
                                  MAU["dijkstra_do"], 2, "dash", hover=" → ".join(ten[p_dj])))
            moi.append(them_tuyen(fig, coords, rt[0], f"Luật chọn DM ({tre(W, rt[0]):.2f} ms)", MAU["dm"], 3,
                                  hover=" → ".join(ten[rt[0]])))
            quyet = "DM (tuyến Dijkstra đi qua liên kết nhiễu do mưa)"
        tieu_de = f"Giờ mưa thật {str(tt.time_utc[h])[:13]}h UTC | σ̂0 = {s_hat:.3f} | luật chọn: {quyet}"
        ghi_chu = ["Xanh nhạt: liên kết đang có mưa", "Chấm xanh: điểm đang mưa (to = mưa nhiều)",
                   "Cam: tuyến Dijkstra", "Xanh đậm: tuyến luật chọn"]

    khung_nhin(fig, coords, [p_nhan] + [q for q in (locals().get("p_dm"), locals().get("p_dj")) if q] +
               ([rt[0]] if "rt" in locals() else []), data)
    for tr in fig.data[:so_cu]:
        if tr.mode == "lines" and tr.name != "Selected route":
            tr.visible = "legendonly"                  # ẩn lưới liên kết để thấy rõ tuyến; bấm chú giải để hiện lại
    menu = fig.layout.updatemenus[0]
    for nut, hien in zip(menu.buttons, (True, True, False, False)):
        vis = list(nut.args[0]["visible"]) + [False] * (so_cu - len(nut.args[0]["visible"]))
        nut.args[0]["visible"] = vis + [hien] * (len(fig.data) - so_cu)
    fig.update_geos(resolution=50)
    fig.update_layout(title=dict(text=tieu_de, font=dict(size=15)),
                      legend=dict(y=0.99, yanchor="top"))
    fig.add_annotation(text="<br>".join(ghi_chu), xref="paper", yref="paper", x=0.01, y=0.02, xanchor="left",
                       yanchor="bottom", showarrow=False, align="left", bgcolor="rgba(255,255,255,0.85)",
                       font=dict(size=12))
    ra = Path(a.output) if a.output else GOC / "huong_dan" / "ket_qua" / f"ban_do_{a.che_do}.html"
    ra.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(ra), include_plotlyjs=True, config={"scrollZoom": True, "doubleClick": "reset"})
    nhung_topojson(ra)
    json.dump({"che_do": a.che_do, "sample": a.sample, "tieu_de": tieu_de}, open(ra.with_suffix(".json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    if not a.khong_mo:
        webbrowser.open(ra.resolve().as_uri())
    return ra


if __name__ == "__main__":
    main()
