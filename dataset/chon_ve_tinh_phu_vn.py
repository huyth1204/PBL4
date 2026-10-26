"""
chon_ve_tinh_phu_vn.py — lọc từ TLE Starlink thật ra đúng N vệ tinh PHỦ VIỆT NAM.

Vì sao cần: `--sat-select random` của labeler lấy mẫu đều trên TOÀN CẦU. Với
mạng Việt Nam, trung bình chỉ **2.36/150** vệ tinh nối được xuống trạm ở mỗi
snapshot (đo trên v3, 100 vệ tinh: 1.80). Nghĩa là phần lớn thời gian mạng hoàn
toàn mất GSL, và nhãn chỉ còn phụ thuộc HAPS/UAV — bài toán mất hẳn ý nghĩa
"định tuyến qua vệ tinh".

Cách làm: xếp hạng từng vệ tinh theo **số snapshot nó nối được xuống Việt Nam**
(góc nâng >= `--min-elev` tại một trong các điểm đại diện Bắc/Trung/Nam/đảo),
rồi lấy tham lam từ trên xuống.

BẪY QUAN TRỌNG — đừng chọn theo "đang bay trên đầu Việt Nam lúc t0":

    | Cách chọn                | nối được TB | snapshot có >=1 |
    |--------------------------|-------------|-----------------|
    | random 100 (như v3)      | 1.80        | 87.8 %          |
    | random 150               | 2.36        | 89.8 %          |
    | góc nâng cao nhất ở t0   | 4.01        | **11.2 %**      |
    | nhóm epoch + tham lam    | **6.55**    | **98.8 %**      |

Chọn theo vị trí tức thời TỆ HƠN CẢ RANDOM: 150 vệ tinh đang bay trên Việt Nam
ở t0 đều nằm trong vài mặt phẳng gần nhau, chúng rời đi gần như đồng thời và
sau đó mạng trống hoàn toàn.

BẪY THỨ HAI — mốc `t_start` của labeler KHÔNG phải median của toàn tệp TLE:

    `resolve_start_time(ts, "tle", sample_sats)` lấy median epoch của CHÍNH tập
    truyền vào. Tối ưu trên 10 728 vệ tinh cho 9.50 cặp/snapshot, nhưng tệp 150
    con có median lệch ~5 giờ nên dataset thật chỉ còn 0.69 (494/600 snapshot
    trống). Lặp "chọn -> đọc mốc -> chọn lại" cũng KHÔNG hội tụ (mốc nhảy -5.67 h
    rồi +1.30 h). Lối ra: chỉ chọn trong nhóm vệ tinh CÙNG EPOCH đến giây
    (tệp thật có 585 con) — khi đó median = epoch chung, bất kể chọn con nào,
    nên mốc labeler cố định. Script assert điều này trước khi ghi tệp.

Ràng buộc đa dạng độ nghiêng (`--tran-nghieng`): tập tối ưu thuần tuý dồn
140/150 vệ tinh vào vỏ 43° (vì 43° là vỏ thấp nhất, phủ nhiệt đới tốt nhất),
không giống chòm Starlink thật (43/53/70/97°). Đặt trần 55 % cho mỗi nhóm
nghiêng chỉ mất 0.25 vệ tinh nối được trung bình mà vẫn giữ được tính đa dạng.

Tệp sinh ra **vẫn là TLE Starlink THẬT**, chỉ là tập con — khác hẳn
`sinh_tle_walker.py` (chòm lý tưởng). Đừng trộn lẫn hai loại.

Cách dùng:
    python dataset/chon_ve_tinh_phu_vn.py --so-ve-tinh 150 \\
        --out data/tle/starlink_vn150.txt

    python dataset/oracle_labeler_sagsin.py --mo-rong --sats 150 \\
        --sat-select first --tle data/tle/starlink_vn150.txt \\
        --out dataset/oracle_dataset_v5.npz
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))

for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        try:
            _r(encoding="utf-8")
        except (ValueError, OSError):
            pass

from skyfield.api import load, wgs84  # noqa: E402

from src.physics.tle_loader import load_tle  # noqa: E402

RAD = 57.29577951308232  # radian -> độ

# Điểm đại diện Việt Nam: Bắc / Trung / Nam / Côn Đảo.
# Góc nâng lấy MAX trên 4 điểm — vệ tinh chỉ cần phủ MỘT vùng là hữu ích, vì
# các trạm gateway nằm rải khắp đất nước và đường đi có thể qua gateway khác.
DIEM_VN = [
    ("Bac", 21.0285, 105.8542),    # Hà Nội
    ("Trung", 16.0544, 108.2022),  # Đà Nẵng
    ("Nam", 10.7626, 106.6602),    # TP.HCM
    ("ConDao", 8.6833, 106.6167),  # Côn Đảo — cực Nam
]


def resolve_start_time(ts, start_time: str, sats):
    """Mốc thời gian của snapshot đầu tiên.

    Chép nguyên từ `oracle_labeler_sagsin.py` (12 dòng) thay vì import chéo:
    `dataset/` không phải package nên `from dataset.oracle_labeler_sagsin ...`
    ném `ModuleNotFoundError`. Dùng đúng logic đó để mốc t0 ở đây TRÙNG KHÍT
    với mốc labeler sẽ dùng — lệch mốc là bảng độ phủ vô nghĩa.
    """
    if start_time == "now":
        return ts.now()
    if start_time == "tle":
        epochs = [float(sat.epoch.tt) for sat in sats if hasattr(sat, "epoch")]
        if epochs:
            return ts.tt_jd(float(np.median(epochs)))
        print("!! Cảnh báo: vệ tinh không có thuộc tính epoch, dùng thời điểm hiện tại.")
        return ts.now()
    start_dt = datetime.fromisoformat(start_time)
    return ts.utc(start_dt.year, start_dt.month, start_dt.day,
                  start_dt.hour, start_dt.minute, start_dt.second)


def _doc_tle_theo_ten(duong: Path) -> dict[str, tuple[str, str, str]]:
    """Đọc NGUYÊN VĂN tệp TLE -> `{tên: (tên, dòng 1, dòng 2)}`.

    Cần vì `sgp4.Satrec` không giữ lại hai dòng gốc: muốn tách tập con mà
    không phá checksum hay mất chữ số của trường kiểu "3.27350-4" thì phải
    chép lại từng ký tự y như tệp nguồn.
    """
    ra: dict[str, tuple[str, str, str]] = {}
    with duong.open("r", encoding="utf-8") as f:
        dong = [d.rstrip("\r\n") for d in f]
    i = 0
    while i + 2 < len(dong):
        ten = dong[i].strip()
        if not ten:
            i += 1
            continue
        if ten.startswith("0 ") or ten.startswith("1 ") or ten.startswith("2 "):
            i += 1
            continue
        d1, d2 = dong[i + 1], dong[i + 2]
        if d1.startswith("1 ") and d2.startswith("2 "):
            ra[ten] = (ten, d1, d2)
        i += 3
    return ra


def tinh_goc_nang(sats, mocs, diem_vn=DIEM_VN) -> np.ndarray:
    """Ma trận `[S, Nsat]` góc nâng (độ) lớn nhất trên các điểm đại diện.

    S = số mốc thời gian (snapshot), Nsat = số vệ tinh. Tính lặp từng vệ tinh
    vì `(sat - diem).at(mocs)` đã vector hoá theo thời gian, còn vector hoá
    thêm theo vệ tinh sẽ phải dựng 6 000 `EarthSatellite` cùng lúc.
    """
    max_alt = np.full((len(mocs), len(sats)), -90.0, dtype=np.float64)
    for _, la, lo in diem_vn:
        diem = wgs84.latlon(la, lo)
        for i, sat in enumerate(sats):
            alt, _, _ = (sat - diem).at(mocs).altaz()
            max_alt[:, i] = np.maximum(max_alt[:, i], alt.degrees)
    return max_alt



def chon_tham_lam(diem_phu: np.ndarray, nhom: np.ndarray, n_take: int,
                  tran: float | None) -> np.ndarray:
    """Chọn `n_take` vệ tinh có điểm phủ cao nhất, mỗi nhóm quỹ đạo tối đa
    `tran · n_take` vệ tinh (`tran=None` = không giới hạn).

    Duyệt theo điểm phủ giảm dần, bỏ qua vệ tinh thuộc nhóm đã đầy. Không cần
    tối ưu tổ hợp: điểm phủ của từng vệ tinh gần như độc lập (mỗi con phủ Việt
    Nam ~7 % thời gian, chồng lấn ít), nên tham lam đã sát tối ưu.
    """
    cap = n_take if tran is None else int(n_take * tran)
    dem: Counter = Counter()
    chon: list[int] = []
    for i in np.argsort(-diem_phu):
        g = int(nhom[i])
        if dem[g] >= cap:
            continue
        chon.append(int(i))
        dem[g] += 1
        if len(chon) == n_take:
            break
    if len(chon) < n_take:
        raise ValueError(
            f"Chỉ chọn được {len(chon)}/{n_take} vệ tinh với trần {tran:.0%}. "
            f"Nới `--tran-nghieng` hoặc tăng số vệ tinh nguồn.")
    return np.array(chon)


def main():
    p = argparse.ArgumentParser(
        description="Lọc TLE Starlink thật -> N vệ tinh phủ Việt Nam bền vững")
    p.add_argument("--tle", default="data/tle/starlink.txt", help="TLE nguồn")
    p.add_argument("--out", default="data/tle/starlink_vn150.txt", help="TLE đích")
    p.add_argument("--so-ve-tinh", type=int, default=150, help="số vệ tinh cần giữ")
    p.add_argument("--snapshots", type=int, default=600,
                   help="số snapshot của dataset sẽ sinh (để tính độ phủ)")
    p.add_argument("--buoc-giay", type=int, default=20, help="bước thời gian (giây)")
    p.add_argument("--min-elev", type=float, default=15.0,
                   help="ngưỡng góc nâng để coi là nối được (khớp --min-elev của labeler)")
    p.add_argument("--tran-nghieng", type=float, default=0.55,
                   help="tỉ lệ tối đa cho mỗi nhóm độ nghiêng; 0 = không giới hạn")
    a = p.parse_args()

    duong_nguon = Path(a.tle) if Path(a.tle).is_absolute() else GOC / a.tle
    ts = load.timescale()
    sats_raw = load_tle(duong_nguon, loc_hong=False)
    if not sats_raw:
        raise ValueError(f"Tệp TLE rỗng: {duong_nguon}")
    t0 = resolve_start_time(ts, "tle", sats_raw)

    # Lọc vệ tinh đã rơi/hỏng trên TOÀN BỘ khoảng thời gian dataset — giống hệt
    # labeler, để tập chọn ra không chứa con nào mà labeler sẽ loại bỏ sau.
    sats = load_tle(duong_nguon, luc=t0,
                    span_giay=(a.snapshots - 1) * a.buoc_giay,
                    buoc_giay=max(a.buoc_giay * 20, 600))
    print(f"Tệp nguồn : {duong_nguon.relative_to(GOC)} ({len(sats_raw)} vệ tinh)")
    print(f"Sau lọc   : {len(sats)} vệ tinh còn tốt trên {a.snapshots} snapshot")
    print(f"Mốc đầu   : {t0.utc_datetime()}")

    if a.so_ve_tinh > len(sats):
        raise ValueError(f"Cần {a.so_ve_tinh} nhưng chỉ có {len(sats)} vệ tinh tốt")

    mocs = ts.tt_jd(t0.tt + np.arange(a.snapshots) * a.buoc_giay / 86400.0)
    print("Đang tính góc nâng tới Việt Nam ...", flush=True)
    max_alt = tinh_goc_nang(sats, mocs)

    noi_duoc = max_alt >= a.min_elev
    diem_phu = noi_duoc.sum(0)
    inc = np.array([round(float(s.model.inclo) * RAD) for s in sats])

    tran = None if a.tran_nghieng <= 0 else a.tran_nghieng

    # ==================================================================
    # CHỌN Ở ĐÚNG MỐC THỜI GIAN MÀ LABELER SẼ DÙNG.
    #
    # BẪY LỚN NHẤT CỦA CẢ SCRIPT: `oracle_labeler_sagsin.py` dòng 200 gọi
    #     t_start = resolve_start_time(ts, "tle", sample_sats)
    # tức median epoch của CHÍNH tập vệ tinh được truyền vào (tệp đã lọc), KHÔNG
    # phải của toàn tệp nguồn. Tối ưu ở mốc toàn tệp rồi để labeler chạy ở mốc
    # khác là tối ưu vô nghĩa. Đo được trên cùng một tập 150 vệ tinh:
    #   - đánh giá ở mốc toàn tệp (10 728 epoch): 9.50 cặp/snapshot
    #   - đánh giá ở mốc labeler (median của chính 150 con): 0.69 cặp/snapshot
    # Lệch tới vậy vì LEO 550 km quay 15.3 vòng/ngày; lệch mốc vài giờ là gần
    # 4 vòng, toàn bộ hình học đổi hết.
    #
    # `t_start` lại PHỤ THUỘC chính tập được chọn (median của nó) nên không thể
    # "tính đúng mốc rồi chọn" một lần. Lặp cũng không hội tụ: đo được vòng 2
    # nhảy mốc -5.67 h -> +1.30 h, mỗi lần chọn lại làm median dịch tiếp.
    #
    # LỐI RA: chọn vệ tinh **cùng một epoch đến giây**. Khi mọi epoch của tập
    # trùng nhau (sai khác <1 s) thì median của tập = đúng epoch đó, BẤT KỂ chọn
    # con nào => mốc labeler CỐ ĐỊNH, tính trước được, không phụ thuộc kết quả
    # chọn. Tệp TLE thật có 585 vệ tinh cùng epoch `04:00:01.816`, đủ cho 150.
    # Nhóm epoch lớn nhất chỉ đạt 7.21 cặp/snapshot (so với 9.50 khi được trộn
    # mọi epoch) — đánh đổi này ĐÁNG, vì 9.50 kia là con số không bao giờ xuất
    # hiện trong dataset thật.
    # ==================================================================
    ep = np.array([float(s.epoch.tt) for s in sats])
    giay = np.round(ep * 86400).astype(np.int64)
    u, dem = np.unique(giay, return_counts=True)
    i_nhom = int(np.argmax(dem))
    if dem[i_nhom] < a.so_ve_tinh:
        raise ValueError(
            f"Nhóm epoch lớn nhất chỉ có {dem[i_nhom]} vệ tinh, không đủ "
            f"{a.so_ve_tinh}. Giảm --so-ve-tinh hoặc dùng --cho-phep-lech-epoch")
    nhom = np.where(giay == u[i_nhom])[0]
    # Mốc mà labeler SẼ dùng với tập con của nhóm này — chính là epoch chung.
    t_moc = ts.tt_jd(float(np.median(ep[nhom])))
    print(f"Nhóm epoch  : {len(nhom)} vệ tinh cùng epoch {t_moc.utc_datetime()} "
          f"(= mốc labeler sẽ dùng)")
    if abs(t_moc.tt - t0.tt) * 24 > 1.0:
        print(f"  !! mốc này lệch {(t_moc.tt - t0.tt) * 24:+.2f} giờ so với median "
              f"toàn tệp — đó chính là lý do phải chọn trong nhóm epoch")

    mocs_moc = ts.tt_jd(t_moc.tt + np.arange(a.snapshots) * a.buoc_giay / 86400.0)
    print("Đang tính góc nâng tại mốc labeler ...", flush=True)
    nd_nhom = tinh_goc_nang([sats[i] for i in nhom], mocs_moc) >= a.min_elev

    chon_trong_nhom = chon_tham_lam(nd_nhom.sum(0), inc[nhom],
                                    a.so_ve_tinh, tran)
    idx = nhom[chon_trong_nhom]

    # Xác nhận mốc mà labeler THẬT SỰ sẽ dùng với tệp vừa chọn — phải bằng t_moc.
    t_lab = resolve_start_time(ts, "tle", [sats[i] for i in idx])
    lech_s = abs(t_lab.tt - t_moc.tt) * 86400
    assert lech_s < 1.0, (
        f"mốc labeler ({t_lab.utc_datetime()}) lệch {lech_s:.1f} s so với mốc đã "
        f"tối ưu ({t_moc.utc_datetime()}) — nhóm epoch không còn đồng nhất")
    n_lab = (tinh_goc_nang([sats[i] for i in idx], mocs_moc) >= a.min_elev).sum(1)

    print(f"\nĐã chọn {len(idx)} vệ tinh "
          f"(trần nghiêng: {'không' if tran is None else f'{tran:.0%}'})")
    print(f"  mốc labeler dùng : {t_lab.utc_datetime()} "
          f"(lệch {(t_lab.tt - t0.tt) * 24:+.2f} giờ so với median toàn tệp)")
    print(f"  nối được TB      : {n_lab.mean():.2f} vệ tinh/snapshot")
    print(f"  snapshot có ≥1   : {(n_lab >= 1).mean() * 100:.1f} %")
    print(f"  snapshot có ≥5   : {(n_lab >= 5).mean() * 100:.1f} %")
    print(f"  snapshot trống   : {(n_lab == 0).sum()} / {a.snapshots}")
    print(f"  độ nghiêng       : {dict(sorted(Counter(inc[idx]).items()))}")

    # Ghi ra tệp theo đúng định dạng 3 dòng/vệ tinh mà `load.tle_file` đọc được.
    # KHÔNG dựng lại dòng TLE từ `Satrec`: `sgp4.Satrec` không giữ `line1`/
    # `line2`, và dựng lại từ các phần tử số sẽ làm sai checksum + mất độ chính
    # xác của trường dạng "3.27350-4". Đọc NGUYÊN VĂN từ tệp nguồn theo tên.
    dong_tle = _doc_tle_theo_ten(duong_nguon)
    thieu = [sats[i].name for i in idx if sats[i].name not in dong_tle]
    if thieu:
        raise ValueError(f"Không tìm thấy dòng TLE gốc cho: {thieu[:5]}")

    duong_dich = Path(a.out) if Path(a.out).is_absolute() else GOC / a.out
    duong_dich.parent.mkdir(parents=True, exist_ok=True)
    with duong_dich.open("w", encoding="utf-8", newline="\n") as f:
        for i in idx:
            ten, d1, d2 = dong_tle[sats[i].name]
            f.write(f"{ten}\n{d1}\n{d2}\n")

    tuong_doi = duong_dich.relative_to(GOC) if duong_dich.is_relative_to(GOC) else duong_dich
    print(f"\nĐã ghi {len(idx)} vệ tinh -> {tuong_doi}")
    print(f"  sinh dataset: python dataset/oracle_labeler_sagsin.py --mo-rong "
          f"--sats {len(idx)} --sat-select first --tle {tuong_doi} --out <đích>")


if __name__ == "__main__":
    main()
