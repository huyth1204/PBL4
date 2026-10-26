"""
sinh_tle_walker.py — sinh tệp TLE chòm Walker-Delta đồng nhất trên MỘT vỏ quỹ đạo.

Vì sao cần: `data/tle/starlink.txt` là TLE THẬT nhưng trộn nhiều vỏ và nhiều thế
hệ (550 / 540 / 570 km, nhiều mặt phẳng nghiêng khác nhau) và có vệ tinh đã rơi.
Dùng nó cho kết quả phụ thuộc "hôm đó tải về những con nào" — tái lập được nhưng
không mô tả được một thiết kế chòm có chủ đích.

Tệp sinh ra KHÔNG phải dữ liệu thật. Mọi kết quả dùng nó phải ghi rõ là
"chòm Walker lý tưởng", không được trộn lẫn với số liệu TLE Starlink thật.

Mô hình Walker-Delta (i: T/P/F):
    * T vệ tinh chia đều thành P mặt phẳng, mỗi mặt phẳng T/P vệ tinh;
    * cùng độ nghiêng i, cùng độ cao h (quỹ đạo tròn);
    * RAAN mặt phẳng p = p · 360°/P;
    * vị trí trong mặt phẳng: vệ tinh thứ s của mặt phẳng p lệch pha
      θ = s · 360°/(T/P) + p · F · 360°/T.
Đây là định nghĩa chuẩn, nên tệp kiểm chứng được bằng công thức độc lập
(`test/test_chom_walker.py` làm đúng việc đó).

Cách dùng:
    python dataset/sinh_tle_walker.py                       # 256/16/1 @ 550 km
    python dataset/sinh_tle_walker.py --tong 256 --mat-phang 16 --he-so-f 1 \\
        --cao-km 550 --nghieng-deg 53 --out data/tle/walker256.txt

Sau khi có tệp:
    python dataset/oracle_labeler_sagsin.py --mo-rong --sats 256 \\
        --tle data/tle/walker256.txt --out dataset/oracle_dataset_v4.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Hằng số Trái Đất dùng cho việc đổi độ cao -> số vòng/ngày
# ---------------------------------------------------------------------------
MU_KM3_S2 = 398600.4418      # hằng số hấp dẫn Trái Đất, km^3/s^2
R_TRAI_DAT_KM = 6378.137      # bán kính xích đạo, km
NGAY_S = 86400.0


def chu_ky_phut(cao_km: float) -> float:
    """Chu kỳ quỹ đạo tròn ở độ cao `cao_km` (phút) — công thức Kepler thứ ba.

    T = 2π · sqrt(a³/μ),  a = R_E + h.
    """
    a = R_TRAI_DAT_KM + cao_km
    return 2.0 * 3.141592653589793 * (a ** 3 / MU_KM3_S2) ** 0.5 / 60.0


def so_vong_ngay(cao_km: float) -> float:
    """Số vòng quỹ đạo trong một ngày — chính là trường mean motion của TLE."""
    return NGAY_S / (chu_ky_phut(cao_km) * 60.0)


def _checksum(dong: str) -> str:
    """Checksum chuẩn TLE: tổng các chữ số + 1 cho mỗi dấu '-', lấy mod 10.

    68 ký tự đầu của mỗi dòng mới được tính; ký tự thứ 69 là chính checksum.
    `skyfield`/`sgp4` không kiểm checksum nên thiếu nó vẫn chạy, nhưng mọi công
    cụ TLE khác sẽ từ chối tệp — nên phải tính cho đúng.
    """
    tong = 0
    for ch in dong[:68]:
        if ch.isdigit():
            tong += int(ch)
        elif ch == "-":
            tong += 1
    return str(tong % 10)


def make_tle(
    tong: int = 256,
    mat_phang: int = 16,
    he_so_f: int = 1,
    cao_km: float = 550.0,
    nghieng_deg: float = 53.0,
    ten_goc: str = "SYN",
    epoch_nam: int = 26,
    epoch_ngay: float = 1.0,
) -> list[tuple[str, str, str]]:
    """Sinh danh sách (tên, dòng 1, dòng 2) của chòm Walker-Delta T/P/F.

    Tham số:
      tong       : tổng số vệ tinh T.
      mat_phang  : số mặt phẳng quỹ đạo P (T phải chia hết cho P).
      he_so_f    : hệ số dịch pha F của Walker (0 .. P-1).
      cao_km     : độ cao quỹ đạo tròn (km).
      nghieng_deg: độ nghiêng (độ).
      ten_goc    : tiền tố tên; tên đầy đủ "{ten_goc}-PP-SS" theo (mặt phẳng, slot).
      epoch_nam  : hai chữ số cuối của năm epoch.
      epoch_ngay : ngày trong năm dạng thực (1.0 = 1/1 00:00 UTC).

    Trả về list đã sắp theo (mặt phẳng, slot) — thứ tự ổn định để `--sat-select
    first` cắt ra một mẫu tất định.

    LƯU Ý VỀ ĐỘ CHÍNH XÁC (đo trên chính tệp sinh ra, không suy đoán):
    các phần tử ở đây là giá trị DANH NGHĨA. Khi propagate bằng SGP4 và đo
    TRONG HỆ TEME (hệ quy chiếu của chính phần tử TLE), vị trí tức thời lệch
    khỏi vị trí lý thuyết của công thức Walker trung bình **≈0.066° (tối đa
    0.099°, tức ~12 km)** vì SGP4 coi phần tử TLE là giá trị TRUNG BÌNH
    (Kozai) rồi cộng dao động short-period của J2. Đây là bản chất của mô
    hình lực, KHÔNG phải lỗi tệp — đừng "sửa" bằng cách bù góc vào đây.

    Nếu đo trong GCRS (`skyfield .position`) sẽ thấy 0.30° TB / 0.45° max
    (~55 km). Phần dôi ra ~0.23° KHÔNG phải sai số quỹ đạo mà là phép quay
    precession/nutation giữa TEME và GCRS. Đã xác nhận bằng cách so |r| của
    skyfield với sgp4-api: khớp tới 1e-11 km — cùng một quỹ đạo, chỉ khác hệ
    quy chiếu. Vì vậy test trong `test/test_chom_walker.py` đo trong TEME.
    """

    if mat_phang <= 0:
        raise ValueError(f"P phải > 0, đang {mat_phang}")
    if tong % mat_phang:
        raise ValueError(f"T={tong} không chia hết cho P={mat_phang}")
    if not 0 <= he_so_f < mat_phang:
        raise ValueError(f"F phải trong [0, {mat_phang}), đang {he_so_f}")

    moi_mp = tong // mat_phang
    n_vong = so_vong_ngay(cao_km)

    out: list[tuple[str, str, str]] = []
    so_thu_tu = 1
    for p in range(mat_phang):
        raan = (p * 360.0 / mat_phang) % 360.0
        for s in range(moi_mp):
            # Dịch pha trong mặt phẳng + dịch pha giữa các mặt phẳng (Walker F).
            ma_anomaly = (s * 360.0 / moi_mp + p * he_so_f * 360.0 / tong) % 360.0
            ten = f"{ten_goc}-{p:02d}-{s:02d}"

            # --- Dòng 1 (68 ký tự cố định rồi tới checksum).
            #  3-7  : số hiệu vệ tinh       8   : 'U' = unclassified
            # 10-17 : quốc tế hiệu (giả định "00000A" — không phải vệ tinh thật)
            # 19-20 : 2 chữ số cuối năm epoch, 21-32: ngày epoch (12 ký tự, 8 lẻ)
            # 34-43 : đạo hàm bậc 1 của n (0 = quỹ đạo tròn lý tưởng, không suy giảm)
            d1 = (f"1 {so_thu_tu:05d}U 00000A   {epoch_nam:02d}"
                  f"{epoch_ngay:012.8f}  .00000000  00000-0  00000-0 0  01")
            d1 = d1[:68].ljust(68)
            d1 += _checksum(d1)

            # --- Dòng 2 (68 ký tự cố định rồi tới checksum).
            #  9-16 : độ nghiêng (8 ký tự, 4 lẻ)     18-25: RAAN
            # 27-33 : độ lệch tâm (0 = tròn)         35-42: arg. of perigee (vô nghĩa khi e=0)
            # 44-51 : mean anomaly                   53-63: mean motion (11 ký tự, 8 lẻ)
            # 64-68 : số vòng đã bay (0)
            d2 = (f"2 {so_thu_tu:05d} {nghieng_deg:8.4f} {raan:8.4f} 0000000 "
                  f"  0.0000 {ma_anomaly:8.4f} {n_vong:11.8f}    0")
            d2 = d2[:68].ljust(68)
            d2 += _checksum(d2)

            out.append((ten, d1, d2))
            so_thu_tu += 1
    return out


def ghi_tle(bo: list[tuple[str, str, str]], duong: Path) -> None:
    """Ghi danh sách (tên, d1, d2) ra tệp 3 dòng/vệ tinh, đúng định dạng `load.tle_file`."""
    duong.parent.mkdir(parents=True, exist_ok=True)
    with duong.open("w", encoding="utf-8", newline="\n") as f:
        for ten, d1, d2 in bo:
            f.write(f"{ten}\n{d1}\n{d2}\n")


def main():
    p = argparse.ArgumentParser(description="Sinh TLE chòm Walker-Delta đồng nhất")
    p.add_argument("--tong", type=int, default=256, help="tổng số vệ tinh T")
    p.add_argument("--mat-phang", type=int, default=16, help="số mặt phẳng P")
    p.add_argument("--he-so-f", type=int, default=1, help="hệ số Walker F")
    p.add_argument("--cao-km", type=float, default=550.0)
    p.add_argument("--nghieng-deg", type=float, default=53.0)
    p.add_argument("--ten-goc", type=str, default="SYN")
    p.add_argument("--epoch-nam", type=int, default=26, help="2 chữ số cuối của năm epoch")
    p.add_argument("--epoch-ngay", type=float, default=1.0, help="ngày trong năm (1.0 = 1/1)")
    p.add_argument("--out", type=str, default="data/tle/walker256.txt")
    a = p.parse_args()

    bo = make_tle(a.tong, a.mat_phang, a.he_so_f, a.cao_km, a.nghieng_deg,
                  a.ten_goc, a.epoch_nam, a.epoch_ngay)
    duong = Path(a.out) if Path(a.out).is_absolute() else GOC / a.out
    ghi_tle(bo, duong)

    print(f"Đã ghi {len(bo)} vệ tinh -> {duong.relative_to(GOC) if duong.is_relative_to(GOC) else duong}")
    print(f"  Walker {a.tong}/{a.mat_phang}/{a.he_so_f}"
          f"  |  cao {a.cao_km:.2f} km"
          f"  |  nghiêng {a.nghieng_deg:.2f}°"
          f"  |  mỗi mặt phẳng {a.tong // a.mat_phang} vệ tinh")
    print(f"  Chu kỳ {chu_ky_phut(a.cao_km):.3f} phút  |  {so_vong_ngay(a.cao_km):.8f} vòng/ngày")
    print("\n  3 dòng đầu của tệp:")
    for dong in bo[0]:
        print(f"    {dong}")


if __name__ == "__main__":
    main()



