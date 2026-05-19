import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import os

# =============================================================
# KONFIGURASI
# =============================================================
NAMA_FOTO      = "foto_kakey.png"   # ganti sesuai nama file foto kamu
BLOCK_SIZE     = 8                   # ukuran blok DCT (standar JPEG = 8x8)
WATERMARK_TILE = 32                  # ukuran kotak watermark biner
QUALITY_FACTORS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
THRESHOLD_BER  = 0.1                 # batas BER untuk dianggap berhasil
THRESHOLD_NC   = 0.9                 # batas NC untuk dianggap berhasil

# =============================================================
# BAGIAN 1: LOAD & PERSIAPAN GAMBAR
# =============================================================
print("=" * 60)
print("LOAD GAMBAR")
print("=" * 60)

img = Image.open(NAMA_FOTO).convert("L")   # grayscale
img_array = np.array(img, dtype=np.float64)
H, W = img_array.shape
print(f"Gambar dimuat : {NAMA_FOTO}")
print(f"Ukuran        : {W} x {H} piksel")
print(f"Tipe data     : {img_array.dtype}")

# =============================================================
# BAGIAN 2: DCT (Discrete Cosine Transform)
# =============================================================
print("\n" + "=" * 60)
print("DCT (Discrete Cosine Transform)")
print("=" * 60)

def dct_2d(block):
    N = block.shape[0]
    result = np.zeros_like(block)
    for u in range(N):
        for v in range(N):
            cu = (1/np.sqrt(2)) if u == 0 else 1.0
            cv = (1/np.sqrt(2)) if v == 0 else 1.0
            total = 0
            for x in range(N):
                for y in range(N):
                    total += block[x, y] * \
                             np.cos((2*x+1)*u*np.pi/(2*N)) * \
                             np.cos((2*y+1)*v*np.pi/(2*N))
            result[u, v] = (2/N) * cu * cv * total
    return result

def idct_2d(block):
    """Inverse DCT 2D manual"""
    N = block.shape[0]
    result = np.zeros_like(block)
    for x in range(N):
        for y in range(N):
            total = 0
            for u in range(N):
                for v in range(N):
                    cu = (1/np.sqrt(2)) if u == 0 else 1.0
                    cv = (1/np.sqrt(2)) if v == 0 else 1.0
                    total += cu * cv * block[u, v] * \
                             np.cos((2*x+1)*u*np.pi/(2*N)) * \
                             np.cos((2*y+1)*v*np.pi/(2*N))
            result[x, y] = (1/N) * total
    return result

def dct_2d_fast(block):
    from scipy.fft import dctn
    return dctn(block, norm='ortho')

def idct_2d_fast(block):
    from scipy.fft import idctn
    return idctn(block, norm='ortho')

def apply_dct_blocks(image, block_size=8):
    H, W = image.shape
    dct_image = np.zeros_like(image)
    for i in range(0, H - H % block_size, block_size):
        for j in range(0, W - W % block_size, block_size):
            blok = image[i:i+block_size, j:j+block_size]
            dct_image[i:i+block_size, j:j+block_size] = dct_2d_fast(blok)
    return dct_image

def apply_idct_blocks(dct_image, block_size=8):
    """Terapkan Inverse DCT pada seluruh gambar per blok 8x8"""
    H, W = dct_image.shape
    result = np.zeros_like(dct_image)
    for i in range(0, H - H % block_size, block_size):
        for j in range(0, W - W % block_size, block_size):
            blok = dct_image[i:i+block_size, j:j+block_size]
            result[i:i+block_size, j:j+block_size] = idct_2d_fast(blok)
    return result

# Hitung DCT pada blok pertama sebagai contoh demonstrasi
blok_contoh = img_array[:BLOCK_SIZE, :BLOCK_SIZE]
dct_contoh  = dct_2d_fast(blok_contoh)

print(f"Blok piksel asli (8x8 pertama):")
print(np.round(blok_contoh).astype(int))
print(f"\nHasil DCT blok tersebut:")
print(np.round(dct_contoh, 1))
print(f"\nKoefisien DC (rata-rata blok) : {dct_contoh[0,0]:.2f}")
print(f"Total koefisien AC            : {BLOCK_SIZE*BLOCK_SIZE - 1}")

# DCT seluruh gambar
print(f"\nMenghitung DCT seluruh gambar ({H}x{W})...")
dct_full = apply_dct_blocks(img_array, BLOCK_SIZE)
print("DCT selesai!")

# =============================================================
# BAGIAN 3: KUANTISASI
# =============================================================
print("\n" + "=" * 60)
print("KUANTISASI")
print("=" * 60)

# Tabel kuantisasi standar JPEG untuk luminance (grayscale)
TABEL_KUANTISASI_STANDAR = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68,  109, 103, 77],
    [24, 35, 55, 64, 81,  104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float64)

def buat_tabel_kuantisasi(qf):
    if qf <= 0:   qf = 1
    if qf >= 100: qf = 100
    if qf < 50:
        skala = 5000 / qf
    else:
        skala = 200 - 2 * qf
    tabel = np.floor((TABEL_KUANTISASI_STANDAR * skala + 50) / 100)
    tabel = np.clip(tabel, 1, 255)
    return tabel

def kuantisasi(dct_block, tabel_q):
    return np.round(dct_block / tabel_q)

def dekuantisasi(q_block, tabel_q):
    return q_block * tabel_q

# Demonstrasi kuantisasi pada blok pertama
for qf_demo in [10, 50, 100]:
    tabel_q = buat_tabel_kuantisasi(qf_demo)
    q_blok  = kuantisasi(dct_contoh, tabel_q)
    print(f"\nTabel kuantisasi QF={qf_demo} (baris pertama): {tabel_q[0].astype(int)}")
    print(f"Koefisien DC setelah kuantisasi QF={qf_demo} : {q_blok[0,0]:.1f}")

# =============================================================
# BAGIAN 4: KOMPRESI JPEG MANUAL (simulasi)
# =============================================================
print("\n" + "=" * 60)
print("SIMULASI KOMPRESI JPEG")
print("=" * 60)

def kompresi_jpeg_manual(image, qf, block_size=8):
    H, W = image.shape
    tabel_q   = buat_tabel_kuantisasi(qf)
    hasil     = np.zeros_like(image)

    for i in range(0, H - H % block_size, block_size):
        for j in range(0, W - W % block_size, block_size):
            blok    = image[i:i+block_size, j:j+block_size]
            dct_blk = dct_2d_fast(blok)
            q_blk   = kuantisasi(dct_blk, tabel_q)
            dq_blk  = dekuantisasi(q_blk, tabel_q)
            hasil[i:i+block_size, j:j+block_size] = idct_2d_fast(dq_blk)

    return np.clip(hasil, 0, 255)

def kompresi_jpeg_pil(image_array, qf):
    img_pil = Image.fromarray(image_array.astype(np.uint8), mode="L")
    buffer  = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=qf)
    buffer.seek(0)
    return np.array(Image.open(buffer).convert("L"), dtype=np.uint8)

# Kompres gambar asli dengan beberapa QF untuk demonstrasi
print("Mengkompresi gambar asli (demonstrasi)...")
contoh_kompres = {}
for qf in [10, 50, 100]:
    contoh_kompres[qf] = kompresi_jpeg_pil(img_array.astype(np.uint8), qf)
    psnr = 10 * np.log10(255**2 / np.mean((img_array - contoh_kompres[qf])**2) + 1e-10)
    print(f"  QF {qf:3d} → PSNR = {psnr:.2f} dB")

# =============================================================
# BAGIAN 5: WATERMARKING (EMBED)
# =============================================================
print("\n" + "=" * 60)
print("EMBED WATERMARK (LSB)")
print("=" * 60)

# Buat watermark biner (pola kotak-kotak)
watermark = np.zeros((H, W), dtype=np.uint8)
for i in range(0, H, WATERMARK_TILE):
    for j in range(0, W, WATERMARK_TILE):
        if (i // WATERMARK_TILE + j // WATERMARK_TILE) % 2 == 0:
            watermark[i:i+WATERMARK_TILE, j:j+WATERMARK_TILE] = 1

print(f"Watermark biner dibuat ({WATERMARK_TILE}x{WATERMARK_TILE} kotak)")
print(f"Jumlah bit '1' : {np.sum(watermark == 1):,}")
print(f"Jumlah bit '0' : {np.sum(watermark == 0):,}")

def embed_watermark(image_array, wm):
    img_uint8   = image_array.astype(np.uint8)
    watermarked = (img_uint8 & 0xFE) | wm
    return watermarked

img_uint8        = img_array.astype(np.uint8)
watermarked_array = embed_watermark(img_uint8, watermark)

# Simpan gambar ter-watermark
Image.fromarray(watermarked_array, mode="L").save("watermarked_original.png")

# Hitung perbedaan sebelum dan sesudah watermark
mse_wm   = np.mean((img_uint8.astype(float) - watermarked_array.astype(float))**2)
psnr_wm  = 10 * np.log10(255**2 / (mse_wm + 1e-10))
print(f"\nGambar ter-watermark disimpan: watermarked_original.png")
print(f"MSE antara asli vs watermarked : {mse_wm:.4f}")
print(f"PSNR                           : {psnr_wm:.2f} dB  (makin tinggi makin bagus)")
print(f"(Tidak terlihat bedanya karena perubahan max 1 nilai piksel)")

# =============================================================
# BAGIAN 6: KOMPRESI + EKSTRAKSI + EVALUASI
# =============================================================
print("\n" + "=" * 60)
print("KOMPRESI → EKSTRAKSI → EVALUASI")
print("=" * 60)

def extract_watermark(image_array):
    return image_array.astype(np.uint8) & 1

def hitung_ber(wm_asli, wm_ekstrak):
    return np.sum(wm_asli != wm_ekstrak) / wm_asli.size

def hitung_nc(wm_asli, wm_ekstrak):
    orig = wm_asli.astype(float)
    extr = wm_ekstrak.astype(float)
    nc   = np.sum(orig * extr) / (
           np.sqrt(np.sum(orig**2)) * np.sqrt(np.sum(extr**2)) + 1e-10)
    return nc

def hitung_psnr(asli, hasil):
    mse = np.mean((asli.astype(float) - hasil.astype(float))**2)
    return 10 * np.log10(255**2 / (mse + 1e-10))

# Evaluasi semua QF
print(f"\n{'QF':<6} {'BER':<10} {'NC':<10} {'PSNR(dB)':<12} {'Status'}")
print("-" * 55)

results          = []
compressed_images = {}

for qf in QUALITY_FACTORS:
    # Kompres gambar ter-watermark
    compressed = kompresi_jpeg_pil(watermarked_array, qf)
    compressed_images[qf] = compressed

    # Ekstrak watermark dari hasil kompresi
    wm_ekstrak = extract_watermark(compressed)

    # Hitung metrik
    ber  = hitung_ber(watermark, wm_ekstrak)
    nc   = hitung_nc(watermark, wm_ekstrak)
    psnr = hitung_psnr(img_uint8, compressed)

    # Tentukan status
    if ber <= THRESHOLD_BER and nc >= THRESHOLD_NC:
        status = "Berhasil"
    else:
        status = "Gagal"

    print(f"QF {qf:<4} BER={ber:.4f}  NC={nc:.4f}  PSNR={psnr:.2f}     {status}")
    results.append({
        "qf": qf, "ber": ber, "nc": nc,
        "psnr": psnr, "status": status
    })

# Temukan threshold QF
qf_berhasil = [r["qf"] for r in results if "Berhasil" in r["status"]]
qf_gagal    = [r["qf"] for r in results if "GAGAL" in r["status"]]

print(f"\n{'='*55}")
print(f"KESIMPULAN:")
print(f"  QF yang GAGAL diekstrak  : {qf_gagal}")
print(f"  QF yang BERHASIL         : {qf_berhasil}")
if qf_gagal:
    print(f"  → Watermark tidak dapat diekstrak pada QF ≤ {max(qf_gagal)}")

# =============================================================
# BAGIAN 7: VISUALISASI LENGKAP
# =============================================================
print("\n" + "=" * 60)
print("VISUALISASI")
print("=" * 60)

qf_vals   = [r["qf"]   for r in results]
ber_vals  = [r["ber"]  for r in results]
nc_vals   = [r["nc"]   for r in results]
psnr_vals = [r["psnr"] for r in results]

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Analisis Watermarking LSB dengan Kompresi JPEG", 
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Gambar-gambar
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(img_uint8, cmap='gray', vmin=0, vmax=255)
ax1.set_title("1. Gambar Asli", fontweight='bold')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(watermark, cmap='gray')
ax2.set_title("2. Watermark Biner", fontweight='bold')
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(watermarked_array, cmap='gray', vmin=0, vmax=255)
ax3.set_title("3. Gambar + Watermark\n(tidak terlihat bedanya)", fontweight='bold')
ax3.axis('off')

ax4 = fig.add_subplot(gs[0, 3])
diff = np.abs(img_uint8.astype(int) - watermarked_array.astype(int))
ax4.imshow(diff * 100, cmap='hot')  # dikali 100 biar terlihat
ax4.set_title("4. Perbedaan × 100\n(asli vs watermarked)", fontweight='bold')
ax4.axis('off')

# DCT & Kuantisasi
ax5 = fig.add_subplot(gs[1, 0])
ax5.imshow(np.log(np.abs(dct_full) + 1), cmap='viridis')
ax5.set_title("5. Koefisien DCT\n(skala log)", fontweight='bold')
ax5.axis('off')

ax6 = fig.add_subplot(gs[1, 1])
tabel_10  = buat_tabel_kuantisasi(10)
tabel_100 = buat_tabel_kuantisasi(100)
x = np.arange(8)
ax6.bar(x - 0.2, tabel_10[0],  0.4, label='QF=10',  color='red',  alpha=0.7)
ax6.bar(x + 0.2, tabel_100[0], 0.4, label='QF=100', color='blue', alpha=0.7)
ax6.set_title("6. Tabel Kuantisasi\n(baris pertama)", fontweight='bold')
ax6.set_xlabel("Koefisien DCT")
ax6.set_ylabel("Nilai Kuantisasi")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

ax7 = fig.add_subplot(gs[1, 2])
ax7.imshow(compressed_images[10], cmap='gray', vmin=0, vmax=255)
ax7.set_title("7. Gambar Dikompres\nQF=10 (kualitas rendah)", fontweight='bold')
ax7.axis('off')

ax8 = fig.add_subplot(gs[1, 3])
ax8.imshow(compressed_images[100], cmap='gray', vmin=0, vmax=255)
ax8.set_title("8. Gambar Dikompres\nQF=100 (kualitas tinggi)", fontweight='bold')
ax8.axis('off')

# Grafik Evaluasi
ax9 = fig.add_subplot(gs[2, 0])
colors = ['green' if r["status"] == "✅ Berhasil" else 'red' for r in results]
bars = ax9.bar(qf_vals, ber_vals, color=colors, alpha=0.8, edgecolor='black', width=7)
ax9.axhline(y=THRESHOLD_BER, color='black', linestyle='--', linewidth=1.5,
            label=f'Threshold BER={THRESHOLD_BER}')
ax9.set_title("9. BER vs Quality Factor", fontweight='bold')
ax9.set_xlabel("Quality Factor (QF)")
ax9.set_ylabel("Bit Error Rate (BER)")
ax9.set_xticks(qf_vals)
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3, axis='y')

ax10 = fig.add_subplot(gs[2, 1])
ax10.bar(qf_vals, nc_vals, color=colors, alpha=0.8, edgecolor='black', width=7)
ax10.axhline(y=THRESHOLD_NC, color='black', linestyle='--', linewidth=1.5,
             label=f'Threshold NC={THRESHOLD_NC}')
ax10.set_title("10. NC vs Quality Factor", fontweight='bold')
ax10.set_xlabel("Quality Factor (QF)")
ax10.set_ylabel("Normalized Correlation (NC)")
ax10.set_xticks(qf_vals)
ax10.legend(fontsize=8)
ax10.grid(True, alpha=0.3, axis='y')

ax11 = fig.add_subplot(gs[2, 2])
ax11.plot(qf_vals, psnr_vals, 'purple', marker='o', linewidth=2)
ax11.set_title("11. PSNR vs Quality Factor\n(kualitas gambar)", fontweight='bold')
ax11.set_xlabel("Quality Factor (QF)")
ax11.set_ylabel("PSNR (dB)")
ax11.set_xticks(qf_vals)
ax11.grid(True, alpha=0.3)

ax12 = fig.add_subplot(gs[2, 3])
wm_ekstrak_qf10  = extract_watermark(compressed_images[10])
wm_ekstrak_qf100 = extract_watermark(compressed_images[100])
# Tunjukkan watermark yang diekstrak dari QF terendah dan QF 100
combined = np.hstack([wm_ekstrak_qf10[:H//2, :W//2],
                      wm_ekstrak_qf100[:H//2, :W//2]])
ax12.imshow(combined, cmap='gray')
ax12.set_title("12. Watermark Diekstrak\nKiri: QF=10 | Kanan: QF=100", fontweight='bold')
ax12.axis('off')
ax12.axvline(x=combined.shape[1]//2, color='yellow', linewidth=2)

# Tambahkan legend warna
from matplotlib.patches import Patch
legend_els = [Patch(fc='green', ec='black', label='Berhasil diekstrak'),
              Patch(fc='red',   ec='black', label='GAGAL diekstrak')]
fig.legend(handles=legend_els, loc='lower center', ncol=2,
           fontsize=11, bbox_to_anchor=(0.5, 0.01),
           frameon=True, edgecolor='black')

plt.savefig("hasil_watermarking_lengkap.png", dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Grafik disimpan: hasil_watermarking_lengkap.png")

# =============================================================
# RINGKASAN AKHIR
# =============================================================
print("\n" + "=" * 60)
print("RINGKASAN AKHIR")
print("=" * 60)
print(f"Metode watermark : LSB (Least Significant Bit)")
print(f"Watermark        : Biner (pola kotak {WATERMARK_TILE}x{WATERMARK_TILE})")
print(f"Ukuran gambar    : {W}x{H} piksel")
print(f"File output      :")
print(f"  - watermarked_original.png  (gambar + watermark)")
print(f"  - hasil_watermarking_lengkap.png  (grafik evaluasi)")
print(f"\nHasil evaluasi:")
for r in results:
    print(f"  QF {r['qf']:>3} | BER={r['ber']:.4f} | NC={r['nc']:.4f} | "
          f"PSNR={r['psnr']:.1f}dB | {r['status']}")
if qf_gagal:
    print(f"\n→ Watermark tidak dapat diekstrak pada QF ≤ {max(qf_gagal)}")
if qf_berhasil:
    print(f"→ Watermark dapat diekstrak pada QF ≥ {min(qf_berhasil)}")