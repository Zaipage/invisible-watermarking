import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import io
import os

# =============================================================
# KONFIGURASI
# =============================================================
NAMA_FOTO       = "foto_kakey.png"
BLOCK_SIZE      = 8
WATERMARK_TILE  = 32
QUALITY_FACTORS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
THRESHOLD_BER   = 0.1
THRESHOLD_NC    = 0.9

# =============================================================
# BAGIAN 1: LOAD & PERSIAPAN GAMBAR
# =============================================================
print("=" * 60)
print("BAGIAN 1: LOAD GAMBAR")
print("=" * 60)

img = Image.open(NAMA_FOTO).convert("L")
img_array = np.array(img, dtype=np.float64)
H, W = img_array.shape
print(f"Gambar dimuat : {NAMA_FOTO}")
print(f"Ukuran        : {W} x {H} piksel")
print(f"Tipe data     : {img_array.dtype}")

# --- Output Tahap 1: Gambar asli grayscale ---
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(img_array, cmap='gray', vmin=0, vmax=255)
ax.set_title(f"Gambar Asli (Grayscale)\n{W}x{H} piksel", fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig("tahap1_gambar_asli_grayscale.png", dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Output: tahap1_gambar_asli_grayscale.png")

# =============================================================
# BAGIAN 2: DCT (Discrete Cosine Transform)
# =============================================================
print("\n" + "=" * 60)
print("BAGIAN 2: DCT (Discrete Cosine Transform)")
print("=" * 60)

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

blok_contoh = img_array[:BLOCK_SIZE, :BLOCK_SIZE]
dct_contoh  = dct_2d_fast(blok_contoh)

print(f"Blok piksel asli (8x8 pertama):")
print(np.round(blok_contoh).astype(int))
print(f"\nHasil DCT blok tersebut:")
print(np.round(dct_contoh, 1))
print(f"\nKoefisien DC (rata-rata blok) : {dct_contoh[0,0]:.2f}")
print(f"Total koefisien AC            : {BLOCK_SIZE*BLOCK_SIZE - 1}")

print(f"\nMenghitung DCT seluruh gambar ({H}x{W})...")
dct_full = apply_dct_blocks(img_array, BLOCK_SIZE)
print("DCT selesai!")

# Visualisasi koefisien DCT seluruh gambar
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_array, cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Gambar Asli (Grayscale)", fontweight='bold')
axes[0].axis('off')

axes[1].imshow(np.log(np.abs(dct_full) + 1), cmap='viridis')
axes[1].set_title("Koefisien DCT Seluruh Gambar\n(skala logaritmik)", fontweight='bold')
axes[1].axis('off')
plt.suptitle("Hasil DCT per Blok 8×8", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("tahap2a_hasil_dct_gambar.png", dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Output: tahap2a_hasil_dct_gambar.png")

# Visualisasi blok 8x8 pertama (piksel vs DCT)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im0 = axes[0].imshow(blok_contoh, cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Blok 8×8 Pertama\n(nilai piksel asli)", fontweight='bold')
for i in range(BLOCK_SIZE):
    for j in range(BLOCK_SIZE):
        axes[0].text(j, i, str(int(blok_contoh[i, j])),
                     ha='center', va='center', fontsize=7,
                     color='white' if blok_contoh[i, j] < 128 else 'black')
axes[0].set_xticks(range(BLOCK_SIZE))
axes[0].set_yticks(range(BLOCK_SIZE))

im1 = axes[1].imshow(dct_contoh, cmap='RdBu_r')
axes[1].set_title("Blok 8×8 Pertama\n(koefisien DCT)", fontweight='bold')
for i in range(BLOCK_SIZE):
    for j in range(BLOCK_SIZE):
        axes[1].text(j, i, f"{dct_contoh[i,j]:.0f}",
                     ha='center', va='center', fontsize=6,
                     color='white' if abs(dct_contoh[i, j]) > np.max(np.abs(dct_contoh)) * 0.5 else 'black')
axes[1].set_xticks(range(BLOCK_SIZE))
axes[1].set_yticks(range(BLOCK_SIZE))

plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
plt.suptitle("Perbandingan Piksel Asli vs Koefisien DCT (Blok 8×8 Pertama)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("tahap2b_blok_piksel_vs_dct.png", dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Output: tahap2b_blok_piksel_vs_dct.png")

# =============================================================
# BAGIAN 3: KUANTISASI
# =============================================================
print("\n" + "=" * 60)
print("BAGIAN 3: KUANTISASI")
print("=" * 60)

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
    skala = 5000 / qf if qf < 50 else 200 - 2 * qf
    tabel = np.floor((TABEL_KUANTISASI_STANDAR * skala + 50) / 100)
    tabel = np.clip(tabel, 1, 255)
    return tabel

def kuantisasi(dct_block, tabel_q):
    return np.round(dct_block / tabel_q)

def dekuantisasi(q_block, tabel_q):
    return q_block * tabel_q

# Tabel kuantisasi QF 10, 50, 100 + blok setelah kuantisasi
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
qf_demos = [10, 50, 100]

for col, qf_d in enumerate(qf_demos):
    tabel_q = buat_tabel_kuantisasi(qf_d)
    q_blok  = kuantisasi(dct_contoh, tabel_q)
    dq_blok = dekuantisasi(q_blok, tabel_q)

    # Baris atas: tabel kuantisasi
    im = axes[0, col].imshow(tabel_q, cmap='YlOrRd', vmin=1, vmax=255)
    axes[0, col].set_title(f"Tabel Kuantisasi QF={qf_d}", fontweight='bold')
    for i in range(8):
        for j in range(8):
            axes[0, col].text(j, i, str(int(tabel_q[i, j])),
                              ha='center', va='center', fontsize=6,
                              color='white' if tabel_q[i, j] > 150 else 'black')
    axes[0, col].set_xticks(range(8))
    axes[0, col].set_yticks(range(8))
    plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

    # Baris bawah: blok setelah kuantisasi
    im2 = axes[1, col].imshow(q_blok, cmap='RdBu_r')
    axes[1, col].set_title(f"Blok 8×8 Setelah Kuantisasi QF={qf_d}\n"
                            f"(nilai 0 = informasi hilang)", fontweight='bold', fontsize=9)
    for i in range(8):
        for j in range(8):
            axes[1, col].text(j, i, f"{int(q_blok[i,j])}",
                              ha='center', va='center', fontsize=6,
                              color='white' if abs(q_blok[i,j]) > np.max(np.abs(q_blok))*0.5 else 'black')
    axes[1, col].set_xticks(range(8))
    axes[1, col].set_yticks(range(8))
    plt.colorbar(im2, ax=axes[1, col], fraction=0.046, pad=0.04)

    nonzero = np.sum(q_blok != 0)
    print(f"  QF={qf_d}: koefisien non-zero = {nonzero}/64, DC = {q_blok[0,0]:.1f}")

plt.suptitle("Tabel Kuantisasi dan Hasil Kuantisasi Blok DCT 8×8\n"
             "(QF kecil = pembagi besar = banyak koefisien jadi 0 = banyak detail hilang)",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("tahap3_kuantisasi.png", dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Output: tahap3_kuantisasi.png")

# =============================================================
# BAGIAN 4: KOMPRESI JPEG MANUAL (simulasi)
# =============================================================
print("\n" + "=" * 60)
print("BAGIAN 4: SIMULASI KOMPRESI JPEG")
print("=" * 60)

def kompresi_jpeg_pil(image_array, qf):
    img_pil = Image.fromarray(image_array.astype(np.uint8), mode="L")
    buffer  = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=qf)
    buffer.seek(0)
    return np.array(Image.open(buffer).convert("L"), dtype=np.uint8)

img_uint8 = img_array.astype(np.uint8)

# Kompres beberapa QF untuk demonstrasi
qf_demo_list = [10, 30, 50, 70, 90, 100]
contoh_kompres = {}
print(f"\n{'QF':<6} {'PSNR (dB)':<12} {'MSE':<12}")
print("-" * 30)
for qf in qf_demo_list:
    compressed = kompresi_jpeg_pil(img_uint8, qf)
    contoh_kompres[qf] = compressed
    mse  = np.mean((img_uint8.astype(float) - compressed.astype(float))**2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    print(f"QF {qf:<4} PSNR={psnr:.2f} dB   MSE={mse:.2f}")

# Perbandingan hasil kompresi berbagai QF
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
for idx, qf in enumerate(qf_demo_list):
    comp = contoh_kompres[qf]
    mse  = np.mean((img_uint8.astype(float) - comp.astype(float))**2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    axes[idx].imshow(comp, cmap='gray', vmin=0, vmax=255)
    axes[idx].set_title(f"QF = {qf}\nPSNR = {psnr:.2f} dB | MSE = {mse:.1f}",
                         fontweight='bold', fontsize=10)
    axes[idx].axis('off')

plt.suptitle("Hasil Kompresi JPEG pada Berbagai Quality Factor (QF)\n"
             "(QF makin kecil = kualitas makin buruk = artefak makin jelas)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("tahap4_perbandingan_kompresi_qf.png", dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Output: tahap4_perbandingan_kompresi_qf.png")

# =============================================================
# BAGIAN 5: WATERMARKING (EMBED)
# =============================================================
print("\n" + "=" * 60)
print("BAGIAN 5: EMBED WATERMARK (LSB)")
print("=" * 60)

# Buat watermark biner pola kotak-kotak
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

watermarked_array = embed_watermark(img_uint8, watermark)
Image.fromarray(watermarked_array, mode="L").save("watermarked_original.png")

mse_wm  = np.mean((img_uint8.astype(float) - watermarked_array.astype(float))**2)
psnr_wm = 10 * np.log10(255**2 / (mse_wm + 1e-10))
diff    = np.abs(img_uint8.astype(int) - watermarked_array.astype(int))

print(f"\nGambar ter-watermark disimpan: watermarked_original.png")
print(f"MSE  : {mse_wm:.4f}")
print(f"PSNR : {psnr_wm:.2f} dB")

# Watermark + proses embed
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(img_uint8, cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Gambar Asli\n(sebelum embed)", fontweight='bold')
axes[0].axis('off')

axes[1].imshow(watermark, cmap='gray', vmin=0, vmax=1)
axes[1].set_title(f"Watermark Biner\n(pola {WATERMARK_TILE}×{WATERMARK_TILE} kotak)", fontweight='bold')
axes[1].axis('off')

axes[2].imshow(watermarked_array, cmap='gray', vmin=0, vmax=255)
axes[2].set_title(f"Gambar + Watermark\nPSNR = {psnr_wm:.2f} dB\n(tidak terlihat perbedaannya)", fontweight='bold')
axes[2].axis('off')

im3 = axes[3].imshow(diff * 100, cmap='hot', vmin=0, vmax=100)
axes[3].set_title("Perbedaan Asli vs Watermarked\n(nilai dikali ×100 agar terlihat)\nMax diff = 1 piksel", fontweight='bold')
axes[3].axis('off')
plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

plt.suptitle("Proses Embed Watermark LSB (Least Significant Bit)\n"
             "Teknik: bit terakhir setiap piksel diganti dengan bit watermark",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("tahap5_embed_watermark.png", dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Output: tahap5_embed_watermark.png")

# =============================================================
# BAGIAN 6: KOMPRESI + EKSTRAKSI + EVALUASI PER QF
# =============================================================
print("\n" + "=" * 60)
print("BAGIAN 6: KOMPRESI → EKSTRAKSI → EVALUASI")
print("=" * 60)

def extract_watermark(image_array):
    return image_array.astype(np.uint8) & 1

def hitung_ber(wm_asli, wm_ekstrak):
    return np.sum(wm_asli != wm_ekstrak) / wm_asli.size

def hitung_nc(wm_asli, wm_ekstrak):
    orig = wm_asli.astype(float)
    extr = wm_ekstrak.astype(float)
    return np.sum(orig * extr) / (
           np.sqrt(np.sum(orig**2)) * np.sqrt(np.sum(extr**2)) + 1e-10)

def hitung_psnr(asli, hasil):
    mse = np.mean((asli.astype(float) - hasil.astype(float))**2)
    return 10 * np.log10(255**2 / (mse + 1e-10))

print(f"\n{'QF':<6} {'BER':<10} {'NC':<10} {'PSNR(dB)':<12} {'Status'}")
print("-" * 55)

results           = []
compressed_images = {}

for qf in QUALITY_FACTORS:
    compressed  = kompresi_jpeg_pil(watermarked_array, qf)
    compressed_images[qf] = compressed
    wm_ekstrak  = extract_watermark(compressed)
    ber         = hitung_ber(watermark, wm_ekstrak)
    nc          = hitung_nc(watermark, wm_ekstrak)
    psnr        = hitung_psnr(img_uint8, compressed)
    status      = "[BERHASIL]" if (ber <= THRESHOLD_BER and nc >= THRESHOLD_NC) else "[GAGAL]"
    print(f"QF {qf:<4} BER={ber:.4f}  NC={nc:.4f}  PSNR={psnr:.2f}     {status}")
    results.append({"qf": qf, "ber": ber, "nc": nc, "psnr": psnr, "status": status})

qf_berhasil = [r["qf"] for r in results if "BERHASIL" in r["status"]]
qf_gagal    = [r["qf"] for r in results if "GAGAL"    in r["status"]]
print(f"\nQF Gagal    : {qf_gagal}")
print(f"QF Berhasil : {qf_berhasil}")

# Watermark yang diekstrak per QF (semua 10 QF)
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()
for idx, qf in enumerate(QUALITY_FACTORS):
    wm_ext  = extract_watermark(compressed_images[qf])
    r       = results[idx]
    status  = r["status"]
    color   = "green" if "BERHASIL" in status else "red"
    axes[idx].imshow(wm_ext, cmap='gray', vmin=0, vmax=1)
    axes[idx].set_title(
        f"QF = {qf}\nBER={r['ber']:.4f} | NC={r['nc']:.4f}\n{status}",
        fontweight='bold', fontsize=9, color=color
    )
    for spine in axes[idx].spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
    axes[idx].axis('off')

plt.suptitle("Hasil Ekstraksi Watermark Setelah Kompresi JPEG\n"
             "(Semakin rusak pola kotak = semakin tinggi BER = semakin gagal)",
             fontsize=13, fontweight='bold')
legend_els = [Patch(fc='green', ec='black', label='[BERHASIL] (BER ≤ 0.1 dan NC ≥ 0.9)'),
              Patch(fc='red',   ec='black', label='[GAGAL]')]
fig.legend(handles=legend_els, loc='lower center', ncol=2,
           fontsize=11, bbox_to_anchor=(0.5, 0.01), frameon=True, edgecolor='black')
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("tahap6a_ekstraksi_watermark_semua_qf.png", dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Output: tahap6a_ekstraksi_watermark_semua_qf.png")

# Perbandingan gambar terkompresi per QF
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()
for idx, qf in enumerate(QUALITY_FACTORS):
    comp   = compressed_images[qf]
    r      = results[idx]
    status = r["status"]
    color  = "green" if "BERHASIL" in status else "red"
    axes[idx].imshow(comp, cmap='gray', vmin=0, vmax=255)
    axes[idx].set_title(
        f"QF = {qf} | PSNR={r['psnr']:.1f}dB\n{status}",
        fontweight='bold', fontsize=9, color=color
    )
    for spine in axes[idx].spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
    axes[idx].axis('off')

plt.suptitle("Gambar Watermarked Setelah Kompresi JPEG per Quality Factor\n"
             "(Bingkai merah = watermark gagal diekstrak dari gambar ini)",
             fontsize=13, fontweight='bold')
fig.legend(handles=legend_els, loc='lower center', ncol=2,
           fontsize=11, bbox_to_anchor=(0.5, 0.01), frameon=True, edgecolor='black')
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("tahap6b_gambar_terkompresi_semua_qf.png", dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Output: tahap6b_gambar_terkompresi_semua_qf.png")

# =============================================================
# BAGIAN 7: GRAFIK EVALUASI METRIK
# =============================================================
print("\n" + "=" * 60)
print("BAGIAN 7: GRAFIK EVALUASI METRIK")
print("=" * 60)

qf_vals   = [r["qf"]   for r in results]
ber_vals  = [r["ber"]  for r in results]
nc_vals   = [r["nc"]   for r in results]
psnr_vals = [r["psnr"] for r in results]
bar_colors = ['green' if "BERHASIL" in r["status"] else 'red' for r in results]

# Grafik BER vs QF
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(qf_vals, ber_vals, color=bar_colors, alpha=0.85, edgecolor='black', width=6)
ax.axhline(y=THRESHOLD_BER, color='navy', linestyle='--', linewidth=2,
           label=f'Threshold BER = {THRESHOLD_BER}')
for bar, val in zip(bars, ber_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_title("Bit Error Rate (BER) vs Quality Factor\n"
             "BER ≤ 0.1 → Watermark berhasil diekstrak", fontweight='bold', fontsize=12)
ax.set_xlabel("Quality Factor (QF)", fontsize=11)
ax.set_ylabel("Bit Error Rate (BER)", fontsize=11)
ax.set_xticks(qf_vals)
ax.set_ylim(0, max(ber_vals) * 1.25 + 0.02)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
legend_els2 = [Patch(fc='green', ec='black', label='[BERHASIL]'),
               Patch(fc='red',   ec='black', label='[GAGAL]')]
ax.legend(handles=[ax.get_legend_handles_labels()[0][0]] + legend_els2,
          labels=[f'Threshold BER={THRESHOLD_BER}', '[BERHASIL]', '[GAGAL]'],
          fontsize=9)
plt.tight_layout()
plt.savefig("tahap7a_grafik_ber.png", dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Output: tahap7a_grafik_ber.png")

# Grafik NC vs QF
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(qf_vals, nc_vals, color=bar_colors, alpha=0.85, edgecolor='black', width=6)
ax.axhline(y=THRESHOLD_NC, color='navy', linestyle='--', linewidth=2,
           label=f'Threshold NC = {THRESHOLD_NC}')
for bar, val in zip(bars, nc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_title("Normalized Correlation (NC) vs Quality Factor\n"
             "NC ≥ 0.9 → Watermark berhasil diekstrak", fontweight='bold', fontsize=12)
ax.set_xlabel("Quality Factor (QF)", fontsize=11)
ax.set_ylabel("Normalized Correlation (NC)", fontsize=11)
ax.set_xticks(qf_vals)
ax.set_ylim(0, 1.2)
ax.legend(handles=[plt.Line2D([0],[0], color='navy', linestyle='--', linewidth=2,
                               label=f'Threshold NC={THRESHOLD_NC}')] + legend_els2,
          fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("tahap7b_grafik_nc.png", dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Output: tahap7b_grafik_nc.png")

# Grafik PSNR vs QF
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(qf_vals, psnr_vals, 'purple', marker='o', linewidth=2.5, markersize=8)
for x, y in zip(qf_vals, psnr_vals):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
ax.fill_between(qf_vals, psnr_vals, alpha=0.1, color='purple')
ax.set_title("PSNR (Peak Signal-to-Noise Ratio) vs Quality Factor\n"
             "PSNR makin tinggi = kualitas gambar makin baik setelah kompresi",
             fontweight='bold', fontsize=12)
ax.set_xlabel("Quality Factor (QF)", fontsize=11)
ax.set_ylabel("PSNR (dB)", fontsize=11)
ax.set_xticks(qf_vals)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("tahap7c_grafik_psnr.png", dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Output: tahap7c_grafik_psnr.png")

# Grafik ketiga metrik dalam satu figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].bar(qf_vals, ber_vals, color=bar_colors, alpha=0.85, edgecolor='black', width=6)
axes[0].axhline(y=THRESHOLD_BER, color='navy', linestyle='--', linewidth=2,
                label=f'Threshold = {THRESHOLD_BER}')
axes[0].set_title("BER vs QF", fontweight='bold')
axes[0].set_xlabel("Quality Factor")
axes[0].set_ylabel("BER")
axes[0].set_xticks(qf_vals)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(qf_vals, nc_vals, color=bar_colors, alpha=0.85, edgecolor='black', width=6)
axes[1].axhline(y=THRESHOLD_NC, color='navy', linestyle='--', linewidth=2,
                label=f'Threshold = {THRESHOLD_NC}')
axes[1].set_title("NC vs QF", fontweight='bold')
axes[1].set_xlabel("Quality Factor")
axes[1].set_ylabel("NC")
axes[1].set_xticks(qf_vals)
axes[1].set_ylim(0, 1.2)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

axes[2].plot(qf_vals, psnr_vals, 'purple', marker='o', linewidth=2.5, markersize=8)
axes[2].fill_between(qf_vals, psnr_vals, alpha=0.1, color='purple')
axes[2].set_title("PSNR vs QF", fontweight='bold')
axes[2].set_xlabel("Quality Factor")
axes[2].set_ylabel("PSNR (dB)")
axes[2].set_xticks(qf_vals)
axes[2].grid(True, alpha=0.3)

plt.suptitle("Ringkasan Evaluasi Metrik Watermarking vs Quality Factor JPEG",
             fontsize=13, fontweight='bold')
fig.legend(handles=legend_els, loc='lower center', ncol=2,
           fontsize=10, bbox_to_anchor=(0.5, 0.01), frameon=True, edgecolor='black')
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("tahap7d_ringkasan_semua_metrik.png", dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Output: tahap7d_ringkasan_semua_metrik.png")

# =============================================================
# RINGKASAN AKHIR
# =============================================================
print("\n" + "=" * 60)
print("RINGKASAN AKHIR")
print("=" * 60)
print(f"Metode watermark : LSB (Least Significant Bit)")
print(f"Watermark        : Biner pola kotak {WATERMARK_TILE}×{WATERMARK_TILE}")
print(f"Ukuran gambar    : {W}×{H} piksel")
print(f"\nHasil evaluasi:")
for r in results:
    print(f"  QF {r['qf']:>3} | BER={r['ber']:.4f} | NC={r['nc']:.4f} | "
          f"PSNR={r['psnr']:.1f}dB | {r['status']}")
print(f"\n→ Watermark GAGAL diekstrak pada QF    : {qf_gagal}")
print(f"→ Watermark BERHASIL diekstrak pada QF : {qf_berhasil}")
if qf_gagal:
    print(f"→ Threshold: watermark aman pada QF ≥ {min(qf_berhasil)}")

print("\n" + "=" * 60)
print("DAFTAR FILE OUTPUT YANG DIHASILKAN")
print("=" * 60)
output_files = [
    ("tahap1_gambar_asli_grayscale.png",         "Bagian 1 - Gambar asli grayscale"),
    ("tahap2a_hasil_dct_gambar.png",              "Bagian 2 - Koefisien DCT seluruh gambar"),
    ("tahap2b_blok_piksel_vs_dct.png",            "Bagian 2 - Perbandingan blok piksel vs DCT 8×8"),
    ("tahap3_kuantisasi.png",                     "Bagian 3 - Tabel kuantisasi & blok setelah kuantisasi"),
    ("tahap4_perbandingan_kompresi_qf.png",       "Bagian 4 - Perbandingan hasil kompresi JPEG"),
    ("tahap5_embed_watermark.png",                "Bagian 5 - Proses embed watermark LSB"),
    ("tahap6a_ekstraksi_watermark_semua_qf.png",  "Bagian 6 - Watermark diekstrak tiap QF"),
    ("tahap6b_gambar_terkompresi_semua_qf.png",   "Bagian 6 - Gambar terkompresi tiap QF"),
    ("tahap7a_grafik_ber.png",                    "Bagian 7 - Grafik BER vs QF"),
    ("tahap7b_grafik_nc.png",                     "Bagian 7 - Grafik NC vs QF"),
    ("tahap7c_grafik_psnr.png",                   "Bagian 7 - Grafik PSNR vs QF"),
    ("tahap7d_ringkasan_semua_metrik.png",        "Bagian 7 - Ringkasan semua metrik"),
    ("watermarked_original.png",                  "Gambar ter-watermark (sebelum kompresi)"),
]
for fname, desc in output_files:
    print(f"  {fname:<48} ← {desc}")