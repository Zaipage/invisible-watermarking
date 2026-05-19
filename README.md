# Watermarking Digital dengan Metode LSB pada Domain Spasial + Analisis Kompresi JPEG

> Tugas ini mengimplementasikan watermarking digital menggunakan metode **LSB (Least Significant Bit)** pada gambar grayscale, kemudian mengevaluasi ketahanan (*robustness*) watermark terhadap kompresi JPEG pada berbagai *Quality Factor* (QF). Seluruh proses mencakup transformasi DCT, kuantisasi, embedding, ekstraksi, dan evaluasi metrik BER, NC, dan PSNR.

---

## Daftar Isi

1. [Deskripsi Proyek](#1-deskripsi-proyek)
2. [Cara Kerja Sistem](#2-cara-kerja-sistem)
3. [Struktur File Output](#3-struktur-file-output)
4. [Tahap 1 — Load & Persiapan Gambar](#4-tahap-1--load--persiapan-gambar)
5. [Tahap 2 — DCT (Discrete Cosine Transform)](#5-tahap-2--dct-discrete-cosine-transform)
6. [Tahap 3 — Kuantisasi](#6-tahap-3--kuantisasi)
7. [Tahap 4 — Simulasi Kompresi JPEG](#7-tahap-4--simulasi-kompresi-jpeg)
8. [Tahap 5 — Embed Watermark (LSB)](#8-tahap-5--embed-watermark-lsb)
9. [Tahap 6 — Ekstraksi & Evaluasi per QF](#9-tahap-6--ekstraksi--evaluasi-per-qf)
10. [Tahap 7 — Grafik Evaluasi Metrik](#10-tahap-7--grafik-evaluasi-metrik)
11. [Tabel Hasil Lengkap](#11-tabel-hasil-lengkap)
12. [Analisis & Kesimpulan](#12-analisis--kesimpulan)
13. [Cara Menjalankan](#13-cara-menjalankan)

---

## 1. Deskripsi Tugas

Watermarking digital adalah teknik menyisipkan informasi tersembunyi (*watermark*) ke dalam suatu media (gambar, audio, video) tanpa mengubah kualitas visualnya secara signifikan. Tujuannya antara lain untuk **perlindungan hak cipta**, **autentikasi**, dan **pelacakan distribusi** konten digital.

Proyek ini menggunakan pendekatan **LSB (Least Significant Bit)** — metode paling sederhana dalam watermarking domain spasial, kemudian menguji seberapa tahan watermark tersebut setelah gambar dikompres menggunakan algoritma **JPEG** pada *Quality Factor* 10 hingga 100.

**Mengapa LSB rentan terhadap JPEG?**
Kompresi JPEG bekerja di domain frekuensi (via DCT + kuantisasi). Proses kuantisasi secara agresif membulatkan koefisien frekuensi tinggi, yang ketika dikembalikan ke domain spasial akan mengubah nilai LSB piksel. Inilah mengapa watermark LSB mudah hancur oleh JPEG.

---

## 2. Cara Kerja Sistem

```
[Gambar Asli]
      │
      ▼
[1] Load & konversi ke Grayscale (1600×1600)
      │
      ▼
[2] DCT per blok 8×8 → analisis koefisien frekuensi
      │
      ▼
[3] Kuantisasi → simulasi lossy compression (QF 10/50/100)
      │
      ▼
[4] Simulasi Kompresi JPEG (PIL) → lihat degradasi kualitas
      │
      ▼
[5] Embed Watermark LSB → ganti bit LSB setiap piksel dengan bit watermark
      │
      ▼
[6] Kompres gambar ter-watermark → ekstrak ulang watermark → hitung BER & NC
      │
      ▼
[7] Visualisasi grafik BER, NC, PSNR vs Quality Factor
```

---

## 3. Struktur File Output

Setelah menjalankan `watermarking.py`, seluruh file berikut akan dihasilkan di direktori yang sama:

| File Output | Tahap | Deskripsi |
|---|---|---|
| `tahap1_gambar_asli_grayscale.png` | 1 | Gambar asli setelah dikonversi ke grayscale |
| `tahap2a_hasil_dct_gambar.png` | 2 | Visualisasi koefisien DCT seluruh gambar (skala log) |
| `tahap2b_blok_piksel_vs_dct.png` | 2 | Perbandingan nilai piksel vs koefisien DCT blok 8×8 pertama |
| `tahap3_kuantisasi.png` | 3 | Tabel kuantisasi dan hasil kuantisasi blok untuk QF 10, 50, 100 |
| `tahap4_perbandingan_kompresi_qf.png` | 4 | Hasil kompresi gambar pada QF 10, 30, 50, 70, 90, 100 |
| `tahap5_embed_watermark.png` | 5 | Proses embed: watermark, gambar ter-watermark, peta perbedaan |
| `tahap6a_ekstraksi_watermark_semua_qf.png` | 6 | Watermark hasil ekstraksi untuk semua 10 QF |
| `tahap6b_gambar_terkompresi_semua_qf.png` | 6 | Gambar terkompresi untuk semua 10 QF |
| `tahap7a_grafik_ber.png` | 7 | Grafik BER vs Quality Factor |
| `tahap7b_grafik_nc.png` | 7 | Grafik NC vs Quality Factor |
| `tahap7c_grafik_psnr.png` | 7 | Grafik PSNR vs Quality Factor |
| `tahap7d_ringkasan_semua_metrik.png` | 7 | Ringkasan ketiga metrik dalam satu figure |
| `watermarked_original.png` | 5 | File gambar ter-watermark (sebelum kompresi) |

---

## 4. Tahap 1 — Load & Persiapan Gambar

**Apa yang dilakukan:**
Gambar input (`foto_kakey.png`) dimuat menggunakan PIL (*Python Imaging Library*) dan dikonversi ke mode **grayscale** (`L`). Konversi ke grayscale menyederhanakan pemrosesan — hanya ada satu kanal intensitas piksel dengan nilai 0–255, tanpa perlu menangani 3 kanal RGB secara terpisah.

```python
img = Image.open(NAMA_FOTO).convert("L")
img_array = np.array(img, dtype=np.float64)
```

**Spesifikasi gambar:**

| Parameter | Nilai |
|---|---|
| Nama file | `foto_kakey.png` |
| Ukuran | 1600 × 1600 piksel |
| Mode warna | Grayscale (L) |
| Tipe data array | float64 |
| Total piksel | 2.560.000 |
| Total blok 8×8 | 40.000 blok |

**Output:**

![Tahap 1 — Gambar Asli Grayscale](output/tahap1_gambar_asli_grayscale.png)

---

## 5. Tahap 2 — DCT (Discrete Cosine Transform)

**Konsep:**
DCT adalah transformasi matematika yang mengubah sinyal dari **domain spasial** (nilai piksel) ke **domain frekuensi** (koefisien frekuensi). JPEG menggunakan DCT-2D pada blok 8×8 piksel.

Rumus DCT-2D:

$$F(u,v) = \frac{2}{N} C(u) C(v) \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} f(x,y) \cos\frac{(2x+1)u\pi}{2N} \cos\frac{(2y+1)v\pi}{2N}$$

Di mana:
- $f(x,y)$ = nilai piksel pada posisi $(x,y)$
- $F(u,v)$ = koefisien DCT pada frekuensi $(u,v)$
- $C(u) = \frac{1}{\sqrt{2}}$ jika $u=0$, dan $1$ untuk selainnya
- $N$ = ukuran blok (8)

**Jenis koefisien:**
- **Koefisien DC** - posisi $(0,0)$, merepresentasikan **nilai rata-rata** seluruh blok. Nilainya besar karena mengandung informasi energi utama.
- **Koefisien AC** - posisi lainnya (63 koefisien), merepresentasikan **variasi frekuensi** dari rendah ke tinggi. Frekuensi tinggi = detail halus, frekuensi rendah = kontur besar.

**Contoh pada blok 8×8 pertama gambar:**
Blok pertama gambar (pojok kiri atas) bernilai seragam 134 di semua piksel. Hasilnya, DCT hanya menghasilkan satu koefisien DC = 1072 dan semua koefisien AC = 0, karena tidak ada variasi frekuensi sama sekali pada area yang seragam.

**Implementasi:**
Proyek ini menggunakan `scipy.fft.dctn` dengan normalisasi `ortho` untuk efisiensi, menggantikan implementasi manual yang terlalu lambat untuk gambar 1600×1600.

```python
from scipy.fft import dctn, idctn

def dct_2d_fast(block):
    return dctn(block, norm='ortho')

def apply_dct_blocks(image, block_size=8):
    H, W = image.shape
    dct_image = np.zeros_like(image)
    for i in range(0, H - H % block_size, block_size):
        for j in range(0, W - W % block_size, block_size):
            blok = image[i:i+block_size, j:j+block_size]
            dct_image[i:i+block_size, j:j+block_size] = dct_2d_fast(blok)
    return dct_image
```

**Output:**

![Tahap 2a — Koefisien DCT Seluruh Gambar](output/tahap2a_hasil_dct_gambar.png)

*Pada visualisasi DCT (skala log), area terang di pojok kiri atas menunjukkan energi tinggi pada frekuensi rendah - sesuai dengan sifat alami gambar yang sebagian besar berisi perubahan warna yang gradual, bukan detail tajam.*

![Tahap 2b — Blok Piksel vs Koefisien DCT](output/tahap2b_blok_piksel_vs_dct.png)

*Kiri: nilai piksel mentah blok 8×8 pertama. Kanan: koefisien DCT dari blok yang sama. Terlihat energi terkonsentrasi di koefisien DC (kiri atas), sedangkan koefisien AC bernilai mendekati nol.*

---

## 6. Tahap 3 - Kuantisasi

**Konsep:**
Kuantisasi adalah proses **pembulatan** koefisien DCT menggunakan tabel pembagi (*quantization table*). Inilah inti dari sifat *lossy* pada JPEG, koefisien yang kecil setelah dibagi akan dibulatkan ke nol, sehingga banyak detail (terutama frekuensi tinggi) yang hilang secara permanen.

$$Q(u,v) = \text{round}\left(\frac{F(u,v)}{T(u,v)}\right)$$

Di mana $T(u,v)$ adalah tabel kuantisasi standar JPEG (luminance).

**Pengaruh Quality Factor:**
*Quality Factor* (QF) menentukan seberapa agresif kuantisasi dilakukan. Tabel kuantisasi disesuaikan dari tabel standar JPEG berdasarkan rumus:

```
Jika QF < 50 : skala = 5000 / QF
Jika QF ≥ 50 : skala = 200 - 2×QF
Tabel = floor((Tabel_Standar × skala + 50) / 100)
```

Semakin kecil QF → skala semakin besar → pembagi semakin besar → lebih banyak koefisien yang dibulatkan ke nol → **lebih banyak informasi hilang**.

**Tabel kuantisasi standar JPEG (luminance):**

```
16  11  10  16  24  40  51  61
12  12  14  19  26  58  60  55
14  13  16  24  40  57  69  56
14  17  22  29  51  87  80  62
18  22  37  56  68 109 103  77
24  35  55  64  81 104 113  92
49  64  78  87 103 121 120 101
72  92  95  98 112 100 103  99
```

**Perbandingan koefisien non-zero setelah kuantisasi:**

| QF | Koefisien Non-Zero (dari 64) | Koefisien Hilang | Keterangan |
|---|---|---|---|
| 10 | 1 | 63 (98%) | Hampir semua detail hilang |
| 50 | 1 | 63 (98%) | Blok seragam → hanya DC tersisa |
| 100 | 1 | 63 (98%) | Blok seragam → sama, karena blok contoh seragam |

> **Catatan:** Blok 8×8 pertama gambar ini kebetulan seragam (nilai piksel 134 semua), sehingga hanya koefisien DC yang bertahan di semua QF. Pada blok dengan variasi piksel, QF yang lebih tinggi akan mempertahankan jauh lebih banyak koefisien.

**Output:**

![Tahap 3 — Kuantisasi](output/tahap3_kuantisasi.png)

*Baris atas: tabel kuantisasi untuk QF 10, 50, 100 (semakin cerah) = nilai pembagi semakin besar. Baris bawah: blok DCT setelah kuantisasi - QF rendah membuat hampir semua koefisien menjadi nol.*

---

## 7. Tahap 4 — Simulasi Kompresi JPEG

**Konsep:**
Kompresi JPEG penuh melibatkan: konversi warna → DCT per blok 8×8 → kuantisasi → *zigzag scan* → *run-length encoding* → *Huffman coding*. Proyek ini menggunakan PIL untuk mensimulasikan seluruh pipeline ini secara transparan:

```python
def kompresi_jpeg_pil(image_array, qf):
    img_pil = Image.fromarray(image_array.astype(np.uint8), mode="L")
    buffer  = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=qf)
    buffer.seek(0)
    return np.array(Image.open(buffer).convert("L"), dtype=np.uint8)
```

**Hasil kompresi gambar asli pada berbagai QF:**

| QF | PSNR (dB) | MSE | Keterangan |
|---|---|---|---|
| 10 | 35.70 | 17.50 | Artefak blok sangat terlihat |
| 30 | 42.53 | 3.63 | Artefak masih terlihat di tepi |
| 50 | 46.23 | 1.55 | Kualitas mulai baik |
| 70 | 50.78 | 0.54 | Kualitas bagus |
| 90 | 58.17 | 0.10 | Hampir tidak ada perbedaan |
| 100 | 64.68 | 0.02 | Sangat mendekati lossless |

**Output:**

![Tahap 4 — Perbandingan Kompresi QF](output/tahap4_perbandingan_kompresi_qf.png)

*Terlihat jelas artefak "blocking" pada QF=10 - efek kotak-kotak 8×8 yang muncul karena kuantisasi agresif. Semakin tinggi QF, semakin halus hasilnya.*

---

## 8. Tahap 5 — Embed Watermark (LSB)

**Konsep LSB:**
Metode LSB menyisipkan watermark dengan **mengganti bit paling tidak signifikan** (bit ke-0) dari setiap nilai piksel dengan satu bit watermark. Karena bit ini hanya mengubah nilai piksel sebesar ±1, perubahan tidak terlihat oleh mata manusia.

```
Piksel asli   : 10110101  (= 181)
Watermark bit :        1
Piksel baru   : 10110101  (bit LSB sudah 1, tidak berubah)

Piksel asli   : 10110100  (= 180)
Watermark bit :        1
Piksel baru   : 10110101  (= 181, berubah +1)
```

**Implementasi:**
```python
def embed_watermark(image_array, wm):
    img_uint8   = image_array.astype(np.uint8)
    # & 0xFE  → set bit LSB ke 0  (11111110)
    # | wm    → set bit LSB sesuai watermark
    watermarked = (img_uint8 & 0xFE) | wm
    return watermarked
```

**Pola watermark:**
Watermark yang digunakan adalah **pola kotak-kotak biner** (*checkerboard*) berukuran tile 32×32 piksel — mirip papan catur. Pola ini dipilih karena:
- Mudah dibuat secara programatik
- Mudah dievaluasi secara visual setelah ekstraksi
- Distribusi bit 0 dan 1 seimbang (50:50)

```python
# Setiap kotak 32×32 diisi 1 atau 0 secara bergantian
for i in range(0, H, 32):
    for j in range(0, W, 32):
        if (i//32 + j//32) % 2 == 0:
            watermark[i:i+32, j:j+32] = 1
```

**Spesifikasi watermark:**

| Parameter | Nilai |
|---|---|
| Ukuran tile | 32 × 32 piksel |
| Total bit watermark | 2.560.000 bit |
| Bit bernilai '1' | 1.280.000 (50%) |
| Bit bernilai '0' | 1.280.000 (50%) |
| MSE (asli vs watermarked) | 0.5006 |
| PSNR (asli vs watermarked) | **51.14 dB** |

> PSNR 51.14 dB mengkonfirmasi bahwa perubahan akibat watermark **tidak terlihat secara visual** - nilai di atas 40 dB umumnya dianggap tidak dapat dibedakan oleh mata manusia.

**Output:**

![Tahap 5 — Embed Watermark LSB](output/tahap5_embed_watermark.png)

*Dari kiri ke kanan: gambar asli, watermark biner pola kotak-kotak, gambar ter-watermark (identik secara visual dengan asli), dan peta perbedaan yang dikali ×100 agar terlihat. Perubahan maksimal hanya 1 nilai piksel.*

Gambar ter-watermark juga disimpan terpisah sebagai:

![Gambar Ter-Watermark](output/watermarked_original.png)

---

## 9. Tahap 6 — Ekstraksi & Evaluasi per QF

**Proses:**
Setelah gambar ter-watermark dikompres dengan JPEG pada berbagai QF, watermark diekstrak kembali dengan mengambil bit LSB setiap piksel:

```python
def extract_watermark(image_array):
    return image_array.astype(np.uint8) & 1   # ambil bit LSB
```

**Metrik evaluasi:**

**1. BER (Bit Error Rate)**
Mengukur proporsi bit watermark yang salah setelah ekstraksi:

$$BER = \frac{\text{jumlah bit yang berbeda}}{\text{total bit watermark}}$$

- BER = 0.0 → watermark sempurna terekstrak
- BER = 0.5 → watermark hancur total (acak)
- **Threshold: BER ≤ 0.1** (maks 10% bit boleh salah)

**2. NC (Normalized Correlation)**
Mengukur kemiripan antara watermark asli dan watermark hasil ekstraksi:

$$NC = \frac{\sum_{i,j} W(i,j) \cdot W'(i,j)}{\sqrt{\sum_{i,j} W(i,j)^2} \cdot \sqrt{\sum_{i,j} W'(i,j)^2}}$$

- NC = 1.0 → watermark identik dengan asli
- NC = 0.0 → tidak ada korelasi
- **Threshold: NC ≥ 0.9**

**3. PSNR (Peak Signal-to-Noise Ratio)**
Mengukur kualitas gambar terkompresi dibanding gambar asli (bukan watermarked):

$$PSNR = 10 \cdot \log_{10}\left(\frac{255^2}{MSE}\right) \quad \text{dB}$$

**Output — Watermark terekstrak untuk semua QF:**

![Tahap 6a — Ekstraksi Watermark Semua QF](output/tahap6a_ekstraksi_watermark_semua_qf.png)

*Bingkai **merah** = watermark gagal diekstrak (BER > 0.1 atau NC < 0.9). Bingkai **hijau** = berhasil. Terlihat jelas bahwa pada QF 10–90, pola kotak-kotak watermark rusak parah hingga menjadi noise acak. Hanya pada QF 100 pola kotak-kotak masih terbaca jelas.*

**Output — Gambar terkompresi untuk semua QF:**

![Tahap 6b — Gambar Terkompresi Semua QF](output/tahap6b_gambar_terkompresi_semua_qf.png)

*Perhatikan bahwa secara visual, gambar terkompresi QF 50 ke atas sudah terlihat bagus — namun watermark di dalamnya sudah hancur karena LSB-nya telah diubah oleh proses kuantisasi JPEG.*

---

## 10. Tahap 7 — Grafik Evaluasi Metrik

### Grafik BER vs Quality Factor

![Tahap 7a — Grafik BER](output/tahap7a_grafik_ber.png)

*Batang **merah** = gagal (BER > 0.1). Batang **hijau** = berhasil (BER ≤ 0.1). Pada QF 10–90, BER berkisar antara 0.24–0.53 — jauh di atas threshold, bahkan mendekati 0.5 yang berarti watermark nyaris acak. Hanya QF 100 (BER = 0.04) yang berhasil.*

### Grafik NC vs Quality Factor

![Tahap 7b — Grafik NC](output/tahap7b_grafik_nc.png)

*NC yang rendah (0.20–0.76) pada QF 10–90 mengkonfirmasi rendahnya korelasi antara watermark asli dan watermark terekstrak. QF 100 menghasilkan NC = 0.96, di atas threshold 0.9.*

### Grafik PSNR vs Quality Factor

![Tahap 7c — Grafik PSNR](output/tahap7c_grafik_psnr.png)

*PSNR meningkat seiring QF yang lebih tinggi — sesuai ekspektasi karena kuantisasi yang lebih ringan menghasilkan gambar lebih mendekati asli. Namun penting dicatat bahwa PSNR tinggi **tidak menjamin** watermark berhasil diekstrak, karena PSNR mengukur kualitas gambar, bukan integritas LSB.*

### Ringkasan Semua Metrik

![Tahap 7d — Ringkasan Semua Metrik](output/tahap7d_ringkasan_semua_metrik.png)

---

## 11. Tabel Hasil Lengkap

| QF | BER | NC | PSNR (dB) | Status |
|:---:|:---:|:---:|:---:|:---:|
| 10 | 0.4998 | 0.2009 | 35.62 | ❌ GAGAL |
| 20 | 0.4751 | 0.5195 | 39.89 | ❌ GAGAL |
| 30 | 0.5267 | 0.5152 | 42.12 | ❌ GAGAL |
| 40 | 0.5308 | 0.4620 | 43.40 | ❌ GAGAL |
| 50 | 0.5007 | 0.3598 | 45.01 | ❌ GAGAL |
| 60 | 0.4544 | 0.5405 | 47.04 | ❌ GAGAL |
| 70 | 0.3702 | 0.6294 | 49.04 | ❌ GAGAL |
| 80 | 0.2893 | 0.7108 | 49.73 | ❌ GAGAL |
| 90 | 0.2446 | 0.7553 | 50.73 | ❌ GAGAL |
| **100** | **0.0400** | **0.9600** | **50.71** | **✅ BERHASIL** |

**Threshold:** BER ≤ 0.1 **DAN** NC ≥ 0.9

---

## 12. Analisis & Kesimpulan

### Mengapa watermark LSB gagal pada hampir semua QF?

Kompresi JPEG bekerja sebagai berikut:
1. Gambar dibagi blok 8×8 → DCT dilakukan per blok
2. Koefisien DCT dibagi tabel kuantisasi → dibulatkan ke bilangan bulat
3. Invers DCT menghasilkan nilai piksel yang **sedikit berbeda** dari aslinya

Perubahan kecil inilah — bahkan hanya ±1 atau ±2 pada nilai piksel — yang cukup untuk **membalik bit LSB** secara acak. Karena seluruh watermark disimpan di bit LSB, satu operasi kompresi JPEG sudah cukup untuk menghancurkan watermark hampir sepenuhnya.

### Mengapa QF 100 berhasil?

QF 100 menggunakan tabel kuantisasi dengan nilai pembagi yang sangat kecil (mendekati 1), sehingga hampir tidak ada pembulatan yang terjadi. Artefak kuantisasi sangat minimal, dan nilai piksel hasil dekompresi sangat dekat dengan aslinya — cukup untuk melestarikan banyak bit LSB. Namun BER masih 0.04 (4%), artinya 4% bit tetap berubah — QF 100 bukan lossless.

### Kesimpulan

| Aspek | Temuan |
|---|---|
| Metode | LSB (Least Significant Bit) domain spasial |
| Robustness terhadap JPEG | **Sangat rendah** — hanya bertahan di QF 100 |
| Invisibility | Sangat baik — PSNR 51.14 dB (tidak terdeteksi mata) |
| Threshold QF minimal | QF ≥ 100 (praktis tidak bisa dikompres) |
| Penyebab kegagalan | Kuantisasi JPEG merusak bit LSB secara acak |

### Rekomendasi Peningkatan

Untuk watermarking yang lebih tahan terhadap kompresi JPEG, disarankan menggunakan:

- **DCT Domain Watermarking** — menyisipkan watermark langsung ke koefisien DCT frekuensi menengah (bukan LSB piksel), sehingga tahan terhadap kuantisasi JPEG
- **DWT (Discrete Wavelet Transform)** — watermarking di domain wavelet yang lebih robust
- **Spread Spectrum Watermarking** — menyebarkan energi watermark ke banyak koefisien sehingga lebih tahan gangguan

---

## 13. Cara Menjalankan

### Prasyarat

```bash
pip install numpy pillow matplotlib scipy
```

### Menjalankan

```bash
# Pastikan foto_kakey.png ada di direktori yang sama
python watermarking.py
```

### Konfigurasi (opsional)

Edit bagian `KONFIGURASI` di awal `watermarking.py`:

```python
NAMA_FOTO       = "foto_kakey.png"   # ganti dengan nama foto kamu
BLOCK_SIZE      = 8                   # ukuran blok DCT (standar JPEG = 8)
WATERMARK_TILE  = 32                  # ukuran kotak pola watermark
QUALITY_FACTORS = [10, 20, ..., 100]  # daftar QF yang diuji
THRESHOLD_BER   = 0.1                 # batas BER untuk dinyatakan berhasil
THRESHOLD_NC    = 0.9                 # batas NC untuk dinyatakan berhasil
```

### Output yang dihasilkan

Setelah dijalankan, 13 file gambar dan 1 file `watermarked_original.png` akan tersimpan di direktori yang sama — siap dimasukkan ke README atau laporan.

---

*Dibuat dengan Python 3 · NumPy · Pillow · Matplotlib · SciPy*
