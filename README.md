# Invisible Watermarking

Implementasi invisible watermarking pada citra wajah menggunakan metode LSB (Least Significant Bit) dengan evaluasi ketahanan terhadap kompresi JPEG pada berbagai nilai Quality Factor (QF).

---

## Deskripsi

Project ini menyisipkan watermark yang tidak terlihat oleh mata manusia ke dalam citra grayscale menggunakan teknik LSB. Watermark disisipkan pada bit paling rendah (LSB) setiap piksel sehingga perubahan visual tidak dapat dideteksi secara kasat mata.

Ketahanan watermark dievaluasi dengan mengkompres citra menggunakan kompresi JPEG pada berbagai nilai Quality Factor (QF = 10, 20, ..., 100), kemudian watermark diekstraksi kembali dan diukur menggunakan metrik BER dan NC.

---

## Struktur File

```
invisible-watermarking/
├── watermarking/
│   └── Watermarking_18224054_KaylaFiyazaNZ.py
├── foto_kakey.png
├── .gitignore
├── LICENSE
└── README.md
```

---

## Cara Kerja

### 1. Load Gambar
Citra wajah dibaca dan dikonversi ke grayscale.

### 2. Buat Watermark Biner
Watermark dibuat dalam bentuk pola kotak-kotak hitam putih (checkerboard) berukuran 32x32 piksel.

### 3. Embed Watermark (LSB)
Watermark disisipkan ke LSB setiap piksel menggunakan operasi bitwise:
```
piksel_baru = (piksel_asli & 0xFE) | bit_watermark
```
Perubahan nilai piksel maksimal 1, sehingga tidak terlihat oleh mata manusia.

### 4. Kompresi JPEG
Citra ter-watermark dikompres menggunakan kompresi JPEG dengan QF = 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.

### 5. Ekstraksi Watermark
Watermark diekstraksi kembali dari LSB citra hasil kompresi:
```
bit_watermark = piksel & 1
```

### 6. Evaluasi
Kualitas ekstraksi diukur menggunakan:
- BER (Bit Error Rate): proporsi bit yang salah, makin rendah makin bagus
- NC (Normalized Correlation): kemiripan watermark, makin mendekati 1 makin bagus

---

## Hasil Evaluasi

| QF | BER | NC | Status |
|---|---|---|---|
| 10 | ~0.50 | ~0.20 | GAGAL |
| 20-90 | ~0.25-0.50 | ~0.50-0.74 | GAGAL |
| 100 | ~0.04 | ~0.96 | Berhasil |

Kesimpulan: Watermark LSB hanya dapat diekstrak pada QF = 100. Pada QF 90 ke bawah, kompresi JPEG mengubah nilai LSB piksel sehingga watermark tidak dapat diekstrak.

---

## Requirements

```
pip install numpy pillow matplotlib scipy
```

---

## Cara Menjalankan

```bash
python watermarking/Watermarking_18224054_KaylaFiyazaNZ.py
```

Pastikan file `foto_kakey.png` berada di folder utama repo.

---

## Informasi

| | |
|---|---|
| Nama | Kayla Fiyaza Nawal Zaghbi |
| NIM | 18224054 |
| Metode | LSB (Least Significant Bit) |
| Bahasa | Python 3 |
