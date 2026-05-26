# Analisis Portofolio Saham LQ45 - R Shiny App

## Deskripsi
Aplikasi R Shiny interaktif untuk analisis portofolio saham pada indeks LQ45 (Bursa Efek Indonesia). Aplikasi ini menyediakan fitur lengkap mulai dari pengambilan data saham, analisis statistik, optimasi portofolio (Markowitz), hingga manajemen risiko.

## Fitur Utama

1. **Data & Saham** - Ambil data harga saham LQ45 dari Yahoo Finance
2. **Analisis Individual** - Statistik deskriptif, distribusi return, return kumulatif
3. **Korelasi** - Matriks korelasi antar saham
4. **Optimasi Portofolio** - Minimum Variance, Maximum Sharpe Ratio, Equal Weight
5. **Kinerja Portofolio** - Return tahunan, volatilitas, Sharpe ratio, pertumbuhan investasi
6. **Manajemen Risiko** - Value at Risk (VaR), CVaR, Maximum Drawdown

## Instalasi

### Prasyarat
- R versi >= 4.0
- RStudio (disarankan)

### Install Packages

```r
install.packages(c(
  "shiny",
  "shinydashboard",
  "quantmod",
  "PerformanceAnalytics",
  "plotly",
  "DT",
  "dplyr",
  "tidyr",
  "ggplot2",
  "corrplot",
  "quadprog",
  "moments"
))
```

## Cara Menjalankan

```r
# Set working directory ke folder project
setwd("path/to/project")

# Jalankan aplikasi
shiny::runApp("app.R")
```

Atau buka file `app.R` di RStudio dan klik tombol **Run App**.

## Cara Penggunaan

1. **Pilih Saham** - Di tab "Data & Saham", pilih minimal 2 saham LQ45
2. **Set Periode** - Tentukan tanggal mulai dan akhir analisis
3. **Ambil Data** - Klik "Ambil Data" untuk download data dari Yahoo Finance
4. **Analisis** - Jelajahi tab-tab analisis yang tersedia
5. **Optimasi** - Di tab "Optimasi Portofolio", pilih metode dan jalankan optimasi
6. **Risiko** - Di tab "Manajemen Risiko", hitung VaR dan metrik risiko lainnya

## Struktur File

```
├── app.R                    # Aplikasi Shiny utama (UI + Server)
├── portfolio_functions.R    # Fungsi-fungsi optimasi portofolio
├── INSTALL.md              # Dokumentasi instalasi
└── README.md               # README repository
```

## Metodologi

### Optimasi Portofolio (Markowitz Mean-Variance)
- **Minimum Variance**: Meminimalkan risiko portofolio
- **Maximum Sharpe Ratio**: Memaksimalkan return per unit risiko
- **Efficient Frontier**: Kurva portofolio optimal pada berbagai level risiko

### Manajemen Risiko
- **VaR (Value at Risk)**: Estimasi kerugian maksimum pada confidence level tertentu
- **CVaR (Conditional VaR)**: Rata-rata kerugian di luar VaR (Expected Shortfall)
- **Maximum Drawdown**: Penurunan maksimum dari puncak ke lembah

## Catatan
- Data diambil dari Yahoo Finance (memerlukan koneksi internet)
- Ticker saham LQ45 menggunakan format `.JK` (Jakarta Stock Exchange)
- Komposisi LQ45 diupdate setiap 6 bulan oleh BEI
