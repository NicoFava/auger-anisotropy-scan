# AstroLab: Cosmic Ray Statistical Analysis

## Overview

**AstroLab** è una suite di strumenti Python per l'analisi statistica avanzata di dati di raggi cosmici ad altissima energia provenienti dal **Pierre Auger Observatory**. Il progetto implementa sofisticate tecniche Monte Carlo per identificare correlazioni anomale tra gli eventi di raggi cosmici osservati e sorgenti astrofisiche extragalattiche, con focus particolare su **Centaurus A (NGC 5128)**.

### Obiettivi Scientifici

- **Anisotropia cosmiche**: Identificare clustering non-isotropi di raggi cosmici attorno a sorgenti extragalattiche
- **Test statistici robusti**: Implementazione di test Monte Carlo pesati sulla mappa di esposizione strumentale
- **Analisi multi-scala**: Scan simultanei su 40 raggi angolari (da 1° a 40°)
- **Penalizzazione statistica**: Correzione per test multipli mantenendo potenza statistica

## Struttura del Progetto

```
AstroLab/
├── astro.py                      # Analisi base globale - test Monte Carlo
├── astroNGC5128.py              # Analisi ottimizzata con Numba (NGC 5128)
├── astroNGC5128CutE.py          # Analisi con filtro energetico
├── astroNGC5128sEDscan.py       # Scan bidimensionale energia-raggio
├── global_pvalue.py              # Calcolo p-value globale penalizzato
├── penalizzazione.py             # Visualizzazione convergenza p-value
├── auger.txt                     # Dataset events (RA, Dec, Energia)
├── exposure.fits                 # Mappa FITS esposizione strumentale
├── penalizzazione_log.txt        # Log simulazioni sequenziali
├── OldPlots/                     # Archivio risultati precedenti
├── plotsNGC5128/                 # Output analisi NGC 5128
├── plotsNGC5128_CutE/            # Output analisi NGC 5128 con filtro
└── plotsNGC5128_Scan2D/          # Output scan bidimensionali
```

## Descrizione dei Moduli

### 1. **astro.py** - Analisi Globale Monte Carlo
Implementa il test statistico base per l'identificazione di anisotropie:

**Funzioni principali:**
- `carica_dati()`: Caricamento e pulizia dataset Auger (RA, Dec, Energia)
- `calcola_distanza_angolare()`: Distanza sferica tra coppie di coordinate celesti
- `genera_cielo_isotropo_pesato()`: Generazione eventi Monte Carlo distribuiti secondo la mappa di esposizione
- `test_statistico_montecarlo_globale()`: Test simultaneo su tutti i raggi

**Algoritmo:**
1. Carica gli eventi osservati dal dataset
2. Genera N cieli isotropi simulati (default 10,000), rispettando il pattern di esposizione dello strumento
3. Per ogni sorgente e ogni raggio (1°-40°), conta gli eventi osservati
4. Confronta con la distribuzione simulata per calcolare p-value
5. Non applica penalizzazioni locali

### 2. **astroNGC5128.py** - Analisi Ottimizzata (Numba JIT)
Versione accelerata tramite compilazione JIT (Just-In-Time) con Numba per prestazioni produzione:

**Ottimizzazioni:**
- `@njit(parallel=True)`: Parallelizzazione multi-core dei loop Monte Carlo
- Inverse Transform Sampling sulla CDF dell'esposizione
- Coordinate in radianti per minimizzare operazioni floating-point
- Pre-calcolo della CDF per estrazione veloce

**Funzioni:**
- `prepara_dati_numba()`: Pre-calcolo strutture dati ottimizzate
- `esegui_simulazioni_numba_multi()`: Motore Monte Carlo Numba (parallelo)

**Vantaggi:**
- 10-100× più veloce di `astro.py` a parità di statistiche
- Permette 100,000+ simulazioni in tempo ragionevole

### 3. **astroNGC5128CutE.py** - Analisi con Filtro Energetico
Extends `astroNGC5128.py` con filtri energetici per selezionare sottocampioni di eventi:

**Feature:**
- Filtraggio su intervalli energetici (es. E > 60 EeV)
- Analisi della correlazione evento-energia con posizione sorgente
- Valutazione della significanza in funzione dell'energia

### 4. **astroNGC5128sEDscan.py** - Scan Bidimensionale Energia-Raggio
Implementa uno scan 2D simultaneo su energia e raggio angolare:

**Algoritmo:**
- Per ogni bin energetico (E_1, E_2):
  - Per ogni raggio da 1° a 40°:
    - Conta gli eventi osservati
    - Calcola la frazione di esposizione attesa
    - Valuta il p-value di Poisson

**Output:**
- Mappa 2D p-value(energia, raggio)
- Identificazione di picchi di significanza anomala
- Visualizzazione con scala logaritmica

### 5. **global_pvalue.py** - Penalizzazione Statistica Globale
Calcola il p-value globale corretto per il test multiplo:

**Metodologia:**
1. Esegue M batch sequenziali di simulazioni Monte Carlo
2. Per ogni batch: calcola quante simulazioni superano il max p-value osservato
3. Somma cumulativamente i superamenti
4. p-value_globale = superamenti_totali / simulazioni_totali

**Gestisce:**
- Correzione per look-elsewhere effect
- Convergenza statistica con >100,000 simulazioni
- Salvare dello stato in log per ripresa incrementale

### 6. **penalizzazione.py** - Visualizzazione Convergenza
Plotta il trend del p-value globale durante le simulazioni sequenziali:

**Output:**
- Grafico con asse Y in scala logaritmica
- Linea di convergenza mostrando stima finale
- Validazione della significanza statistica raggiunta

## Dataset e Configurazione

### Input Data

**auger.txt**: Tab-separated, formato:
```
#  RA         Dec        Energia(EeV)
   12.345     -45.678    78.5
   ...
```

**exposure.fits**: Mappa HEALPix FITS dell'esposizione strumentale in coordinate Galattiche

### Sorgenti Astronomiche Analizzate

Default: `Centaurus A (NGC 5128)`
- **RA**: 201.36° | **Dec**: -43.02°
- **Classe**: AGN (Active Galactic Nuclei) - Galassia ellittica gigante
- **Distanza**: ~3.7 Mpc
- **Significanza**: Potenziale sorgente di raggi cosmici ultra-ad alta energia

Altre sorgenti disponibili: M87 (Virgo A), Fornax A, NGC 253, M83, Vela SNR, Sagittarius A

## Tecnologie e Dipendenze

```python
numpy           # Calcoli numerici vettorizzati
pandas          # Manipolazione dati tabellari
matplotlib      # Visualizzazione
healpy          # Operazioni su mappe sferiche HEALPix
scipy.stats     # Distribuzioni statistiche (Poisson, normale)
astropy         # Coordinate astronomiche + trasformazioni
numba           # Compilazione JIT per Python -> C performance
```

## Utilizzo

### Setup Ambiente

```bash
# Attivare virtual environment (assumendo lab_env)
source /home/fava/lab/lab_env/bin/activate

# Installare dipendenze (se necessario)
pip install numpy pandas matplotlib healpy scipy astropy numba
```

### Esecuzione Analisi

**Analisi base globale** (lenta, per piccole simulazioni):
```bash
python astro.py
```

**Analisi NGC 5128 ottimizzata** (produzione):
```bash
python astroNGC5128.py
```

**Analisi con filtro energetico**:
```bash
python astroNGC5128CutE.py
```

**Scan bidimensionale**:
```bash
python astroNGC5128sEDscan.py
```

**Calcolo p-value globale penalizzato**:
```bash
python global_pvalue.py
```

**Visualizzazione convergenza**:
```bash
python penalizzazione.py
```

## Output e Risultati

### Directory Risultati

- **plotsNGC5128/**: Contiene report statistici e curve p-value per NGC 5128
- **plotsNGC5128_CutE/**: Risultati con filtri energetici
- **plotsNGC5128_Scan2D/**: Mappe 2D energia-raggio
- **OldPlots/**: Storico analisi precedenti (Centaurus A, M87, Fornax A, LMC, SMC, etc.)

### File di Output Tipici

```
report_statistico_SCAN_LOCALE.txt    # Resoconto p-value per raggi
pvalue_scans/                        # Grafici curve p-value
tophat_maps/                         # Mappe di densità eventi
progressione_definitiva_v2.png       # Convergenza p-value globale
```

## Metodologia Statistica

### Test di Anisotropia

**Ipotesi nulla (H₀)**: Gli eventi di raggi cosmici sono distribuiti isotropicamente secondo la mappa di esposizione.

**Ipotesi alternativa (H₁)**: Esiste clustering preferenziale verso specifiche sorgenti astronomiche.

### Approccio Monte Carlo Pesato su Esposizione

1. **Costruzione CDF esposizione**: 
   - Integra la mappa di esposizione in HEALPix
   - Normalizza a distribuzione di probabilità

2. **Generazione cieli simulati**:
   - Inverse Transform Sampling sulla CDF
   - Mantiene statistiche di esposizione osservate
   - Genera N_eventos ~ O(10,000-100,000)

3. **Conteggi e p-value**:
   - Per ogni raggio: n_oss = eventi osservati entro raggio
   - Per ogni simulazione: n_sim = eventi simulati entro raggio
   - p-value = P(N_sim ≥ n_oss | H₀)

4. **Penalizzazione globale**:
   - Correzione for "look-elsewhere effect"
   - p-value_globale con 100,000+ simulazioni

### Curve di Significanza

Espresse in sigma (σ) da una distribuzione normale:

$$\sigma = \Phi^{-1}(1 - p)$$

dove Φ⁻¹ è la funzione quantile della distribuzione normale standard.

## Note Scientifiche

### Assunzioni Critiche

1. **Isotropia baseline**: In assenza di sorgenti, i raggi cosmici dovrebbero distribuirsi uniformemente ponderati dall'esposizione
2. **Stabilità esposizione**: Assumiamo che la mappa di esposizione sia stabile nel periodo di osservazione
3. **Indipendenza eventi**: Non considerati correlazioni temporali tra eventi successivi

### Limitazioni

- **Sensibilità angolare**: Risoluzione limitata da angular resolution dell'osservatorio
- **Statistica bassa**: Piccolo numero di eventi a energie estreme (E > 100 EeV)
- **Sistematiche**: Possibili bias non modellati nella mappa di esposizione
- **Look-elsewhere effect**: Test multipli su 40 raggi riducono significanza locale

## Risultati Precedenti

Nel database `OldPlots/` sono conservate analisi su multiple sorgenti e campioni:
- Centaurus A: Report completo, mappe tophat per 9 sorgenti diverse
- Scan locale dettagliato (multipli raggi simultanei)
- Statistiche Monte Carlo con variazioni di metodologia

## Autore e Licenza

**Autore**: Nicolò Favagrossa  
**Licenza**: MIT (vedi LICENSE)  
**Versione**: 2026  
**Repository**: Private research project

## Contatti e Sviluppi Futuri

### Possibili Estensioni

1. **Correlazioni temporali**: Analisi di clustering temporale degli eventi
2. **Modelli spettrali**: Fit della distribuzione energetica attorno alle sorgenti
3. **Machine Learning**: Classificazione di eventi anomali
4. **Analisi multi-messenger**: Correlazione con osservazioni gravitazionali/neutrini
5. **Scan anisotropia globale**: Ricerca di anisotropie non associate a sorgenti specifiche

### Performance Attuale

- **astro.py**: 10,000 simulazioni → ~5-10 min (single core)
- **astroNGC5128.py**: 100,000 simulazioni → ~2-3 min (8 cores, con Numba)
- **Penalizzazione**: 1,000,000 simulazioni → ~30-40 min (distributed)

---

**Ultimo aggiornamento**: Maggio 2026  
**Status**: Production - Analysis pipeline active
