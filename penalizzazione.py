import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy.stats as stats
import time
from numba import njit, prange
from astropy.coordinates import SkyCoord
import astropy.units as u

def carica_dati(filepath):
    """Carica e pulisce il dataset di Auger."""
    df = pd.read_csv(filepath, sep='\t+', engine='python')
    df.columns = [c.strip().replace('#', '') for c in df.columns]
    return df

def calcola_distanza_angolare(ra1, dec1, ra2, dec2):
    """Calcola la distanza angolare sferica tra due punti in gradi."""
    ra1_rad, dec1_rad = np.radians(ra1), np.radians(dec1)
    ra2_rad, dec2_rad = np.radians(ra2), np.radians(dec2)
    
    cos_theta = (np.sin(dec1_rad) * np.sin(dec2_rad) + 
                 np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(ra1_rad - ra2_rad))
    cos_theta = np.clip(cos_theta, -1.0, 1.0) 
    return np.degrees(np.arccos(cos_theta))

def calcola_attesi_analitici(expo_map, ra_src, dec_src, max_raggio, n_tot_eventi):
    """Calcola il fondo esatto atteso. Ci serve per valutare i p-value locali dei cieli finti."""
    nside = hp.get_nside(expo_map)
    expo_pulita = np.where(expo_map > 0, expo_map, 0)
    tot_expo = np.sum(expo_pulita)
    
    c = SkyCoord(ra=ra_src*u.degree, dec=dec_src*u.degree, frame='icrs')
    l_gal = c.galactic.l.degree
    b_gal = c.galactic.b.degree
    vec_src = hp.ang2vec(np.radians(90 - b_gal), np.radians(l_gal))
    
    raggi = np.arange(1, max_raggio + 1)
    attesi = np.zeros(len(raggi))
    for i, r in enumerate(raggi):
        pixel_in_raggio = hp.query_disc(nside, vec_src, np.radians(r))
        frazione = np.sum(expo_pulita[pixel_in_raggio]) / tot_expo
        attesi[i] = n_tot_eventi * frazione
    return attesi

def prepara_dati_numba(expo_map):
    """Precalcola la CDF dell'esposizione e le coordinate dei pixel in radianti."""
    expo_pulita = np.where(expo_map > 0, expo_map, 0)
    probabilita_pixel = expo_pulita / np.sum(expo_pulita)
    
    pixel_cdf = np.cumsum(probabilita_pixel)
    
    n_pixel_totali = len(probabilita_pixel)
    nside = hp.npix2nside(n_pixel_totali)
    theta, phi = hp.pix2ang(nside, np.arange(n_pixel_totali))
    
    # Da Galattiche a Equatoriali
    l_gal = np.degrees(phi)
    b_gal = 90.0 - np.degrees(theta)
    c = SkyCoord(l=l_gal*u.degree, b=b_gal*u.degree, frame='galactic')
    
    pixel_ra_rad = c.icrs.ra.radian
    pixel_dec_rad = c.icrs.dec.radian
    
    return pixel_cdf, pixel_ra_rad, pixel_dec_rad

@njit(parallel=True, fastmath=True)
def esegui_simulazioni_numba_multi(n_simulazioni, n_eventi, pixel_cdf, pixel_ra_rad, pixel_dec_rad, srcs_ra, srcs_dec, max_raggio):
    """
    MOTORE PURO MONTE CARLO.
    Lancia coordinate sulla mappa e conta quanti eventi cadono nei 40 raggi.
    """
    n_src = len(srcs_ra)
    conteggi_simulati = np.zeros((n_src, n_simulazioni, max_raggio), dtype=np.int32)
    
    srcs_ra_rad = np.radians(srcs_ra)
    srcs_dec_rad = np.radians(srcs_dec)
    
    for i in prange(n_simulazioni):
        for j in range(n_eventi):
            u = np.random.rand()
            pix_idx = np.searchsorted(pixel_cdf, u)
            
            ra_rad = pixel_ra_rad[pix_idx]
            dec_rad = pixel_dec_rad[pix_idx]
            
            for k in range(n_src):
                cos_theta = (np.sin(dec_rad) * np.sin(srcs_dec_rad[k]) + 
                             np.cos(dec_rad) * np.cos(srcs_dec_rad[k]) * np.cos(ra_rad - srcs_ra_rad[k]))
                
                if cos_theta < -1.0: cos_theta = -1.0
                elif cos_theta > 1.0: cos_theta = 1.0
                
                dist = np.degrees(np.arccos(cos_theta))
                
                start_idx = int(np.ceil(dist)) - 1
                if start_idx < 0:
                    start_idx = 0
                
                for r_idx in range(start_idx, max_raggio):
                    conteggi_simulati[k, i, r_idx] += 1
                    
    return conteggi_simulati

def avvia_penalizzazione_continua(dataset, expo_map, ra_src, dec_src, target_pvalue, 
                                  max_raggio=40, cieli_per_batch=100000, n_batches=100,
                                  file_log="penalizzazione_log.txt"):
    """
    Esegue i batch di penalizzazione salvando man mano su file.
    """
    N_totali = len(dataset)
    
    print("\n=====================================================================")
    print(" INIZIO PENALIZZAZIONE GLOBALE (LOOK-ELSEWHERE EFFECT)")
    print(" Metodo: Puro Monte Carlo Spaziale a Blocchi")
    print("=====================================================================")
    print(f" Target (Best P-Value Reale) da battere : {target_pvalue:.3e}")
    print(f" Eventi totali per cielo (Niente tagli) : {N_totali}")
    print(f" Dimensione del singolo blocco (Batch)  : {cieli_per_batch:,} cieli")
    print(f" Numero di blocchi da eseguire          : {n_batches}")
    print(f" Totale cieli simulati alla fine        : {cieli_per_batch * n_batches:,}")
    print("=====================================================================\n")

    # Preparo i dati per Numba (come nel primo script)
    print("-> Preparazione CDF per estrazione Monte Carlo...")
    pixel_cdf, pixel_ra_rad, pixel_dec_rad = prepara_dati_numba(expo_map)
    
    srcs_ra = np.array([ra_src])
    srcs_dec = np.array([dec_src])

    # Calcolo la griglia di riferimento degli "Attesi" per valutare le fluttuazioni dei cieli finti
    print("-> Calcolo il background di riferimento per lo scoring...")
    attesi_array = calcola_attesi_analitici(expo_map, ra_src, dec_src, max_raggio, N_totali)

    # Inizializzo il file di log (se non esiste lo crea con l'intestazione)
    if not os.path.exists(file_log):
        with open(file_log, 'w') as f:
            f.write("Batch_ID\tCieli_nel_Batch\tTot_Cieli_Analizzati\tSuperamenti_nel_Batch\tTot_Superamenti\tP_Value_Globale\n")

    totale_cieli_analizzati = 0
    totale_superamenti = 0
    start_global_time = time.perf_counter()

    for batch_id in range(1, n_batches + 1):
        batch_start_time = time.perf_counter()
        
        # GENERAZIONE FINTI CIELI TRAMITE MONTE CARLO
        # Il motore lancia le coordinate casuali sulla sfera e bina i risultati nei 40 gradi
        conteggi_matrix = esegui_simulazioni_numba_multi(
            cieli_per_batch, N_totali, pixel_cdf, pixel_ra_rad, pixel_dec_rad, 
            srcs_ra, srcs_dec, max_raggio
        )
        conteggi_batch = conteggi_matrix[0] # Seleziono la prima (e unica) sorgente
        
        # SCORING SECONDO IL METODO DEL PROFESSORE
        # Calcoliamo i 40 p-values locali per TUTTI i cieli generati in un colpo solo
        # (Usiamo la formula di Poisson *solo* per assegnare un punteggio istantaneo al cielo generato)
        k_array = np.maximum(0, conteggi_batch - 1)
        p_values_simulati = stats.poisson.sf(k_array, attesi_array)
        
        # Troviamo il "Best Local" (P-Value minimo) per ogni singolo cielo in QUALSIASI raggio
        best_local_per_cielo = np.min(p_values_simulati, axis=1)
        
        # Contiamo quanti di questi cieli finti hanno prodotto un best_local MINORE o UGUALE al target reale
        superamenti_batch = np.sum(best_local_per_cielo <= target_pvalue)
        
        # AGGIORNAMENTO TOTALI E SALVATAGGIO
        totale_cieli_analizzati += cieli_per_batch
        totale_superamenti += superamenti_batch
        
        # Se ancora 0 superamenti, metto un limite superiore fittizio
        if totale_superamenti == 0:
            pval_globale_parziale = 1 / totale_cieli_analizzati
        else:
            pval_globale_parziale = totale_superamenti / totale_cieli_analizzati
        
        # Scrivo la riga nel log
        with open(file_log, 'a') as f:
            f.write(f"{batch_id}\t{cieli_per_batch}\t{totale_cieli_analizzati}\t{superamenti_batch}\t{totale_superamenti}\t{pval_globale_parziale:.3e}\n")
            
        tempo_batch = time.perf_counter() - batch_start_time
        print(f"[{batch_id:03d}/{n_batches}] Cieli cumulati: {totale_cieli_analizzati:,} | "
              f"Nuovi superamenti: {superamenti_batch} | Tot superamenti: {totale_superamenti} | "
              f"P-Value GLOBALE parziale: {pval_globale_parziale:.3e} ({tempo_batch:.1f}s)")

    tempo_totale = time.perf_counter() - start_global_time
    print("\n=====================================================================")
    print(" PROCESSO DI PENALIZZAZIONE COMPLETATO")
    print("=====================================================================")
    if totale_superamenti > 0:
        sigma_globale = stats.norm.isf(totale_superamenti / totale_cieli_analizzati)
        print(f" P-Value Globale Finale  : {totale_superamenti / totale_cieli_analizzati:.3e} ({sigma_globale:.2f} Sigma)")
    else:
        print(f" Nessun superamento trovato su {totale_cieli_analizzati:,} cieli.")
        min_sigma = stats.norm.isf(1 / totale_cieli_analizzati)
        print(f" P-Value Globale Finale  : < {1 / totale_cieli_analizzati:.3e} (> {min_sigma:.2f} Sigma)")
        
    print(f" Tempo totale impiegato  : {tempo_totale / 60:.2f} minuti")
    print(f" I log sono salvati in   : {file_log}")
    print("=====================================================================\n")

if __name__ == "__main__":
    
    file_dati = 'auger.txt'
    file_esposizione = 'exposure.fits'
    
    RA_CEN_A = 201.3
    DEC_CEN_A = -43.0
    
    # ------------------ PARAMETRI FONDAMENTALI ------------------
    TARGET_PVALUE_REALE = 9.8e-06
    
    # Quanti cieli per ciclo.
    CIELI_PER_BATCH = 100_000 
    
    # Quanti blocchi vuoi eseguire.
    NUMERO_DI_BATCH = 100       
    # ------------------------------------------------------------
    
    dataset = carica_dati(file_dati)
    
    
    try:
        expo_map = hp.read_map(file_esposizione, verbose=False)
    except Exception as e:
        print(f"ERRORE CRITICO: Impossibile caricare {file_esposizione}.")
        exit()
        
    avvia_penalizzazione_continua(
        dataset=dataset, 
        expo_map=expo_map, 
        ra_src=RA_CEN_A, 
        dec_src=DEC_CEN_A, 
        target_pvalue=TARGET_PVALUE_REALE,
        max_raggio=40,
        cieli_per_batch=CIELI_PER_BATCH,
        n_batches=NUMERO_DI_BATCH
    )