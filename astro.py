import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy.stats as stats

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

def genera_cielo_isotropo_pesato(expo_map, n_eventi):
    """
    Genera un set di eventi (RA, Dec) distribuiti in modo isotropo, 
    estratti proporzionalmente al 'peso' (esposizione) di ciascun pixel.
    """
    # Normalizzo la mappa di esposizione per ottenere una distribuzione di probabilità
    expo_pulita = np.where(expo_map > 0, expo_map, 0)
    probabilita_pixel = expo_pulita / np.sum(expo_pulita)
    
    # Numero totale di pixel nella mappa
    n_pixel_totali = len(probabilita_pixel)
    
    # Estrazione casuale pesata per ogni pixel
    indici_estratti = np.random.choice(n_pixel_totali, size=n_eventi, p=probabilita_pixel)
    
    # Convertiamo gli indici dei pixel in coordinate angolari (theta, phi)
    nside = hp.npix2nside(n_pixel_totali)
    theta, phi = hp.pix2ang(nside, indici_estratti)
    
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)
    
    return ra, dec

def test_statistico_montecarlo_globale(df, expo_map, sorgenti_dict, max_raggio=40, n_simulazioni=10000):
    """
    Esegue un'analisi Monte Carlo valutando CONTEMPORANEAMENTE tutti i raggi da 1° a 40°.
    Non applica penalizzazioni, restituisce semplicemente l'andamento del p-value locale.
    """
    N_totali = len(df)
    raggi = np.arange(1, max_raggio + 1) # Array [1, 2, ... 40]
    risultati = {}
    
    print(f"   -> Calcolo eventi reali osservati per raggi da 1° a {max_raggio}°...")
    for nome, coord in sorgenti_dict.items():
        dist_reale = calcola_distanza_angolare(df['RA'], df['Dec'], coord['RA'], coord['Dec'])
        dist_reale_sorted = np.sort(dist_reale) # Ordinare serve per far funzionare searchsorted velocemente
        
        # Conta gli eventi <= raggio per tutti i 40 raggi in un colpo solo
        n_oss_array = np.searchsorted(dist_reale_sorted, raggi, side='right')
        
        risultati[nome] = {
            'raggi': raggi,
            'osservati_array': n_oss_array,
            # Matrice per salvare le simulazioni: righe=simulazioni, colonne=raggi
            'conteggi_simulati_array': np.zeros((n_simulazioni, max_raggio)) 
        }

    print(f"   -> Avvio simulazioni Monte Carlo globali ({n_simulazioni} cieli totali)...")
    for i in range(n_simulazioni):
        if (i+1) % 1000 == 0: 
            print(f"      ... Generato cielo finto {i+1}/{n_simulazioni}")
            
        ra, dec = genera_cielo_isotropo_pesato(expo_map, N_totali)
        
        for nome, coord in sorgenti_dict.items():
            dist = calcola_distanza_angolare(ra, dec, coord['RA'], coord['Dec'])
            dist_sorted = np.sort(dist)
            
            # Contiamo gli eventi finti per tutti i 40 raggi
            n_in_tophat_array = np.searchsorted(dist_sorted, raggi, side='right')
            risultati[nome]['conteggi_simulati_array'][i, :] = n_in_tophat_array

    print("   -> Calcolo delle curve di p-value completato.")
    for nome in risultati:
        n_oss = risultati[nome]['osservati_array']
        conteggi_sim = risultati[nome]['conteggi_simulati_array']
        
        # Valori attesi per ogni raggio (media delle colonne)
        n_attesi = np.mean(conteggi_sim, axis=0)
        
        # Calcolo P-value: per ogni raggio, quante sim hanno battuto l'osservazione?
        simulazioni_superiori = np.sum(conteggi_sim >= n_oss, axis=0)
        
        p_values = (simulazioni_superiori) / (n_simulazioni)
        
        sigmas = stats.norm.isf(p_values)
        sigmas[np.isinf(sigmas) | (sigmas < 0)] = 0.0
        
        # Troviamo per comodità il raggio dove il p-value è minimo assoluto
        idx_min = np.argmin(p_values)
        
        risultati[nome].update({
            'attesi_array': n_attesi,
            'p_values_array': p_values,
            'sigmas_array': sigmas,
            'raggio_minimo': raggi[idx_min],
            'p_value_minimo': p_values[idx_min],
            'sigma_massimo': sigmas[idx_min]
        })
        
    return risultati

def plot_pvalue_scan(risultati_scan, base_dir='plots'):
    """Genera e salva i grafici del P-Value in funzione del raggio per ogni sorgente."""
    print("\n---> Generazione dei grafici di scan del P-Value...")
    out_dir = os.path.join(base_dir, 'pvalue_scans')
    os.makedirs(out_dir, exist_ok=True)
    
    for nome, dati in risultati_scan.items():
        plt.figure(figsize=(10, 6))
        raggi = dati['raggi']
        p_values = dati['p_values_array']
        
        plt.plot(raggi, p_values, marker='o', linestyle='-', color='blue', alpha=0.8)
        
        # Linea orizzontale a 3 sigma per riferimento visivo
        plt.axhline(y=0.0013, color='red', linestyle='--', label='Soglia 3 Sigma')
        
        plt.yscale('log')
        plt.xlabel('Raggio Top Hat (°)', fontsize=12)
        plt.ylabel('P-Value Locale (Monte Carlo)', fontsize=12)
        plt.title(f'Andamento P-Value in funzione del raggio: {nome}', fontsize=14)
        
        # Evidenziamo il punto minimo
        r_min = dati['raggio_minimo']
        p_min = dati['p_value_minimo']
        plt.scatter([r_min], [p_min], color='red', s=100, zorder=5, label=f'Minimo a {r_min}°')
        
        plt.legend(loc='upper left')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        nome_file = f"scan_{nome.replace(' ', '_').replace('*', '')}.png"
        plt.savefig(os.path.join(out_dir, nome_file))
        plt.close()
    
    print(f"Grafici di scan salvati nella cartella: {out_dir}")

def analisi_tophat_sorgente(df, nome_sorgente, ra_src, dec_src, max_raggio=40, base_dir='plots'):
    """Esegue l'analisi Top Hat per una singola sorgente e salva le mappe."""
    print(f"\n---> Generazione mappe Top Hat per: {nome_sorgente}")
    
    nome_cartella = nome_sorgente.replace(" ", "_").replace("*", "")
    out_dir = os.path.join(base_dir, 'tophat_maps', nome_cartella)
    os.makedirs(out_dir, exist_ok=True)
    
    distanza = calcola_distanza_angolare(df['RA'], df['Dec'], ra_src, dec_src)
    
    raggio = 0.0
    for i in range(max_raggio):
        raggio += 1.0
        df_selezionato = df[distanza <= raggio]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df['RA'], df['Dec'], color='gray', alpha=0.3, label='Tutti gli eventi', s=10)
        plt.scatter(df_selezionato['RA'], df_selezionato['Dec'], color='red', alpha=0.8, label=f'Eventi (R<={raggio}°)', s=20)
        plt.scatter(ra_src, dec_src, color='black', marker='*', s=200, label=nome_sorgente)

        plt.title(f'Top Hat (R={raggio}°) - {nome_sorgente}')
        plt.xlabel('Ascensione Retta (gradi)')
        plt.ylabel('Declinazione (gradi)')
        plt.xlim(0, 360)
        plt.ylim(-90, 90)
        # plt.gca().invert_xaxis() 
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.savefig(os.path.join(out_dir, f"map_R{int(raggio)}.png"))
        plt.close()

def mappa_calore_globale(df, dizionario_sorgenti, base_dir='plots'):
    """Crea la heatmap dell'energia e ci posiziona sopra TUTTE le sorgenti analizzate."""
    print("\n---> Generazione mappa celeste dell'energia globale in corso...")
    os.makedirs(base_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    mappa_colori = plt.scatter(df['RA'], df['Dec'], c=df['E'], cmap='plasma', alpha=0.8, s=25)
    
    cbar = plt.colorbar(mappa_colori)
    cbar.set_label('Energia [EeV]', fontsize=12)

    colori_marker = ['black', 'red', 'cyan', 'magenta', 'yellow', 'lime', 'orange', 'green', 'blue']
    for idx, (nome, coordinate) in enumerate(dizionario_sorgenti.items()):
        colore = colori_marker[idx % len(colori_marker)]
        plt.scatter(coordinate['RA'], coordinate['Dec'], color=colore, marker='*', 
                    s=300, label=nome, edgecolors='black')

    plt.title("Mappa Celeste degli Eventi Auger (Intensità per Energia)", fontsize=14)
    plt.xlabel('Ascensione Retta (gradi)')
    plt.ylabel('Declinazione (gradi)')
    plt.xlim(0, 360)
    plt.ylim(-90, 90)
    # plt.gca().invert_xaxis() 
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0)) 
    plt.grid(True, linestyle='--', alpha=0.5)
    
    heatmap_path = os.path.join(base_dir, 'sky_map_energy_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight') 
    plt.close()
    print(f"Mappa dell'energia globale salvata in: {heatmap_path}")

def mappa_eventi_casuali(expo_map, dizionario_sorgenti, n_eventi=2635, base_dir='plots'):
    """
    Genera UN cielo finto usando la mappa di esposizione e lo disegna su 
    una mappa sferica Mollweide, aggiungendo le sorgenti come riferimento.
    Serve come check visivo per assicurarsi che l'estrazione Monte Carlo funzioni.
    """
    print(f"\n---> Generazione mappa di check: Cielo finto ({n_eventi} eventi) su Esposizione...")
    os.makedirs(base_dir, exist_ok=True)
    
    ra, dec = genera_cielo_isotropo_pesato(expo_map, n_eventi)
    
    plt.figure(figsize=(12, 7))
    
    hp.mollview(expo_map, hold=True, title=f"Cielo Finto Monte Carlo ({n_eventi} eventi)", 
                cmap='viridis', unit='Esposizione Relativa', coord=['C'])
    hp.graticule() # Aggiunge la griglia (meridiani e paralleli)

    hp.projscatter(ra, dec, lonlat=True, coord='C', 
                   marker='o', facecolors='white', edgecolors='black', linewidths=0.5,
                   s=30, alpha=0.8, label='Eventi Simulati')

    colori_marker = ['black', 'red', 'cyan', 'magenta', 'yellow', 'lime', 'orange', 'green', 'blue']
    for idx, (nome, coordinate) in enumerate(dizionario_sorgenti.items()):
        colore = colori_marker[idx % len(colori_marker)]
        hp.projscatter(coordinate['RA'], coordinate['Dec'], lonlat=True, coord='C',
                       color=colore, marker='*', s=300, edgecolors='white', label=nome)

    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
    
    map_path = os.path.join(base_dir, 'check_montecarlo_cielo_finto.png')
    plt.savefig(map_path, bbox_inches='tight')
    plt.close()
    
    print(f"Mappa di check del cielo finto salvata in: {map_path}")

if __name__ == "__main__":
    
    sorgenti_da_analizzare = {
        "Sagittarius A*":     {"RA": 266.4168, "Dec": -29.0078},
        "Centaurus A":        {"RA": 201.3,    "Dec": -43.0},
        "Fornax A":           {"RA": 50.67,    "Dec": -37.2},
        "NGC 253":            {"RA": 11.89,    "Dec": -25.29},
        "M83":                {"RA": 253.47,   "Dec": -24.38},
        "M87 (Virgo A)":      {"RA": 187.7,    "Dec": 12.39},
        "Vela SNR":           {"RA": 128.4,    "Dec": -45.18},
        "LMC":                {"RA": 80.89,    "Dec": -69.76},
        "SMC":                {"RA": 13.16,    "Dec": -72.8}
    }
    
    cartella_output = 'plots'
    file_dati = 'auger.txt'
    file_esposizione = 'exposure.fits'
    
    raggio_di_ricerca = 15.0  
    NUM_SIMULAZIONI = 100000
    
    os.makedirs(cartella_output, exist_ok=True)
    
    dataset = carica_dati(file_dati)
    print(f"Dati caricati: {len(dataset)} eventi trovati.")
    
    try:
        expo_map = hp.read_map(file_esposizione)
        nside = hp.get_nside(expo_map) # indica come sono costruiti i pixel della mappa
        print(f"Mappa Healpix caricata con successo (NSIDE={nside}).")
    except Exception as e:
        print(f"ERRORE CRITICO: Impossibile caricare {file_esposizione}. Assicurati che sia nella cartella.")
        exit()
        
    '''for nome, coordinate in sorgenti_da_analizzare.items():
        analisi_tophat_sorgente(
            df=dataset, 
            nome_sorgente=nome, 
            ra_src=coordinate['RA'], 
            dec_src=coordinate['Dec'], 
            max_raggio=40, 
            base_dir=cartella_output
        )'''
    print(f"\n---> Inizio Analisi Statistica Globale (Scan 1°-40°)...")
    risultati_statistici = test_statistico_montecarlo_globale(
        df=dataset, 
        expo_map=expo_map, 
        sorgenti_dict=sorgenti_da_analizzare, 
        max_raggio=40,  # Fissiamo il raggio massimo dello scan a 40°
        n_simulazioni=NUM_SIMULAZIONI
    )
    
    # Plottiamo le curve P-Value vs Raggio
    plot_pvalue_scan(risultati_statistici, base_dir=cartella_output)
        
    percorso_report = os.path.join(cartella_output, 'report_statistico_SCAN_LOCALE.txt')
    
    with open(percorso_report, 'w') as f_out:
        f_out.write("=========================================================================\n")
        f_out.write(f"  REPORT STATISTICO (Valori al raggio di massimo eccesso locale)\n")
        f_out.write(f"  Metodo: Monte Carlo ({NUM_SIMULAZIONI} cataloghi simulati globali)\n")
        f_out.write("=========================================================================\n")
        f_out.write(f"{'Sorgente':<18} | {'R-Opt':<5} | {'Sigma-Loc':<10} | {'P-Value-Loc'}\n")
        f_out.write("-" * 60 + "\n")
    
        for nome, dati in risultati_statistici.items():
            riga_report = (f"{nome[:18]:<18} | "
                           f"{dati['raggio_minimo']:>3}°  | "
                           f"{dati['sigma_massimo']:<10.3f} | "
                           f"{dati['p_value_minimo']:.3e}\n")
            f_out.write(riga_report)
            print(f"   -> {nome}: Il segnale più forte è a {dati['raggio_minimo']}° ({dati['sigma_massimo']:.2f} sigma locali)")
            
        f_out.write("=========================================================================\n")
                
    mappa_calore_globale(dataset, sorgenti_da_analizzare, cartella_output)
    
    mappa_eventi_casuali(
        expo_map=expo_map, 
        dizionario_sorgenti=sorgenti_da_analizzare, 
        n_eventi=len(dataset), 
        base_dir=cartella_output
    )
    
    print("\n---------------------------------------------------")
    print("TUTTE LE ANALISI SONO STATE COMPLETATE CON SUCCESSO!")
    print(f"I risultati statistici aggiornati sono in: {percorso_report}")