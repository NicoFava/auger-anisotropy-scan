import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cartelle di output
output_dir = 'plots'
separate_dir = os.path.join(output_dir, 'tophat_maps')
os.makedirs(separate_dir, exist_ok=True)

# Caricamento e pulizia dati
df = pd.read_csv('auger.txt', sep='\t+', engine='python')
df.columns = [c.strip().replace('#', '') for c in df.columns]

# --- DEFINIZIONE DELLA SORGENTE E DELLA TOP HAT ---
ra_sorgente = 266.4168  # gradi (Sagittarius A*)
dec_sorgente = -29.0078 # gradi

# 1. Convertiamo in radianti
ra_rad = np.radians(df['RA'])
dec_rad = np.radians(df['Dec'])
ra_src_rad = np.radians(ra_sorgente)
dec_src_rad = np.radians(dec_sorgente)

# 2. CALCOLO DELLA DISTANZA (FUORI DAL CICLO!)
# Si calcola una sola volta per tutto il dataset
cos_theta = (np.sin(dec_rad) * np.sin(dec_src_rad) + 
             np.cos(dec_rad) * np.cos(dec_src_rad) * np.cos(ra_rad - ra_src_rad))
cos_theta = np.clip(cos_theta, -1.0, 1.0) 
distanza_angolare = np.degrees(np.arccos(cos_theta))

raggio_top_hat = 0.0

# 3. CICLO SUI RAGGI DELLA TOP HAT
for i in range(40):
    raggio_top_hat += 1.0 # Aumenta il raggio di 1 grado
    
    # Applichiamo il filtro usando la distanza già calcolata
    df_sorgente = df[distanza_angolare <= raggio_top_hat]
    
    print(f"Raggio R={raggio_top_hat}° -> Eventi nella Top Hat: {len(df_sorgente)} / {len(df)}")

    # --- PLOT ---
    plt.figure(figsize=(10, 6))

    # Background (tutti gli eventi)
    plt.scatter(df['RA'], df['Dec'], color='gray', alpha=0.3, label='Tutti gli eventi', s=10)

    # Eventi nella Top Hat
    plt.scatter(df_sorgente['RA'], df_sorgente['Dec'], color='red', alpha=0.8, label=f'Eventi (R<={raggio_top_hat}°)', s=20)

    # Centro della sorgente
    plt.scatter(ra_sorgente, dec_sorgente, color='black', marker='*', s=200, label='Sagittarius A*')

    plt.title(f'Top Hat (R={raggio_top_hat}°) - Coordinate Celesti')
    plt.xlabel('Ascensione Retta (gradi)')
    plt.ylabel('Declinazione (gradi)')
    plt.xlim(0, 360)
    plt.ylim(-90, 90)
    # plt.gca().invert_xaxis() # Ripristinato: l'Ascensione Retta cresce verso sinistra
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Salvataggio
    filename = os.path.join(separate_dir, f"sky_map_tophat_R{int(raggio_top_hat)}.png")
    plt.savefig(filename)
    
    # CHIUSURA DELLA FIGURA (Fondamentale nei cicli!)
    plt.close()

print("Elaborazione e salvataggio mappe completato.")

# Creiamo una figura leggermente più larga per far spazio alla barra dei colori
plt.figure(figsize=(12, 7))

# Creiamo lo scatter plot. 
# - c=df['E'] dice a matplotlib di colorare i punti in base al valore dell'energia.
# - cmap='plasma' (o 'viridis', 'inferno', 'jet') stabilisce la scala di colori. 'plasma' va dal blu (bassa E) al giallo/bianco (alta E).
mappa_colori = plt.scatter(df['RA'], df['Dec'], c=df['E'], cmap='plasma', alpha=0.8, s=25)

# Aggiungiamo la barra dei colori (Colorbar) a destra del grafico
cbar = plt.colorbar(mappa_colori)
cbar.set_label('Energia [EeV]', fontsize=12)

# Mettiamo un marker nero a forma di stella per ricordarci dov'è Sagittarius A*
plt.scatter(ra_sorgente, dec_sorgente, color='black', marker='*', s=200, label='Sagittarius A*', edgecolors='white')

# Impostazioni degli assi e del titolo
plt.title("Mappa Celeste degli Eventi Auger (Intensità per Energia)", fontsize=14)
plt.xlabel('Ascensione Retta (gradi)', fontsize=12)
plt.ylabel('Declinazione (gradi)', fontsize=12)
plt.xlim(0, 360)
plt.ylim(-90, 90)

# Convenzione astronomica: RA cresce da destra verso sinistra
# plt.gca().invert_xaxis() 

plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

# Salviamo l'immagine nella cartella principale dei plot
heatmap_path = os.path.join(output_dir, 'sky_map_energy_heatmap.png')
plt.savefig(heatmap_path)
plt.close() # Chiudiamo la figura per liberare la memoria

print(f"Mappa dell'energia salvata con successo in: {heatmap_path}")