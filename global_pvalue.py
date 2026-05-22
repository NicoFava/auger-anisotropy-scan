import pandas as pd
import matplotlib.pyplot as plt
import os

def plotta_progressione_definitiva(file_log='penalizzazione_log.txt'):
    # Leggiamo tutto il file come una serie di dati grezzi
    if not os.path.exists(file_log):
        print(f"Errore: {file_log} non trovato.")
        return

    # Leggiamo il file ignorando le righe di intestazione
    data = []
    with open(file_log, 'r') as f:
        for line in f:
            if line.startswith("Batch_ID"): continue
            try:
                parts = line.strip().split('\t')
                # Ci interessano solo due colonne:
                # [1] = Cieli nel Batch, [3] = Superamenti nel Batch
                if len(parts) >= 4:
                    data.append([float(parts[1]), float(parts[3])])
            except ValueError:
                continue
    
    df = pd.DataFrame(data, columns=["Cieli_Batch", "Superamenti_Batch"])
    
    # Sommiamo tutto in modo sequenziale per vedere la convergenza
    # Creiamo un DataFrame che somma i cieli e i superamenti man mano
    df['Cieli_Cumulati'] = df['Cieli_Batch'].cumsum()
    df['Superamenti_Cumulativi'] = df['Superamenti_Batch'].cumsum()
    df['P_Value_Globale'] = df['Superamenti_Cumulativi'] / df['Cieli_Cumulati']

    plt.style.use('seaborn-v0_8-darkgrid') 
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['Cieli_Cumulati'], df['P_Value_Globale'], 
            color='#e67e22', linewidth=2.5, marker='.', markersize=4, label='P-Value Globale')
    
    # Linea orizzontale di riferimento (media finale)
    p_final = df['P_Value_Globale'].iloc[-1]
    ax.axhline(y=p_final, color='black', linestyle=':', alpha=0.6, label=f'Stima finale: {p_final:.2e}')
    
    ax.set_yscale('log')
    ax.set_xlabel('Numero Totale di Simulazioni (Sequenziale)', fontsize=12)
    ax.set_ylabel('P-Value Globale', fontsize=12)
    ax.set_title(f'Convergenza P-Value: Totale {int(df["Cieli_Cumulati"].iloc[-1]):,} simulazioni', 
                 fontsize=14, fontweight='bold')
    
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('progressione_definitiva_v2.png', dpi=300)
    print(f"Grafico aggiornato. P-Value finale su {int(df['Cieli_Cumulati'].iloc[-1]):,} simulazioni: {p_final:.3e}")

if __name__ == "__main__":
    plotta_progressione_definitiva()