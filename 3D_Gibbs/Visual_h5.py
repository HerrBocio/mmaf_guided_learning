import h5py

# Apri il file in sola lettura
with h5py.File('datasetGauSep1.01.02.02.08.h5', 'r') as f:
    # Stampa i nomi dei dataset all'interno del file
    print(list(f.keys()))
    
    # Accesso a un dataset
    dataset = f['data']
    data = dataset[:]  # Leggi tutti i dati
    print(data.shape)