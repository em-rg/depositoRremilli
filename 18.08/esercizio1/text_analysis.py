import string
from collections import Counter

def pulisci_testo(testo: str):
    """TODO: minuscole + rimozione punteggiatura + split"""
    return []  # stub

def conta_righe(testo: str) -> int:
    """TODO: conta le righe"""
    return 0  # stub

def conta_parole(testo: str) -> int:
    """TODO: conta le parole"""
    return 0  # stub

def parole_frequenti(testo: str, top_n: int = 5):
    """TODO: top-n parole più frequenti"""
    return []  # stub

def main():
    try:
        with open("input.txt", "r", encoding="utf-8") as f:
            testo = f.read()
    except FileNotFoundError:
        print("Errore: file 'input.txt' mancante nella cartella corrente.")
        return

    # placeholder output
    print("Numero righe: (da implementare)")
    print("Numero parole: (da implementare)")
    print("Top-5 parole più frequenti: (da implementare)")

if __name__ == "__main__":
    main()
