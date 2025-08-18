import string
from collections import Counter

def pulisci_testo(testo: str):
    """Lowercase + rimozione punteggiatura + split in parole."""
    return (
        testo.lower()
             .translate(str.maketrans('', '', string.punctuation))
             .split()
    )

def conta_righe(testo: str) -> int:
    """Conta numero totale di righe."""
    return len(testo.splitlines())

def conta_parole(testo: str) -> int:
    """Conta numero totale di parole (case-insensitive)."""
    return len(pulisci_testo(testo))

def parole_frequenti(testo: str, top_n: int = 5):
    """Restituisce le top-n parole più frequenti."""
    return Counter(pulisci_testo(testo)).most_common(top_n)

def main():
    try:
        with open("input.txt", "r", encoding="utf-8") as f:
            testo = f.read()
    except FileNotFoundError:
        print("Errore: file 'input.txt' mancante nella cartella corrente.")
        return

    print("Numero righe:", conta_righe(testo))
    print("Numero parole:", conta_parole(testo))
    print("Top-5 parole più frequenti:")
    for parola, n in parole_frequenti(testo, 5):
        print(f"{parola}: {n}")

if __name__ == "__main__":
    main()
