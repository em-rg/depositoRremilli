import text_analysis

def test_conta_righe():
    testo = "ciao\ncome va\n"
    assert text_analysis.conta_righe(testo) == 2

def test_conta_parole():
    testo = "Ciao! Ciao, test."
    assert text_analysis.conta_parole(testo) == 3

def test_parole_frequenti():
    testo = "Ciao ciao test"
    top = text_analysis.parole_frequenti(testo, 2)
    assert top[0][0] == "ciao"
    assert top[0][1] == 2
