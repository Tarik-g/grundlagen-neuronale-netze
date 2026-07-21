# Grundlagen neuronaler Netze

Dieses Repository enthält die praktischen Implementierungen aus dem Modul **"Grundlagen neuronaler Netze"**.

Ziel der Aufgaben war es, zentrale Konzepte künstlicher neuronaler Netze nicht nur theoretisch zu verstehen, sondern eigenständig zu implementieren und deren Funktionsweise praktisch zu untersuchen.

**Modulergebnis: 1,3**

---

## Inhalte

## 1. Perzeptron-Lernregel

Implementierung der klassischen Perzeptron-Lernregel zur Lösung einer logischen AND-Funktion.

Behandelte Konzepte:

* künstliche Neuronen
* Perzeptron-Modell
* Stufenfunktion
* Perzeptron-Lernregel
* lineare Separierbarkeit
* Unterschied zwischen Perzeptron-Regel und Delta-Regel

---

## 2. iRProp− statt klassischem Gradientenabstieg

Anpassung eines mehrschichtigen neuronalen Netzes, sodass die Gewichte mit **improved Resilient Backpropagation (iRProp−)** aktualisiert werden.

Behandelte Konzepte:

* Feedforward-Netze
* Backpropagation
* Kettenregel
* Gradientenberechnung
* adaptive Schrittweiten
* RProp− Optimierungsverfahren

---

## 3. Restricted Boltzmann Machine (RBM)

Implementierung einer Restricted Boltzmann Machine als einfacher Autoencoder zur Rekonstruktion von MNIST-Ziffern.

Behandelte Konzepte:

* unüberwachtes Lernen
* generative Modelle
* Restricted Boltzmann Machines
* Contrastive Divergence (CD-1)
* Weight Sharing
* Encoder-Decoder-Prinzip

---

## 4. 2D Max-Pooling für Convolutional Neural Networks

Manuelle Implementierung eines zweidimensionalen Max-Pooling Layers ohne fertige Bibliotheksfunktion.

Behandelte Konzepte:

* CNN-Grundlagen
* Sliding-Window-Verfahren
* Pooling Size
* Stride
* Downsampling
* Ausgabegrößenberechnung

---

## 5. Rekurrentes neuronales Netz (RNN)

Simulation eines kleinen rekurrenten neuronalen Netzes mit zwei Neuronen und zeitlicher Rückkopplung.

Behandelte Konzepte:

* rekurrente Verbindungen
* Zustandsänderungen über die Zeit
* tanh-Aktivierungsfunktion
* synchrones Update
* dynamisches Verhalten neuronaler Systeme

---

# Installation

## Voraussetzungen

* Python 3.x
* pip

Es wird empfohlen, eine virtuelle Python-Umgebung zu verwenden.

---

## Virtuelle Umgebung erstellen

```bash
python -m venv venv
```

Virtuelle Umgebung aktivieren:

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

---

## Abhängigkeiten installieren

Nach Aktivierung der virtuellen Umgebung:

```bash
pip install -r requirements.txt
```

---

# Verwendete Bibliotheken

| Bibliothek | Verwendung                                    |
| ---------- | --------------------------------------------- |
| NumPy      | Matrixoperationen und numerische Berechnungen |
| Matplotlib | Visualisierung und Plot-Erstellung            |
| Pygame     | Grafische Darstellung einzelner Aufgaben      |
| Pillow     | Bildverarbeitung                              |

---

# Projektstruktur

```
grundlagen-neuronaler-netze/

├── README.md
├── requirements.txt
├── .gitignore

├── 01_perzeptron_lernregel/
│   └── Aufgabe1.py

├── 02_irprop_minus/
│   └── Aufgabe2.py

├── 03_rbm_contrastive_divergence/
│   └── Aufgabe3.py

├── 04_cnn_max_pooling/
│   └── Aufgabe4.py

└── 05_rnn/
    └── Aufgabe5.py
```

---

# Ausführung

Nach dem Starten der virtuellen Umgebung können die einzelnen Aufgaben direkt über Python gestartet werden.

```bash
py .\01_perzeptron_lernregel\Aufgabe1.py
py .\02_irprop_minus\Aufgabe2.py
py .\03_rbm_contrastive_divergence\Aufgabe3.py
py .\04_cnn_max_pooling\Aufgabe4.py
py .\05_rnn\Aufgabe5.py

# Themenübersicht

| Aufgabe | Thema                                                   |
| ------- | ------------------------------------------------------- |
| 1       | Perzeptron-Lernregel                                    |
| 2       | Backpropagation und iRProp−                             |
| 3       | Restricted Boltzmann Machine und Contrastive Divergence |
| 4       | CNN Max-Pooling                                         |
| 5       | Rekurrente neuronale Netze                              |

---

# Hinweis

Die Implementierungen entstanden im Rahmen des Moduls **"Grundlagen neuronaler Netze"** und dienen der praktischen Vertiefung der theoretischen Konzepte künstlicher neuronaler Netze.
