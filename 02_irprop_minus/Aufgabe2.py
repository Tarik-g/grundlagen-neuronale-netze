# Aufgabe 2
# Unter den Beispielen aus meinem Buch finden Sie sowohl einen
# Backpropagation-Algorithmus als auch einen Beispielcode für den
# iRProp−-Algorithmus. Verändern Sie den Backpropagation-
# Algorithmus so, dass er den Gradientenabstieg vollständig mit
# iRProp− macht.

import numpy as np

# Sigmoide Aktivierungsfunktion und ihre Ableitung
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return x * (1 - x)

# XOR-Daten (Bias bereits im Input enthalten)
inp = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])
target = np.array([[0], [1], [1], [0]])

# Architektur des neuronalen Netzes
inp_size = 3   # Eingabeneuronen
hid_size = 4   # Hidden-Neuronen
out_size = 1   # Ausgabeneuron

# Gewichte zufällig initialisieren
# weightsInputToHidden = Gewichte Input → Hidden Layer
weightsInputToHidden = np.random.randn(inp_size, hid_size) * 0.5

# weightsHiddenToOutput = Gewichte Hidden → Output Layer (+ Bias)
weightsHiddenToOutput = np.random.randn(hid_size + 1, out_size) * 0.5


## ---------------- iRProp- Parameter ----------------

# Jedes Gewicht hat seine EIGENE Schrittweite (delta).
# Anders als beim normalen Gradientenabstieg (ein globales Lernrate für alle),
# passt iRProp- die Schrittweite pro Gewicht individuell an.
# Startwert 0.1 für alle Gewichte → alle starten gleich, entwickeln sich dann unterschiedlich.
stepSizeInputToHidden  = np.ones_like(weightsInputToHidden)  * 0.1   # Schrittweiten für weightsInputToHidden  (Input → Hidden)
stepSizeHiddenToOutput = np.ones_like(weightsHiddenToOutput) * 0.1   # Schrittweiten für weightsHiddenToOutput (Hidden → Output)

# Die Gradienten des VORHERIGEN Schritts werden gespeichert,
# um sie mit dem aktuellen Gradienten zu vergleichen.
# Entscheidend ist das VORZEICHEN: hat sich der Gradient gedreht?
#   → gleiches Vorzeichen: wir sind noch auf dem richtigen Weg → schneller werden
#   → anderes Vorzeichen: wir haben ein Minimum übersprungen → langsamer werden
# Startwert 0 (= kein vorheriger Gradient im ersten Schritt)
prevGradientInputToHidden  = np.zeros_like(weightsInputToHidden)
prevGradientHiddenToOutput = np.zeros_like(weightsHiddenToOutput)

# Faktoren, mit denen die Schrittweite multipliziert wird:
eta_plus  = 1.2   # Gradient hat gleiches Vorzeichen → Schrittweite um 20% erhöhen
eta_minus = 0.5   # Gradient hat anderes Vorzeichen  → Schrittweite halbieren

# Schrittweiten dürfen nicht unbegrenzt wachsen oder gegen 0 schrumpfen,
# da das Training sonst explodiert oder komplett stagniert.
d_max = 50    # Obergrenze: verhindert, dass ein Gewicht riesige Sprünge macht
d_min = 1e-6  # Untergrenze: verhindert, dass die Schrittweite effektiv 0 wird

# ---------------------------------------------------


def forward():
    inputLayer = inp  # Eingabedaten (4×3): alle 4 XOR-Beispiele auf einmal (Batch)

    # Matrixmultiplikation: jedes Eingabemuster × Gewichtsmatrix → gewichtete Summe
    # sigmoid drückt das Ergebnis in den Bereich (0, 1)
    # hiddenLayer hat Form (4×4): 4 Beispiele, 4 Hidden-Neuronen
    hiddenLayer = sigmoid(inputLayer @ weightsInputToHidden)

    # iRProp- braucht kein explizites Bias-Gewicht in weightsInputToHidden,
    # stattdessen wird eine Spalte mit 1en vor hiddenLayer gehängt.
    # hiddenLayerWithBias hat Form (4×5): Spalte 0 = Bias (immer 1), Spalten 1–4 = Hidden-Aktivierungen
    hiddenLayerWithBias = np.concatenate([np.ones((hiddenLayer.shape[0], 1)), hiddenLayer], axis=1)

    # Gleiche Logik wie oben: hiddenLayerWithBias × weightsHiddenToOutput → sigmoid → Ausgabe
    # outputLayer hat Form (4×1): eine Wahrscheinlichkeit pro XOR-Beispiel
    outputLayer = sigmoid(hiddenLayerWithBias @ weightsHiddenToOutput)

    return inputLayer, hiddenLayer, hiddenLayerWithBias, outputLayer  # alle Layer zurückgeben, da Backprop sie alle braucht


for i in range(100):  # 100 Trainingsdurchläufe (Epochen)

    inputLayer, hiddenLayer, hiddenLayerWithBias, outputLayer = forward()  # Vorwärtsdurchlauf: Ausgabe berechnen

    # Wie weit liegt die Ausgabe vom Zielwert entfernt?
    error = target - outputLayer           # Differenz pro Beispiel (positiv = zu klein, negativ = zu groß)
    # Durchschnitt (arithmetisches Mittel)
    loss  = np.mean(error ** 2)            # Wert der die Güte des Netzes beschreibt

    # --- Backpropagation: Gradienten berechnen ---

    # Gradient am Ausgang: Fehler × Ableitung der Sigmoidfunktion am Ausgang
    # "Wie stark muss sich outputLayer ändern, damit der Fehler kleiner wird?"
    gradientOutputLayer = error * deriv_sigmoid(outputLayer)   # Form (4×1)

    # Gradient für weightsHiddenToOutput: hiddenLayerWithBias.T @ gradientOutputLayer
    # "Wie stark trägt jedes Gewicht in weightsHiddenToOutput zum Fehler bei?"
    gradientWeightsHiddenToOutput = hiddenLayerWithBias.T @ gradientOutputLayer   # Form (5×1), ein Gradientenwert pro Gewicht in weightsHiddenToOutput

    # Fehler zurück durch weightsHiddenToOutput in den Hidden Layer propagieren
    # [:, 1:] entfernt die Bias-Spalte (Bias hat keine Vorgängerschicht)
    # × deriv_sigmoid(hiddenLayer): Ableitung der Sigmoidfunktion im Hidden Layer
    gradientHiddenLayer = (gradientOutputLayer @ weightsHiddenToOutput.T)[:, 1:] * deriv_sigmoid(hiddenLayer)   # Form (4×4)

    # Gradient für weightsInputToHidden: analog zu gradientWeightsHiddenToOutput
    gradientWeightsInputToHidden = inputLayer.T @ gradientHiddenLayer   # Form (3×4), ein Gradientenwert pro Gewicht in weightsInputToHidden


    # --- iRProp- Update für weightsHiddenToOutput ---

    # Vorzeichenvergleich: prevGradientHiddenToOutput * gradientWeightsHiddenToOutput
    #   > 0 → beide gleich  → richtiger Weg, beschleunigen
    #   < 0 → Vorzeichenwechsel → Minimum übersprungen, bremsen
    #   = 0 → erster Schritt oder Gradient war 0
    signHiddenToOutput = prevGradientHiddenToOutput * gradientWeightsHiddenToOutput

    for idx in np.ndindex(weightsHiddenToOutput.shape):   # jedes Gewicht einzeln aktualisieren
        if signHiddenToOutput[idx] > 0:
            stepSizeHiddenToOutput[idx] = min(stepSizeHiddenToOutput[idx] * eta_plus, d_max)              # Schrittweite erhöhen (max. d_max)
            weightsHiddenToOutput[idx] += np.sign(gradientWeightsHiddenToOutput[idx]) * stepSizeHiddenToOutput[idx]   # Schritt in Gradientenrichtung
            prevGradientHiddenToOutput[idx] = gradientWeightsHiddenToOutput[idx]                          # aktuellen Gradienten merken

        elif signHiddenToOutput[idx] < 0:
            stepSizeHiddenToOutput[idx] = max(stepSizeHiddenToOutput[idx] * eta_minus, d_min)             # Schrittweite verkleinern (min. d_min)
            # Kein Gewichtsupdate! (das „−" in iRProp−)
            prevGradientHiddenToOutput[idx] = 0   # Gradient auf 0 setzen → nächster Schritt landet im else-Zweig

        else:
            # Erster Schritt: einfach in Gradientenrichtung gehen, Schrittweite unverändert
            weightsHiddenToOutput[idx] += np.sign(gradientWeightsHiddenToOutput[idx]) * stepSizeHiddenToOutput[idx]
            prevGradientHiddenToOutput[idx] = gradientWeightsHiddenToOutput[idx]


    # --- iRProp- Update für weightsInputToHidden (identische Logik wie weightsHiddenToOutput) ---

    signInputToHidden = prevGradientInputToHidden * gradientWeightsInputToHidden

    for idx in np.ndindex(weightsInputToHidden.shape):
        if signInputToHidden[idx] > 0:
            stepSizeInputToHidden[idx] = min(stepSizeInputToHidden[idx] * eta_plus, d_max)
            weightsInputToHidden[idx] += np.sign(gradientWeightsInputToHidden[idx]) * stepSizeInputToHidden[idx]
            prevGradientInputToHidden[idx] = gradientWeightsInputToHidden[idx]

        elif signInputToHidden[idx] < 0:
            stepSizeInputToHidden[idx] = max(stepSizeInputToHidden[idx] * eta_minus, d_min)
            prevGradientInputToHidden[idx] = 0   # Gradient zurücksetzen, kein Gewichtsupdate

        else:
            weightsInputToHidden[idx] += np.sign(gradientWeightsInputToHidden[idx]) * stepSizeInputToHidden[idx]
            prevGradientInputToHidden[idx] = gradientWeightsInputToHidden[idx]


    if i % 5 == 0:   # alle 5 Iterationen Zwischenstand ausgeben
        print(f"Iteration {i}")
        print(f"Loss: {loss:.6f}")
        print(f"Output:\n{outputLayer}\n")


# Finaler Test mit den trainierten Gewichten
_, _, _, outputLayer = forward()

print("\nFinal Output (class labels):")
print((outputLayer >= 0.5).astype(int))    # Schwellwert 0.5 → 0 oder 1 (XOR-Ergebnis)