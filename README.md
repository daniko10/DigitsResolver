# Rozpoznawacz cyfr

## 1. Opis projektu
Nasza aplikacja, która umożliwia rozpoznawanie cyfr ręcznie pisanych. Projekt oparty jest na wykorzystaniu sieci neuronowych, takich jak CNN (Convolutional Neural Network) oraz MLP (Multi-Layer Perceptron), które klasyfikują cyfry na podstawie ich obrazu. 

---

## 2. Główne funkcjonalności aplikacji:
### Rysowanie cyfr:
- Narysuj cyfrę na płótnie, a aplikacja automatycznie wyświetli przewidywanie.

### Wczytywanie obrazu:
- Kliknij **"Wczytaj obraz"** i wybierz plik PNG zawierający cyfrę.
- Obraz zostanie wyświetlony na płótnie, a aplikacja poda wynik klasyfikacji.

### Zapisywanie obrazu:
- Kliknij **"Zapisz"**, aby zapisać rysunek do pliku PNG.

### Wybór modelu:
- Wybierz model klasyfikacji: **CNN** lub **MLP**, korzystając z opcji **"Wybierz model"**.
  
---

## 3. Opis zastosowanej technologii

W projekcie wykorzystano dwa modele sieci neuronowych: **Convolutional Neural Network (CNN)** i **Multi-Layer Perceptron (MLP)**. 

- **Convolutional Neural Network (CNN)**:
  - Sieci konwolucyjne to rodzaj sieci neuronowych zaprojektowanych do analizy danych obrazowych. Wykorzystują **jądra konwolucyjne** (filtry), które przesuwają się po obrazie, wyodrębniając kluczowe cechy, takie jak krawędzie, tekstury czy wzory. 
  - Dzięki warstwom konwolucyjnym CNN są w stanie automatycznie uczyć się reprezentacji danych obrazowych na różnych poziomach szczegółowości.

- **Multi-Layer Perceptron (MLP)**:
  - Klasyczny rodzaj sieci neuronowej, w której każdy neuron w jednej warstwie łączy się z każdym neuronem w następnej (tzw. warstwy w pełni połączone). MLP pracuje na danych spłaszczonych (np. obraz 28x28 przekształcony na wektor 784 elementów), co ogranicza jego zdolność do wykrywania lokalnych zależności.


### Struktura sieci
#### Convolutional Neural Network (CNN):
- **Architektura**:
  - 3 warstwy konwolucyjne:
    - **Conv1**: 32 filtry o rozmiarze 3x3.
    - **Conv2**: 64 filtry o rozmiarze 3x3.
    - **Conv3**: 128 filtry o rozmiarze 3x3.
  - Warstwy konwolucyjne połączone z:
    - Warstwami pooling (maksymalne zmniejszenie wymiaru obrazu).
    - Dropout (prawdopodobieństwo 25%).
  - Warstwa w pełni połączona:
    - **FC1**: 512 neuronów.
  - Wyjście: 10 neuronów (dla klas cyfr 0-9).

#### Multi-Layer Perceptron (MLP):
- **Architektura**:
  - Warstwa wejściowa: 784 neurony (28x28 pikseli).
  - 5 warstw ukrytych:
    - **Hidden1**: 256 neuronów.
    - **Hidden2**: 256 neuronów.
    - **Hidden3**: 128 neuronów.
    - **Hidden4**: 64 neuronów.
    - **Hidden5**: 32 neuronów.
  - Dropout po każdej warstwie ukrytej (20%).
  - Wyjście: 10 neuronów (dla klas cyfr 0-9).

---

### Stan trenowania modeli
- Modele zostały wytrenowane i zwalidowane na zbiorach:
  - **MNIST Dataset** (cyfry od 0 do 9).
  - **EMNIST Dataset** (split "digits").
- Mechanizmy w procesie trenowania:
  - **Dropout**, aby zapobiec przeuczeniu - mamy okreslone prawdopodobieństwo, że dany neuron zostanie odrzucony.
  - **Batch Normalization**, aby stabilizować proces uczenia.
- Wyniki:
  - CNN: **98% dokładności** na zbiorze testowym.
  - MLP: **94.5% dokładności** na zbiorze testowym.
  - Żaden z modeli nie wykazuje oznak przeuczenia.

---

#### Porównanie:
- **CNN**:
  - Lepsze w analizie danych obrazowych dzięki zastosowaniu warstw konwolucyjnych, które wykrywają lokalne wzorce.
  - Wykorzystuje mechanizmy redukcji wymiarów, co poprawia efektywność działania.
  - Wysoka skuteczność w rozpoznawaniu obrazów z minimalnym przetwarzaniem wstępnym.
- **MLP**:
  - Klasyczny model, który działa na spłaszczonych danych wejściowych.
  - Prostota implementacji, ale niższa skuteczność w porównaniu do CNN dla danych obrazowych narysowanych przez użytkownika.

---

## 4. Dane

### Dane wykorzystane do nauki modelu:
Modele zostały wytrenowane na zbiorze danych **MNIST**:
- **Źródło:** [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
- **Liczba próbek:**
  - 60 000 obrazów treningowych.
  - 10 000 obrazów testowych.
- **Rozmiar obrazów:** 28x28 pikseli w skali szarości -> stosujemy binaryzacje dla określonego wcześniej progu.
- **Cyfry:** od 0 do 9.

---

## 5. Wymagania systemowe

### Technologie:
- **Python 3.11**

### Biblioteki:
- `torch`
- `torchvision`
- `Pillow`
- `tkinter`


## 6. Uruchomienie aplikacji

- użytkownik pobiera z Github zdalne repozytorium
```bash
git clone https://github.com/daniko10/DigitsResolver.git
```
- po pobraniu wchodzimy w folder DigitsResolver
- wykonujemy skrypt pythonowy, który udostępnia nam GUI
```bash
python3 digitsSolver.py
```

## 7. Podsumowanie
- CNN zgodnie z założeniami działa poprawnie nawet dla nowo narysowanych cyfr
- MLP nie jest w stanie dobrze określić predykcji nawet z dobrze wyuczonymi wagami, wynika to z jego ograniczeń strukturowych. MLP traktuje każde wejście jako niezależną cechę. Nie ma mechanizmu do uchwycenia relacji przestrzennych między sąsiadującymi pikselami, dlatego nie potrafi wykrywać struktur takich jak krawędzie, rogi czy tekstury, które są kluczowe w identyfikacji cyfry. 
