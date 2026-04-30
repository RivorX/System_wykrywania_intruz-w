# System wykrywania intruzów

Desktopowa aplikacja do monitoringu wielu kamer i plików wideo, oparta o modele YOLO. System wykrywa osoby, rozróżnia tryb dzienny i nocny, oznacza potencjalnych intruzów, zapisuje klipy zdarzeń oraz pozwala konfigurować źródła obrazu, maski ignorowanych obszarów i parametry modelu bez ręcznego grzebania w kodzie.

Projekt składa się z aplikacji inferencyjnej PyQt6 oraz skryptów do przygotowania danych i treningu modelu.

![Podgląd wielu kamer z detekcją intruzów](docs/działanie-kamer-1.png)

## Najważniejsze możliwości

- Obsługa wielu źródeł obrazu: kamery lokalne, pliki wideo i strumienie.
- Skanowanie dostępnych kamer z aplikacji albo przez parametr `--scan-cameras`.
- Detekcja osób modelem YOLO oraz śledzenie obiektów między klatkami.
- Tryb dzień/noc z osobnymi regułami alarmu.
- W trybie dziennym możliwość odróżniania pracownika od intruza na podstawie zaprogramowanego wzorca ubioru.
- Tryb dzienny oparty o segmentację sylwetki: aplikacja wycina maskę osoby, pobiera z niej kolory ubioru i na tej podstawie rozpoznaje pracowników.
- Tryb nocny oparty o standardową detekcję YOLO, gdzie wykryta osoba może od razu spełniać warunek alarmu.
- Obsługa masek, czyli obszarów kadru ignorowanych przez AI.
- Podgląd live w siatce kamer, zoom/pan obrazu oraz tryb pełnoekranowy dla pojedynczego źródła.
- Wbudowany odtwarzacz nagrań z suwakiem czasu, pauzą, stopem, zoomem i przesuwaniem obrazu.
- Archiwizacja wykryć do klipów wideo z pre-bufferem, czasem widoczności i cooldownem.
- Automatyczne ograniczanie liczby zapisanych zdarzeń i czyszczenie najstarszych plików po przekroczeniu limitu.
- Zakładka wykrytego ruchu z filtrowaniem zdarzeń, podglądem i otwieraniem zapisanych plików.
- Panel ustawień do zmiany progów detekcji, modeli, FPS, zapisu zdarzeń i kolorów ubioru.
- Zapisywanie i wczytywanie presetów ustawień z poziomu interfejsu.
- Pobieranie nowych i starszych modeli YOLO bez znajomości kodu: użytkownik wybiera model w aplikacji, klika `Pobierz wybrany model` i może go od razu załadować.
- Logi aplikacji z możliwością eksportu.
- Skrypty do pobrania/przygotowania datasetu COCO `person` i treningu YOLO.

## Jak działa aplikacja

Aplikacja pracuje w osobnym oknie PyQt6. Podgląd kamer działa niezależnie od inferencji AI: UI może odświeżać obraz płynnie, a model analizuje najnowsze dostępne klatki zgodnie z limitem FPS. Dzięki temu stare klatki nie są kolejkowane, a opóźnienie podglądu pozostaje niskie.

W trybie nocnym działa klasyczny model YOLO do detekcji osób: jeśli ktoś pojawi się w kadrze, system może potraktować go jako intruza według progów alarmu. W dzień aplikacja przełącza się na wariant segmentacyjny. Model wycina maskę sylwetki, a system pobiera z tej maski kolor górnej i dolnej części ubioru. Jeżeli ubiór pasuje do zaprogramowanego wzorca, osoba jest oznaczana jako `pracownik`; w przeciwnym razie jako `intruz`.

![Detekcja po zmianie ustawień](docs/po-zmianie-wykladu-kamera.png)

## Ekrany aplikacji

### Podgląd kamer

Zakładka podglądu pokazuje wszystkie aktywne źródła w siatce. Na obrazie widoczne są ramki detekcji, etykiety `pracownik`/`intruz`, liczba osób, liczba intruzów, tryb pracy oraz aktualne metryki FPS dla każdej kamery. `src` oznacza liczbę klatek dostarczaną przez dane źródło, `ai` pokazuje tempo inferencji modelu, a `view` to płynność podglądu po wygładzeniu wyników z AI i trackerów.

![Podgląd kamer](docs/działanie-kamer-2.png)

Na kafelkach można powiększać obraz kółkiem myszy i przesuwać powiększony kadr przeciąganiem. Pojedyncze źródło można otworzyć w trybie pełnoekranowym, a wyjście z tego widoku działa przez `Esc`, `F11`, prawy klik albo przycisk zamknięcia.

### Nagrania

W zakładce `Podgląd kamer` znajduje się też widok `Nagrania`. Pozwala wczytać plik wideo, odtwarzać go, pauzować, wracać do początku, przewijać suwakiem oraz korzystać z zoomu i przesuwania kadru. Dzięki temu aplikacja służy nie tylko do live monitoringu, ale też do szybkiego sprawdzania materiału z pliku.

### Konfiguracja kamer

W zakładce konfiguracji można dodawać źródła typu kamera, plik wideo albo stream, włączać i wyłączać kamery, ustawiać start od losowego momentu dla wideo oraz zapisywać konfigurację. Źródła są przechowywane poza głównym plikiem `inference.yaml`, w katalogu ustawień aplikacji.

![Konfiguracja kamer](docs/zakladka-konfiguracja-kamer.png)

### Maski ignorowanych obszarów

Dla źródła można ustawić maskę obszaru, którego model nie powinien analizować. To przydatne, gdy fragment kadru stale powoduje fałszywe alarmy albo nie powinien brać udziału w detekcji.

![Kamera z maską ignorowanego obszaru](docs/kamera-z-maska.png)

### Wykryty ruch

Zakładka `Wykryty ruch` prezentuje zapisane zdarzenia. Dostępne są filtry po okresie, trybie, kamerze, konkretnym dniu i godzinach. Po wybraniu wpisu aplikacja pokazuje podgląd zapisanego klipu lub obrazu.

![Lista zapisanych zdarzeń](docs/zakladka-wykryty-ruch.png)

### Ustawienia

Panel ustawień pozwala zmienić reguły alarmu, archiwizację zdarzeń, profil detekcji YOLO, modele dzienne/nocne, urządzenie GPU/CPU, FP16, kompilację Torch oraz wzorzec ubioru dziennego.

![Ustawienia aplikacji](docs/zakladka-ustawienia.png)

### Logi

Zakładka logów pokazuje komunikaty runtime: ładowanie modeli, start/stop źródeł, zapis zdarzeń, zmiany ustawień i ostrzeżenia. Logi można wyczyścić albo wyeksportować do pliku.

![Logi aplikacji](docs/zakladka-logi.png)

## Modele i profile pracy

Konfiguracja inferencji znajduje się w `config/inference.yaml`. Aplikacja obsługuje profile YOLO:

- `low`: szybki profil dla słabszego sprzętu lub wielu kamer.
- `medium`: balans między dokładnością i płynnością.
- `high`: wyższa dokładność kosztem FPS.
- `custom`: ręcznie dobrany model, rozdzielczość, progi i limity detekcji.

Modele można zmieniać bez edycji plików i bez znajomości programowania. W zakładce `Ustawienia` znajduje się zaawansowany wybór modelu: aplikacja pokazuje katalog dostępnych wariantów, rozmiar pliku, źródło modelu i informację, czy model jest już pobrany. Można pobrać nowsze lub starsze modele YOLO jednym kliknięciem (`Pobierz wybrany model`), a potem od razu załadować je do pracy. Dzięki temu można szybko porównać lekki model nano, domyślny profil medium albo cięższy wariant nastawiony na dokładność.

Ustawienia można zapisywać jako presety i później je wczytywać. To ułatwia szybkie przełączanie między konfiguracją demonstracyjną, oszczędną i dokładną bez ręcznego przepisywania wartości.

Aktualna konfiguracja używa między innymi:

- modelu nocnego `yolo26s.pt`,
- modelu dziennego segmentacji `yolo26s-seg.pt`,
- klasy `person`,
- GPU `device: '0'`,
- FP16,
- BoT-SORT jako trackera,
- automatycznego trybu dzień/noc w godzinach 22:00-06:00,
- zapisu zdarzeń do `logs/app/events`.

Domyślnym kompromisem dla aplikacji jest profil `medium`, oparty o model z rodziny `yolo26s`. Model `nano` był testowany jako lżejsza alternatywa, żeby pokazać, jak mocno można ograniczyć użycie GPU przy zachowaniu działania systemu na wielu źródłach.

![Alternatywny profil nano zamiast domyślnego medium](docs/nano-model-ustawienia.png)

Na lżejszym modelu nano aplikacja nadal wykrywa osoby i obsługuje wiele kamer, ale robi to z mniejszym kosztem obliczeniowym niż ustawienia medium/high.

![Podgląd kamer na modelu nano](docs/nano-model-kamery.png)

## Wydajność

Wydajność zależy od modelu, rozdzielczości `imgsz`, liczby źródeł i GPU. Domyślny profil `medium` jest balansem między jakością i płynnością, natomiast `nano` służy jako wariant oszczędny: znacząco zmniejsza użycie GPU, kosztem mniejszego modelu i potencjalnie słabszej detekcji w trudniejszych kadrach.

![Niższe użycie CPU/GPU po przełączeniu na model nano](docs/nano-model-cpuGpu-usage.png)

Przy wyższych ustawieniach detekcji rośnie obciążenie GPU/CPU, szczególnie przy większej rozdzielczości i wielu źródłach.

![Użycie CPU/GPU przy wysokich ustawieniach](docs/użycie-cpuGpu-na-wysokich-ustawieniach-wykrywania.png)

## Instalacja

Projekt był przygotowany pod Windows i CUDA. Wymagania są w `requirements.txt`.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Szybka weryfikacja CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Jeżeli `torch.cuda.is_available()` zwraca `False`, środowisko używa CPU albo zainstalowana wersja PyTorch nie pasuje do sterownika/CUDA.

## Uruchomienie aplikacji

```bash
python scripts/inference_app.py
```

Skanowanie dostępnych kamer:

```bash
python scripts/inference_app.py --scan-cameras
```

Po uruchomieniu aplikacja może automatycznie wystartować w pełnym ekranie, zależnie od ustawień:

```yaml
runtime:
  auto_start_live: true
  start_fullscreen: true
  start_maximized: true
```

## Zapis zdarzeń

Zdarzenie jest zapisywane, gdy osoba lub intruz jest widoczny dłużej niż ustawiony próg. System zapisuje klip wideo, może dodać ramki z opisem i używa pre-bufferu, aby klip zawierał moment sprzed właściwego alarmu.

Najważniejsze opcje:

```yaml
events:
  enabled: true
  min_visible_seconds: 4.0
  cooldown_seconds: 11.0
  linger_seconds: 5.0
  clip_fps: 30
  prebuffer_seconds: 2.0
  max_saved_events: 302
  save_annotated_frame: true
  output_dir: logs/app/events
```

Indeks zdarzeń jest zapisywany razem z plikami, dzięki czemu zakładka `Wykryty ruch` może je filtrować i odtwarzać.

Po przekroczeniu `max_saved_events` aplikacja usuwa najstarsze wpisy i odpowiadające im pliki, więc katalog zdarzeń nie rośnie bez końca. Zdarzenia można też wyczyścić ręcznie z poziomu zakładki `Wykryty ruch`.

## Trening modelu

Repozytorium zawiera pipeline do przygotowania datasetu i treningu YOLO na klasie `person`.

Konfiguracje:

- `config/dataset.yaml`: pobieranie i przygotowanie podzbioru COCO.
- `config/train.yaml`: model, hiperparametry, augmentacje, batch auto-search i eksport wag.
- `scripts/prepare_dataset.py`: przygotowanie danych.
- `scripts/training.py`: trening modelu.

Uruchomienie treningu:

```bash
python scripts/training.py
```

Po treningu skrypt zapisuje wagi w logach runa oraz eksportuje najlepsze modele do `models/weights/`.

## Struktura projektu

```text
config/
  dataset.yaml        konfiguracja przygotowania datasetu
  inference.yaml      konfiguracja aplikacji inferencyjnej
  train.yaml          konfiguracja treningu
docs/                 screeny aplikacji użyte w README
logs/                 logi treningu i działania aplikacji
models/               wagi bazowe i wytrenowane
scripts/
  inference_app.py    aplikacja desktopowa PyQt6
  prepare_dataset.py  przygotowanie danych
  training.py         trening YOLO
  utils/              moduły pomocnicze
```

## Najważniejsze zależności

- `ultralytics` - YOLO, tracking i modele detekcji/segmentacji.
- `torch`, `torchvision`, `torchaudio` - backend deep learning.
- `opencv-python` - odczyt kamer, strumieni i wideo.
- `PyQt6` - interfejs desktopowy.
- `lap` - zależność używana przez trackery.
- `numpy`, `pandas`, `matplotlib`, `pyyaml`, `tqdm` - narzędzia pomocnicze.

## Licencja

Projekt jest udostępniony na warunkach licencji z pliku `LICENSE`.
