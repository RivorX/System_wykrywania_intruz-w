# System wykrywania intruzow

Projekt wykrywania intruzow na podstawie obrazu z kamer lub plikow wideo, oparty o YOLO.

## Co jest gotowe
- trening modelu przez `scripts/training.py`,
- modularne utils w `scripts/utils/`,
- automatyczne przygotowanie datasetu i split `train/val/test`,
- automatyczne pobieranie wag modelu, jesli ich brakuje,
- aplikacja inferencyjna multi-source (`kamera`, `wideo`, `stream`) w `scripts/inference_app.py`,
- tracking obiektow (ByteTrack) per zrodlo w live podgladzie,
- automatyczne zapisywanie zdarzen (snapshot), gdy osoba jest widoczna przez zadany czas,
- logi treningu w `logs/train/<run_name>/` (w tym `results.csv`, `results.png` z Ultralytics).

## Struktura
- `config/` - konfiguracje treningu, datasetu i inferencji,
- `scripts/training.py` - entrypoint treningu,
- `scripts/prepare_dataset.py` - osobne przygotowanie datasetu,
- `scripts/inference_app.py` - aplikacja inferencji (multi-source),
- `scripts/utils/` - moduly pomocnicze,
- `logs/train/` - artefakty treningowe,
- `models/base/` - lokalne bazowe wagi modeli.

## Instalacja (Windows + CUDA)
Jezeli po instalacji masz `torch ... +cpu`, to znaczy, ze pip zainstalowal wariant CPU.

1. Utworz i aktywuj srodowisko:
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Zainstaluj zaleznosci (w tym `torch` z CUDA `cu128` z `requirements.txt`):
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Szybka weryfikacja:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Trening
Domyslny trening jest ustawiony na przygotowanie podzbioru COCO (tylko `person`) przez `config/dataset.yaml`, model `yolo26m.pt` i `compile: true`.
Przy `compile: true` skrypt automatycznie ustawia ASCII-owe sciezki cache (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TEMP`) zeby uniknac bledow Unicode na Windows.
Batch size jest ustawiany automatycznie przez smart search:
- szybkie probe-y syntetyczne na GPU (forward+backward z `compile`),
- szybkie zawiezanie zakresu i wyszukiwanie binarne (nie liniowe brute force),
- margines bezpieczenstwa przez `safety_factor` w `training.auto_batch`.
- limit tylko po dedykowanym VRAM GPU (bez shared RAM) przez `training.auto_batch.max_vram_utilization`.
- metryka limitu VRAM: `training.auto_batch.max_vram_metric: allocated` (domyslnie).
- reuse jednego modelu/probe-session miedzy kolejnymi probe (`training.auto_batch.reuse_synthetic_state: true`).
- `compile_dynamic` jest opcjonalne; domyslnie `false` (stabilniej na Windows bez `cl.exe`).
- dla `compile=true` wynik jest cache'owany i kolejne uruchomienia biora batch z cache (bez ponownego probe).

Jesli chcesz stary, wolniejszy tryb probe przez mini-trening:
- `training.auto_batch.probe_method: mini_train`
- ustawienia cache (`compile=true`): `cache_enabled`, `cache_path`, `cache_max_entries`, `cache_max_age_hours`

Limit liczby obrazow ustawiasz w `config/dataset.yaml`:
- `prepare.limit_mode: cap` + `prepare.max_images: 1000` (albo `10000`),
- `prepare.limit_mode: max` zdejmuje limit.
- zmiana tych ustawien automatycznie przebuduje przygotowany dataset.

Pobieranie datasetu ma domyslnie pasek ogolny:
- `download.show_overall_progress: true` (pokazuje postep calego subsetu),
- `download.show_file_progress: false` (nie pokazuje paska dla kazdego pliku; ustaw `true`, jesli chcesz szczegolowe).

Pobieranie obrazow COCO moze dzialac rownolegle:
- `download.async_download: true`,
- `download.max_download_workers: 12` (zwieksz/zmniejsz zalezenie od lacza),
- `download.retry_attempts: 2`.

```bash
python scripts/training.py --config config/train.yaml
```

W razie potrzeby zmien katalog cache w `config/train.yaml`:
- `training.fix_unicode_cache_paths: true`
- `training.cache_root_dir: D:/torch_cache`

## Inferencja (kamera / wideo / wiele zrodel)
`config/inference.yaml` ma domyslnie `device: 0`, `half: true`, `compile: true`, model `yolo26m.pt`.
Po treningu skrypt:
- zapisuje standardowe `best.pt` i `last.pt` w logach runa (`logs/train/<run>/weights/`),
- eksportuje `best.pt` i `last.pt` do `models/weights/latest/`,
- eksportuje kanoniczny model "najlepszy per architektura" do `models/weights/<model.name>` (np. `models/weights/yolo26n.pt`), ale tylko gdy nowy wynik jest lepszy od juz zapisanego.

To oznacza:
- `yolo26n.pt` w `models/base/` = bazowe (pretrained) wagi startowe,
- `best.pt` / `last.pt` w `models/weights/latest/` = ostatni trening,
- `models/weights/yolo26n.pt` = najlepszy utrwalony wynik dla tej architektury (nie jest nadpisywany slabszym runem).

Domyslny wybor modelu w inferencji:
1. `model.selected_model_path` (jesli ustawiony i plik istnieje),
2. `models/weights/<model.name>` gdy `model.trained_weights.prefer_canonical: true`,
3. `models/weights/latest/best.pt` (lub `last.pt`, zalezne od `preferred`),
4. fallback do bazowego `models/base/<model.name>`.

```bash
python scripts/inference_app.py --config config/inference.yaml
```

To uruchamia aplikacje desktopowa (PyQt6) z:
- tabela zrodel (`camera`/`video`/`stream`),
- przyciskami dodawania/usuwania/toggle source,
- przyciskiem `Start/Stop` inferencji,
- podgladem na zywo w oknie aplikacji,
- zakladka `Wykryty ruch` z lista zapisanych zdarzen, podgladem i opcja "Wyczysc wszystkie zapisane zdarzenia".

Ustawienia archiwizacji zdarzen (zakladka `Ustawienia` -> `Archiwizacja wykryc`):
- `enabled` - wlacza zapisywanie zdarzen,
- `min_visible_seconds` - minimalny czas ciaglej widocznosci osoby,
- `cooldown_seconds` - przerwa miedzy kolejnymi zapisami,
- `min_person_count` - minimalna liczba osob, aby zapisac zdarzenie,
- `max_saved_events` - limit liczby plikow (0 = bez limitu),
- `save_annotated_frame` - zapis ramki z boxami i opisem,
- `once_per_streak` - jeden zapis na ciagla sekwencje wykrycia,
- `output_dir` - folder zapisu snapshotow (domyslnie `logs/inference/events`).

Wydajnosc live mozesz kontrolowac w `config/inference.yaml`:
- `runtime.frame_interval_ms` - jak czesto odswiezany jest podglad live,
- `runtime.view_target_fps` - docelowy FPS podgladu (aplikacja ogranicza do max 60),
- `runtime.view_cap_to_source_fps` - automatyczny cap podgladu do FPS zrodla (domyslnie `true`, mniejsze zuzycie CPU),
- `runtime.video_fps_fallback` - fallback FPS dla plikow video, gdy odczyt FPS z pliku jest niedostepny,
- `runtime.model_target_fps` - docelowa czestotliwosc inferencji AI,
- `runtime.max_infer_per_tick` - ile zrodel max moze byc inferowanych w jednym ticku (round-robin, batch inferencji).
- `runtime.capture_buffer_size` - bufor klatek dla live kamer/streamow (1 = najmniejsza latencja),
- `runtime.camera_width`, `runtime.camera_height`, `runtime.camera_fps` - prosba o parametry kamery (nizsza rozdzielczosc zwykle = wyzszy, stabilniejszy FPS view).
- `runtime.live_tile_spacing` - odstep miedzy kaflami kamer w siatce live.

Overlay metryk live:
- `src` - FPS zrodla (kamera/plik),
- `view` - FPS odswiezania podgladu w UI,
- `ai` - FPS inferencji modelu.

Live pipeline jest asynchroniczny:
- UI (podglad) odswieza sie niezaleznie od AI,
- model działa w osobnym workerze i bierze najnowsze klatki per zrodlo (bez kolejkowania starych),
- dzieki temu `view` moze byc wysokie nawet gdy `ai` jest nizsze.
- kazde zrodlo live ma osobny reader-thread dla `capture.read()`, wiec opoznienia pojedynczej kamery nie blokuja calej aplikacji,
- dla wideo live utrzymywane jest tempo real-time (drop starych klatek przy obciazeniu zamiast spowalniania odtwarzania).

ByteTrack (stabilniejsze boxy i ID osoby):
- `tracker.enabled: true` wlacza tracker,
- `tracker.frame_rate` (domyslnie 30),
- `tracker.track_high_thresh`, `tracker.track_low_thresh`, `tracker.new_track_thresh`,
- `tracker.track_buffer`, `tracker.match_thresh`, `tracker.fuse_score`.

Przydatne opcje:
```bash
python scripts/inference_app.py --scan-cameras
python scripts/inference_app.py --add-camera 0 --add-video data/videos/demo.mp4
python scripts/inference_app.py --auto-start
python scripts/inference_app.py --persist-sources
```

## Uwagi o modelu
Jesli `yolo26m.pt` nie jest jeszcze dostepny przez auto-download Ultralytics, ustaw:
- `model.download_url` w `config/train.yaml`, albo
- lokalny plik `.pt` w `models/base/`.
