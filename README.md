# MIMO DOA

Программная реализация двухэтапного нейросетевого конвейера для оценки направления
прихода сигнала (DOA) в автомобильных MIMO-радарах.

## Структура

```
code/
├── mimo_doa/
│   ├── __init__.py
│   ├── signal_model.py     — модель сигнала виртуальной MIMO-решётки
│   ├── classic_doa.py      — MUSIC, ESPRIT, forward/backward smoothing
│   ├── models.py           — нейросети: классификатор + реконструктор
│   └── dataset.py          — синтетический датасет (single/double/multipath)
├── train.py                — обучение моделей
├── figures.py              — построение всех графиков для ВКР
└── README.md
```

## Запуск

```bash
cd code
pip install numpy scipy torch matplotlib  # если ещё не установлены
python3 train.py --epochs 12 --train-len 6000 --val-len 1500 --out-dir checkpoints
python3 figures.py --checkpoints checkpoints --out-dir ../figures
```

Геометрия по умолчанию: 2 передающих и
4 приёмных антенны, виртуальная решётка из 8 элементов.

## Реализация

* MUSIC и ESPRIT с forward/backward сглаживанием как эталонные методы.
* Свёрточный классификатор сценария распространения сигнала (3 класса).
* Свёрточная нейросеть, реконструирующая теоритическую корреляционную матрицу из
  выборочной — далее на восстановленной R̂ применяется тот же MUSIC.

## Графики

| Файл | Содержание |
|------|-----------|
| fig01_array_geometry | Физическая и виртуальная антенные решётки TI AWR1843 |
| fig02_geometry_direct_vs_multipath | Геометрия прямого луча vs многолучёвого распространения |
| fig03_classifier_architecture | Архитектура нейросети-классификатора |
| fig04_reconstructor_architecture | Архитектура реконструктора корреляционной матрицы |
| fig05_pipeline_overview | Сквозной конвейер обработки данных |
| fig06_training_curves | Кривые обучения (loss, accuracy, MSE) |
| fig07_confusion_matrix | Матрица ошибок классификатора сценариев |
| fig08_pseudospectrum | Пространственный псевдоспектр (1 цель / 2 цели), MUSIC vs реконструкция |
| fig09_rmse_vs_snr | RMSE оценки DOA от SNR — MUSIC, ESPRIT, нейросетевой подход |
| fig10_resolution_probability | Вероятность правильного разрешения двух близких целей |
| fig11_range_doppler_map | Карта дальность/радиальная скорость и геометрия сцены |
| fig12_cartesian_results | Оценка декартовых координат (MUSIC vs реконструкция R̂) |
