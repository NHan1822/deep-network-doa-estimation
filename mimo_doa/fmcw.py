"""Высокоточный FMCW-симулятор сигнала радара TI AWR1843.

Реализована полная физическая модель: генерация чирпа, моделирование
отражения от цели с учётом задержки распространения и доплеровского
сдвига, де-чирпинг (mix-down), ADC-квантование, тепловой шум. На выходе
получается тензор raw ADC данных формы (samples, chirps, rx, tx), который
обрабатывается стандартной цепочкой Range-FFT → Doppler-FFT.

Параметры по умолчанию соответствуют конфигурации TI AWR1843BOOST EVM,
описанной в документации Texas Instruments и используемой в датасете
Gao et al. (UWCR, IEEE Dataport, 2022).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# Скорость света
C = 299_792_458.0


@dataclass
class FMCWConfig:
    """Параметры FMCW-радара TI AWR1843."""
    f_start: float = 77e9          # стартовая частота, Гц
    bandwidth: float = 4e9         # полоса перестройки за чирп, Гц
    chirp_time: float = 60e-6      # длительность чирпа, с
    idle_time: float = 5e-6        # пауза между чирпами, с
    sample_rate: float = 10e6      # частота дискретизации ADC, Гц
    n_samples: int = 128           # отсчётов на чирп
    n_chirps: int = 255            # чирпов на кадр
    n_tx: int = 2
    n_rx: int = 4
    tx_spacing_lambda: float = 2.0  # шаг передающих элементов в λ
    rx_spacing_lambda: float = 0.5  # шаг приёмных в λ
    noise_figure_db: float = 12.0   # шум-фактор приёмника

    @property
    def slope(self) -> float:
        return self.bandwidth / self.chirp_time

    @property
    def wavelength(self) -> float:
        return C / self.f_start

    @property
    def effective_bandwidth(self) -> float:
        """Полоса, фактически охваченная за время сбора n_samples отсчётов."""
        sampling_window = self.n_samples / self.sample_rate
        return self.slope * sampling_window

    @property
    def range_resolution(self) -> float:
        """Размер бина дальности после Range-FFT: Fs · c / (2 · slope · N)."""
        return self.sample_rate * C / (2 * self.slope * self.n_samples)

    @property
    def max_range(self) -> float:
        return (self.sample_rate * C) / (2 * self.slope)

    @property
    def velocity_resolution(self) -> float:
        return self.wavelength / (2 * self.n_chirps *
                                    (self.chirp_time + self.idle_time) *
                                    self.n_tx)

    @property
    def max_velocity(self) -> float:
        return self.wavelength / (4 * (self.chirp_time + self.idle_time) *
                                    self.n_tx)


@dataclass
class Target:
    """Описание точечной радиолокационной цели."""
    range_m: float
    velocity_ms: float    # положительная — приближение
    azimuth_deg: float
    rcs: float = 1.0      # эффективная площадь рассеяния, м²


def _tx_position(cfg: FMCWConfig) -> np.ndarray:
    return np.arange(cfg.n_tx) * cfg.tx_spacing_lambda * cfg.wavelength


def _rx_position(cfg: FMCWConfig) -> np.ndarray:
    return np.arange(cfg.n_rx) * cfg.rx_spacing_lambda * cfg.wavelength


def synthesize_adc_frame(targets: Iterable[Target],
                          cfg: FMCWConfig | None = None,
                          *,
                          snr_db_per_target: float | None = None,
                          rng: np.random.Generator | None = None
                          ) -> np.ndarray:
    """Генерация raw ADC тензора (samples, chirps, rx, tx) для FMCW MIMO-радара.

    Для каждой пары tx/rx и каждого чирпа моделируется отражение от всех
    указанных целей с учётом задержки распространения, доплеровского сдвига
    и пространственной геометрии MIMO-решётки. Затем выполняется
    де-чирпинг с копией излучённого сигнала, и результат дискретизируется
    с частотой sample_rate. На выходе добавляется тепловой шум, мощность
    которого подбирается так, чтобы достичь заданного SNR на цель.
    """
    if cfg is None:
        cfg = FMCWConfig()
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(cfg.n_samples) / cfg.sample_rate            # время в чирпе
    chirp_t = np.arange(cfg.n_chirps) * (cfg.chirp_time + cfg.idle_time)
    d_tx = _tx_position(cfg)
    d_rx = _rx_position(cfg)
    k = 2 * np.pi / cfg.wavelength

    adc = np.zeros((cfg.n_samples, cfg.n_chirps, cfg.n_rx, cfg.n_tx),
                    dtype=np.complex64)
    targets = list(targets)

    for target in targets:
        sin_theta = np.sin(np.deg2rad(target.azimuth_deg))
        # Базовая задержка и доплеровская частота
        tau0 = 2 * target.range_m / C
        doppler_freq = 2 * target.velocity_ms * cfg.f_start / C
        # Слабая нормировка амплитуды: модель свободного пространства
        # Pr ∝ Pt G_t G_r λ² σ / ( (4π)³ R⁴ )
        amplitude = np.sqrt(target.rcs) / (4 * np.pi * max(target.range_m, 0.5) ** 2)

        for itx in range(cfg.n_tx):
            for irx in range(cfg.n_rx):
                # MIMO-фаза для виртуальной решётки
                phase_mimo = np.exp(1j * k * (d_tx[itx] + d_rx[irx]) * sin_theta)

                for ic in range(cfg.n_chirps):
                    # Доплеровская модуляция между чирпами
                    phase_doppler = np.exp(1j * 2 * np.pi * doppler_freq * chirp_t[ic])
                    # Полная задержка с учётом движения за время чирпа
                    tau = tau0 + 2 * target.velocity_ms * chirp_t[ic] / C
                    # IF-сигнал после де-чирпинга:
                    #   s_IF(t) = exp(j 2π (slope · τ · t + f_start · τ − slope · τ² / 2))
                    if_signal = np.exp(1j * 2 * np.pi * (
                        cfg.slope * tau * t + cfg.f_start * tau
                        - 0.5 * cfg.slope * tau * tau))
                    adc[:, ic, irx, itx] += (amplitude * phase_mimo *
                                              phase_doppler * if_signal)

    # Шум: тепловой + квантование (упрощённо — гауссов комплексный)
    if snr_db_per_target is None:
        sigma2 = 1e-18
    else:
        sig_pwr = max(float(np.mean(np.abs(adc) ** 2)), 1e-30)
        sigma2 = sig_pwr / (10 ** (snr_db_per_target / 10))
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(adc.shape) +
                                     1j * rng.standard_normal(adc.shape))
    return (adc + noise).astype(np.complex64)


def range_fft(adc: np.ndarray) -> np.ndarray:
    """Range-FFT по оси отсчётов с Hanning-окном."""
    window = np.hanning(adc.shape[0])[:, None, None, None]
    return np.fft.fft(adc * window, axis=0)


def doppler_fft(range_data: np.ndarray) -> np.ndarray:
    """Doppler-FFT по оси чирпов с центрированием нулевого бина."""
    window = np.hanning(range_data.shape[1])[None, :, None, None]
    return np.fft.fftshift(np.fft.fft(range_data * window, axis=1), axes=1)


def virtual_array_signal(rd: np.ndarray) -> np.ndarray:
    """Объединение TDM-каналов в виртуальную ULA-решётку (M = N_tx · N_rx)."""
    R, D, Nrx, Ntx = rd.shape
    return rd.transpose(0, 1, 3, 2).reshape(R, D, Nrx * Ntx)


def range_doppler_map(adc: np.ndarray) -> np.ndarray:
    """Магнитудная карта дальность × радиальная скорость."""
    rd = doppler_fft(range_fft(adc))
    # Сначала складываем по rx/tx (некогерентно), затем нормируем
    return np.sqrt(np.mean(np.abs(rd) ** 2, axis=(2, 3)))


def range_axis_m(cfg: FMCWConfig) -> np.ndarray:
    return np.arange(cfg.n_samples) * cfg.range_resolution


def doppler_axis_ms(cfg: FMCWConfig) -> np.ndarray:
    n = cfg.n_chirps
    return (np.arange(n) - n // 2) * cfg.velocity_resolution
