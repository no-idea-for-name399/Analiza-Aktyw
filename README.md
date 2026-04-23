# Analiza-Aktyw
Ilościowa analiza korelacji i ryzyka: Bitcoin, Ethereum, S&amp;P 500 oraz Złoto na danych historycznych z 3 lat. Python · yfinance · pandas · statsmodels · seaborn

# 📊 Analiza Porównawcza Aktywów: Tradycyjne vs. Kryptowaluty

Projekt akademicki z zakresu finansów ilościowych badający zależności między
czterema klasami aktywów: **Bitcoin (BTC)**, **Ethereum (ETH)**, **S&P 500**
i **Złotem (XAUUSD)** na danych historycznych z ostatnich 3 lat.

## Cel projektu
Weryfikacja hipotezy „cyfrowego złota" – czy Bitcoin faktycznie zachowuje się
jak aktywo zabezpieczające przed inflacją, czy raczej jak lewarowany instrument
klasy risk-on, silnie skorelowany z rynkiem akcji?

## Zakres analizy
- Pobieranie danych historycznych przez API `yfinance`
- Logarytmiczne stopy zwrotu i harmonizacja danych (forward-fill)
- Statystyki opisowe: średnia, mediana, odch. std., skośność, kurtoza
- Współczynnik Sharpe'a i Maximum Drawdown
- Analiza regresji OLS (model CAPM) → wyznaczenie współczynnika Beta
- Macierz korelacji Pearsona
- 5 profesjonalnych wizualizacji (Matplotlib/Seaborn)

## Technologie
`Python 3.11` · `yfinance` · `pandas` · `numpy` · `statsmodels` · `scipy` · `seaborn` · `matplotlib`

## Uruchomienie
```bash
pip install yfinance pandas numpy matplotlib seaborn scipy statsmodels
python analiza_aktywow.py
```
Lub otwórz bezpośrednio w **Google Colab** – brak konfiguracji środowiska.

## Kluczowy wniosek
Dane empiryczne wskazują, że Bitcoin wykazuje istotnie wyższą korelację
z S&P 500 niż ze Złotem (β > 1), co klasyfikuje go jako aktywo spekulacyjne,
a nie instrument hedgingowy w tradycyjnym sensie.

---
*Projekt przygotowany na potrzeby akademickie. Nie stanowi rekomendacji inwestycyjnej.
