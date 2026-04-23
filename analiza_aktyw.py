# =============================================================================
# ANALIZA PORÓWNAWCZA KORELACJI I RYZYKA: AKTYWA TRADYCYJNE vs. KRYPTOWALUTY
# Autor: [Maksymilian Czerw]
# Data: 2024/2025
# Środowisko: Google Colab / Python 3.10+
# =============================================================================

# ─────────────────────────────────────────────
# SEKCJA 0: INSTALACJA I IMPORT BIBLIOTEK
# ─────────────────────────────────────────────

# Instalacja potrzebnych bibliotek (odkomentuj w Colab)
# !pip install yfinance pandas numpy matplotlib seaborn scipy statsmodels

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Ustawienia globalne dla wykresów – profesjonalny wygląd
plt.rcParams.update({
    'figure.facecolor': '#0D1117',
    'axes.facecolor': '#161B22',
    'axes.edgecolor': '#30363D',
    'axes.labelcolor': '#C9D1D9',
    'xtick.color': '#8B949E',
    'ytick.color': '#8B949E',
    'text.color': '#C9D1D9',
    'grid.color': '#21262D',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'legend.facecolor': '#161B22',
    'legend.edgecolor': '#30363D',
})

# Paleta kolorów – przypisana do aktywów
COLORS = {
    'BTC-USD':  '#F7931A',   # Bitcoin – pomarańczowy
    'ETH-USD':  '#627EEA',   # Ethereum – niebieski
    '^GSPC':    '#00D4AA',   # S&P 500 – zielony
    'GC=F':     '#FFD700',   # Złoto – złoty
}

LABELS = {
    'BTC-USD': 'Bitcoin (BTC)',
    'ETH-USD': 'Ethereum (ETH)',
    '^GSPC':   'S&P 500',
    'GC=F':    'Złoto (Gold)',
}

print("✅ Biblioteki załadowane pomyślnie.")

# ─────────────────────────────────────────────
# SEKCJA 1: POBIERANIE DANYCH HISTORYCZNYCH
# ─────────────────────────────────────────────

# Definicja aktywów i okresu analizy (3 lata wstecz od dziś)
TICKERS   = ['BTC-USD', 'ETH-USD', '^GSPC', 'GC=F']
END_DATE  = pd.Timestamp.today().strftime('%Y-%m-%d')
START_DATE = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')

print(f"\n📥 Pobieranie danych: {START_DATE} → {END_DATE}")

# Pobieranie cen zamknięcia (Adjusted Close) dla wszystkich aktywów
raw_data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)
prices = raw_data['Close'].copy()
prices.columns.name = None  # usuń nazwę kolumny nadrzędnej

print(f"   Pobrano {len(prices)} wierszy danych dla {len(prices.columns)} aktywów.")
print(f"\n📊 Pierwsze 5 wierszy surowych danych:")
print(prices.head())
print(f"\n📊 Ostatnie 5 wierszy surowych danych:")
print(prices.tail())

# ─────────────────────────────────────────────
# SEKCJA 2: PRZETWARZANIE I CZYSZCZENIE DANYCH
# ─────────────────────────────────────────────

print("\n🔧 Przetwarzanie danych...")

# 2a. Obsługa brakujących danych
# Kryptowaluty handlowane są 7 dni w tygodniu, giełdy tradycyjne – 5 dni.
# Braki w cenach S&P 500 i Złota (weekendy, święta) uzupełniamy metodą
# forward-fill (przenosimy ostatnią dostępną cenę na następny dzień).
missing_before = prices.isnull().sum()
prices_filled = prices.ffill()  # forward fill
missing_after  = prices_filled.isnull().sum()

print(f"\n   Braki danych przed uzupełnieniem:\n{missing_before}")
print(f"\n   Braki danych po uzupełnieniu (ffill):\n{missing_after}")

# Usuwamy ewentualne wiersze, gdzie nadal brakuje danych (np. na początku szeregu)
prices_clean = prices_filled.dropna()
print(f"\n   Finalny zbiór danych: {len(prices_clean)} dni handlowych.")

# 2b. Obliczenie dziennych logarytmicznych stóp zwrotu
# Log-returny są addytywne w czasie i mają lepsze właściwości statystyczne
# (bliższe rozkładowi normalnemu) niż zwykłe stopy procentowe.
log_returns = np.log(prices_clean / prices_clean.shift(1)).dropna()

print(f"\n   Obliczono {len(log_returns)} dziennych logarytmicznych stóp zwrotu.")
print("\n📊 Przykładowe dzienne log-stopy zwrotu:")
print(log_returns.head())

# 2c. Resampling do skali tygodniowej i miesięcznej
# Używamy ostatniej ceny w danym okresie jako podstawy do obliczenia zwrotu.
prices_weekly   = prices_clean.resample('W').last()
prices_monthly  = prices_clean.resample('ME').last()  # ME = Month End

returns_weekly  = np.log(prices_weekly  / prices_weekly.shift(1)).dropna()
returns_monthly = np.log(prices_monthly / prices_monthly.shift(1)).dropna()

print(f"\n   Resampling tygodniowy: {len(returns_weekly)} obserwacji.")
print(f"   Resampling miesięczny: {len(returns_monthly)} obserwacji.")

# ─────────────────────────────────────────────
# SEKCJA 3: ANALIZA STATYSTYCZNA
# ─────────────────────────────────────────────

print("\n📈 Obliczanie statystyk opisowych...")

# 3a. Podstawowe statystyki opisowe (w ujęciu rocznym – annualizacja)
# Zakładamy 252 dni handlowych dla aktywów tradycyjnych,
# ale ponieważ używamy wspólnej osi czasu – 365 dni dla krypto.
# Dla uproszczenia przyjmujemy N_DAYS = 252 jako standard rynkowy.
N_DAYS = 252

stats_df = pd.DataFrame(index=TICKERS)
stats_df['Średnia dzienna (%)']   = log_returns.mean() * 100
stats_df['Mediana dzienna (%)']   = log_returns.median() * 100
stats_df['Odch. std. (ryzyko %)'] = log_returns.std() * 100
stats_df['Skośność']              = log_returns.skew()
stats_df['Kurtoza (excess)']      = log_returns.kurt()  # excess kurtosis (fat tails)

# Annualizowane statystyki
stats_df['Zwrot roczny (%)']      = log_returns.mean() * N_DAYS * 100
stats_df['Wol. roczna (%)']       = log_returns.std() * np.sqrt(N_DAYS) * 100
stats_df['Max Drawdown (%)']      = 0.0  # wypełnimy poniżej

# Obliczenie Maximum Drawdown dla każdego aktywa
for ticker in TICKERS:
    cumulative = (1 + log_returns[ticker]).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    stats_df.loc[ticker, 'Max Drawdown (%)'] = drawdown.min() * 100

# Zmiana nazw wierszy na czytelne etykiety
stats_df.index = [LABELS[t] for t in TICKERS]

print("\n" + "="*70)
print("TABELA 1: STATYSTYKI OPISOWE (DZIENNE LOG-STOPY ZWROTU)")
print("="*70)
print(stats_df.round(4).to_string())

# 3b. Współczynnik Sharpe'a (uproszczony)
# Sharpe = (Zwrot_roczny - Rf) / Volatility_roczna
# Przyjmujemy stopę wolną od ryzyka Rf ≈ 4.5% (US Fed Funds Rate ~2024)
RF_ANNUAL = 0.045  # 4.5% rocznie

sharpe_df = pd.DataFrame(index=TICKERS)
ann_return = log_returns.mean() * N_DAYS
ann_vol    = log_returns.std() * np.sqrt(N_DAYS)

sharpe_df['Zwrot roczny']       = ann_return
sharpe_df['Wolat. roczna']      = ann_vol
sharpe_df['Sharpe Ratio']       = (ann_return - RF_ANNUAL) / ann_vol
sharpe_df['Calmar Ratio']       = ann_return / abs(stats_df['Max Drawdown (%)'].values / 100)
sharpe_df.index = [LABELS[t] for t in TICKERS]

print("\n" + "="*70)
print(f"TABELA 2: WSKAŹNIKI EFEKTYWNOŚCI (Rf = {RF_ANNUAL*100:.1f}%)")
print("="*70)
print(sharpe_df.round(4).to_string())

# 3c. Analiza regresji liniowej: Bitcoin ~ S&P 500 (obliczenie Beta)
print("\n📉 Regresja liniowa: Bitcoin ~ S&P 500")

Y = log_returns['BTC-USD'].values   # zmienna objaśniana: Bitcoin
X = log_returns['^GSPC'].values     # zmienna objaśniająca: S&P 500

# Dodajemy stałą (intercept = alfa) do macierzy X
X_with_const = sm.add_constant(X)

# Fit modelu OLS (Ordinary Least Squares)
model = sm.OLS(Y, X_with_const).fit()

alpha_btc = model.params[0]   # Jensen's Alpha
beta_btc  = model.params[1]   # Beta (wrażliwość na rynek)
r_squared = model.rsquared    # R² – jakość dopasowania

print(f"\n   Alpha (Jensen's α): {alpha_btc:.6f}  ({alpha_btc*N_DAYS*100:.2f}% rocznie)")
print(f"   Beta (β):           {beta_btc:.4f}")
print(f"   R²:                 {r_squared:.4f}  ({r_squared*100:.1f}% wariancji wyjaśnione)")
print(f"   p-value (Beta):     {model.pvalues[1]:.6f}")
print("\n" + model.summary().as_text())

# 3d. Macierz korelacji Pearsona
print("\n🔗 Macierz korelacji Pearsona (dzienne log-stopy zwrotu)")
corr_matrix = log_returns.corr(method='pearson')
corr_matrix.columns = [LABELS[t] for t in TICKERS]
corr_matrix.index   = [LABELS[t] for t in TICKERS]
print(corr_matrix.round(4))

# ─────────────────────────────────────────────
# SEKCJA 4: WIZUALIZACJE
# ─────────────────────────────────────────────

print("\n🎨 Generowanie wykresów...")

# ── 4a. Wykres skumulowanych stóp zwrotu (Investment Growth $100) ──────────

fig, ax = plt.subplots(figsize=(14, 7))

# Obliczamy wzrost $100 zainwestowanego na początku okresu
cumulative_returns = (1 + log_returns).cumprod() * 100

for ticker in TICKERS:
    ax.plot(
        cumulative_returns.index,
        cumulative_returns[ticker],
        label=LABELS[ticker],
        color=COLORS[ticker],
        linewidth=2,
        alpha=0.9
    )

# Linia bazowa – $100 (kapitał początkowy)
ax.axhline(y=100, color='#8B949E', linestyle=':', linewidth=1, alpha=0.7, label='Kapitał bazowy ($100)')

ax.set_title('Skumulowany Wzrost Inwestycji ($100 zainwestowane)\n'
             'Bitcoin · Ethereum · S&P 500 · Złoto', pad=20)
ax.set_xlabel('Data')
ax.set_ylabel('Wartość portfela ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend(loc='upper left', framealpha=0.8)
ax.grid(True, alpha=0.4)
ax.tick_params(axis='x', rotation=30)

# Annotacje końcowych wartości
for ticker in TICKERS:
    final_val = cumulative_returns[ticker].iloc[-1]
    ax.annotate(
        f'${final_val:,.0f}',
        xy=(cumulative_returns.index[-1], final_val),
        xytext=(8, 0), textcoords='offset points',
        color=COLORS[ticker], fontsize=9, fontweight='bold', va='center'
    )

plt.tight_layout()
plt.savefig('01_cumulative_returns.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()
print("   ✅ Zapisano: 01_cumulative_returns.png")

# ── 4b. Heatmapa korelacji ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # ukryj górny trójkąt
cmap = sns.diverging_palette(10, 133, as_cmap=True)  # czerwony→zielony

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.3f',
    cmap=cmap,
    center=0,
    vmin=-1, vmax=1,
    linewidths=1,
    linecolor='#0D1117',
    square=True,
    cbar_kws={'shrink': 0.8, 'label': 'Współczynnik korelacji Pearsona'},
    ax=ax,
    annot_kws={'size': 12, 'weight': 'bold'}
)

ax.set_title('Heatmapa Korelacji Pearsona\nDzienne Log-Stopy Zwrotu (3 lata)', pad=20)
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('02_correlation_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()
print("   ✅ Zapisano: 02_correlation_heatmap.png")

# ── 4c. Box Ploty dziennych stóp zwrotu ───────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(16, 7))

for i, ticker in enumerate(TICKERS):
    data = log_returns[ticker] * 100  # konwersja na procenty
    ax   = axes[i]

    # Box plot z Seaborn
    sns.boxplot(
        y=data,
        ax=ax,
        color=COLORS[ticker],
        width=0.5,
        linewidth=1.5,
        flierprops=dict(
            marker='o', markerfacecolor=COLORS[ticker],
            markersize=3, alpha=0.4, linestyle='none'
        ),
        medianprops=dict(color='white', linewidth=2.5),
        boxprops=dict(alpha=0.8),
    )

    # Statystyki na wykresie
    q1, median, q3 = data.quantile([0.25, 0.5, 0.75])
    std = data.std()
    skew_val = data.skew()
    kurt_val = data.kurt()

    ax.set_title(LABELS[ticker], fontsize=12, fontweight='bold', color=COLORS[ticker])
    ax.set_ylabel('Dzienna Log-Stopa Zwrotu (%)' if i == 0 else '')
    ax.set_xlabel('')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}%'))

    # Tekst z kluczowymi statystykami
    info = (f"σ = {std:.2f}%\n"
            f"Skośność = {skew_val:.2f}\n"
            f"Kurtoza = {kurt_val:.2f}\n"
            f"Mediana = {median:.3f}%")
    ax.text(0.97, 0.97, info,
            transform=ax.transAxes, fontsize=9,
            va='top', ha='right', color='#C9D1D9',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262D',
                      edgecolor='#30363D', alpha=0.9))

fig.suptitle('Box Ploty Dziennych Log-Stóp Zwrotu\n'
             '(Widoczne "fat tails" i wartości odstające – "outliers")',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('03_box_plots.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()
print("   ✅ Zapisano: 03_box_plots.png")

# ── 4d. Scatter Plot z linią regresji: Bitcoin ~ S&P 500 ──────────────────

fig, ax = plt.subplots(figsize=(10, 7))

btc_ret  = log_returns['BTC-USD'] * 100
gspc_ret = log_returns['^GSPC'] * 100

# Scatter z przezroczystością (alpha) dla gęstości punktów
ax.scatter(gspc_ret, btc_ret,
           color=COLORS['BTC-USD'], alpha=0.25, s=12,
           label='Dzienne obserwacje', zorder=2)

# Linia regresji
x_range = np.linspace(gspc_ret.min(), gspc_ret.max(), 200)
y_fitted = (alpha_btc + beta_btc * x_range / 100) * 100  # skalowanie
ax.plot(x_range, y_fitted,
        color='#FF4444', linewidth=2.5, zorder=3,
        label=f'Regresja OLS: β={beta_btc:.3f}, α={alpha_btc*N_DAYS*100:.2f}%/rok, R²={r_squared:.3f}')

# Linia zerowa
ax.axhline(0, color='#8B949E', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(0, color='#8B949E', linestyle='--', alpha=0.5, linewidth=1)

ax.set_title('Analiza Regresji Liniowej\nBitcoin vs. S&P 500 (dzienne log-stopy zwrotu)', pad=15)
ax.set_xlabel('S&P 500 – Dzienna Log-Stopa Zwrotu (%)')
ax.set_ylabel('Bitcoin – Dzienna Log-Stopa Zwrotu (%)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}%'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}%'))
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.4)

# Annotacja z interpretacją Bety
beta_note = ("Beta > 1 → Bitcoin amplifikuje\n"
             "ruchy rynkowe S&P 500\n"
             "(wysoka korelacja z risk-on)")
ax.text(0.97, 0.05, beta_note,
        transform=ax.transAxes, fontsize=9,
        va='bottom', ha='right', color='#FFD700',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262D',
                  edgecolor='#F7931A', alpha=0.9))

plt.tight_layout()
plt.savefig('04_regression_btc_sp500.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()
print("   ✅ Zapisano: 04_regression_btc_sp500.png")

# ── 4e. Wykres porównawczy ryzyka i zwrotu (Risk-Return Map) ──────────────

fig, ax = plt.subplots(figsize=(10, 7))

for ticker in TICKERS:
    x = ann_vol[ticker] * 100   # zmienność roczna (ryzyko)
    y = ann_return[ticker] * 100  # zwrot roczny
    sharpe = sharpe_df.loc[LABELS[ticker], 'Sharpe Ratio']

    ax.scatter(x, y, s=300, color=COLORS[ticker], zorder=5,
               edgecolors='white', linewidths=1.5)
    ax.annotate(
        f"{LABELS[ticker]}\n(Sharpe: {sharpe:.2f})",
        xy=(x, y), xytext=(10, 8), textcoords='offset points',
        color=COLORS[ticker], fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262D',
                  edgecolor=COLORS[ticker], alpha=0.9)
    )

ax.axhline(RF_ANNUAL * 100, color='#8B949E', linestyle=':', alpha=0.7,
           label=f'Stopa wolna od ryzyka (Rf = {RF_ANNUAL*100:.1f}%)')
ax.set_title('Mapa Ryzyko–Zwrot (Risk-Return Map)\nRoczne wskaźniki dla każdego aktywa', pad=15)
ax.set_xlabel('Zmienność Roczna (σ, ryzyko) [%]')
ax.set_ylabel('Oczekiwany Zwrot Roczny [%]')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.legend(framealpha=0.8)
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('05_risk_return_map.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()
print("   ✅ Zapisano: 05_risk_return_map.png")

# ─────────────────────────────────────────────
# SEKCJA 5: PODSUMOWANIE – WYNIKI KOŃCOWE
# ─────────────────────────────────────────────

print("\n" + "="*70)
print("PODSUMOWANIE ANALIZY – KLUCZOWE WNIOSKI")
print("="*70)

# Ranking wg Sharpe Ratio
print("\n📊 RANKING AKTYWÓW WG WSPÓŁCZYNNIKA SHARPE'A (im wyższy, tym lepszy):")
sharpe_sorted = sharpe_df['Sharpe Ratio'].sort_values(ascending=False)
for i, (name, val) in enumerate(sharpe_sorted.items(), 1):
    emoji = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else "4️⃣"
    print(f"   {emoji} {name:<20} Sharpe = {val:.3f}")

# Wnioski dot. korelacji
btc_sp500_corr = corr_matrix.loc['Bitcoin (BTC)', 'S&P 500']
btc_gold_corr  = corr_matrix.loc['Bitcoin (BTC)', 'Złoto (Gold)']
gold_sp500_corr= corr_matrix.loc['Złoto (Gold)', 'S&P 500']

print(f"\n🔗 KLUCZOWE KORELACJE (Pearson):")
print(f"   Bitcoin ↔ S&P 500 : {btc_sp500_corr:.3f}  {'⚠️ wysoka (risk-on)' if btc_sp500_corr > 0.4 else '✅ niska'}")
print(f"   Bitcoin ↔ Złoto   : {btc_gold_corr:.3f}   {'✅ niska (hedge)' if abs(btc_gold_corr) < 0.3 else '⚠️ umiarkowana'}")
print(f"   Złoto   ↔ S&P 500 : {gold_sp500_corr:.3f}   {'✅ niska (hedge)' if abs(gold_sp500_corr) < 0.3 else '⚠️ umiarkowana'}")

print(f"\n📉 ANALIZA BETA (Bitcoin względem S&P 500):")
print(f"   Beta (β) = {beta_btc:.3f}")
if beta_btc > 1.5:
    print(f"   ➡️  Bitcoin jest BARDZO AGRESYWNYM aktywem (β >> 1).")
    print(f"      Reaguje {beta_btc:.1f}x mocniej na ruchy S&P 500.")
elif beta_btc > 0.8:
    print(f"   ➡️  Bitcoin ma UMIARKOWANĄ korelację z rynkiem (β ≈ 1).")
else:
    print(f"   ➡️  Bitcoin wykazuje NISKĄ korelację z rynkiem (β < 1).")

print(f"\n💡 WNIOSEK KOŃCOWY:")
if btc_sp500_corr > 0.4 and abs(btc_gold_corr) < btc_sp500_corr:
    print(f"""
   Dane z analizowanego okresu wskazują, że Bitcoin BARDZIEJ zachowuje się
   jak aktywo 'risk-on' (silna korelacja z S&P 500 = {btc_sp500_corr:.2f}) niż
   'cyfrowe złoto' (słabsza korelacja ze Złotem = {btc_gold_corr:.2f}).

   W praktyce oznacza to:
   ▸ W czasie paniki rynkowej BTC spada równolegle z akcjami
   ▸ Złoto wykazuje niższą korelację z akcjami → lepszy hedge
   ▸ Bitcoin NIE zastępuje w pełni roli Złota jako 'bezpiecznej przystani'
   ▸ Wysokie Sharpe Ratio BTC/ETH w okresach hossy – ale kosztem
     ekstremalnego ryzyka (duże odch. std., fat tails, max drawdown)
""")

print("="*70)
print("✅ ANALIZA ZAKOŃCZONA. Wszystkie wykresy zostały zapisane.")
print("="*70)
