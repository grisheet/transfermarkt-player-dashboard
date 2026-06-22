# ⚽ Transfermarkt Player Performance Dashboard

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-success)](https://grisheet.github.io/transfermarkt-player-dashboard/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced football analytics platform that scrapes Transfermarkt data, performs machine learning clustering analysis, and generates interactive visualizations to discover player archetypes and performance patterns.

## 🌟 Features

- **Automated Data Collection**: Web scraping from Transfermarkt covering Premier League, La Liga, Bundesliga, Serie A, Ligue 1, and world's most valuable players
- **Machine Learning Clustering**: KMeans algorithm groups players by style proxies (position, age, market value, league strength)
- **PCA Visualization**: Dimensionality reduction for 2D visualization of complex player profiles
- **Interactive Dashboards**: Beautiful Plotly and Seaborn charts with hover-over details
- **Comprehensive Analytics**: CSV datasets, PNG visualizations, and interactive HTML dashboards

## 📊 Live Demo

🚀 **[View Live Dashboard](https://grisheet.github.io/transfermarkt-player-dashboard/)**

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - KMeans clustering and PCA
- **Plotly** - Interactive visualizations
- **Seaborn & Matplotlib** - Statistical plotting
- **Requests & BeautifulSoup** - Web scraping
- **NumPy** - Numerical computing

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/grisheet/transfermarkt-player-dashboard.git
cd transfermarkt-player-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

Run the dashboard script:
```bash
python dashboard.py
```

This will:
1. Scrape player data from Transfermarkt
2. Clean and engineer features
3. Perform KMeans clustering analysis
4. Generate visualizations
5. Create an interactive HTML dashboard

### Output Files

- `transfermarkt_players.csv` - Raw player data
- `transfermarkt_players_clustered.csv` - Data with cluster assignments
- `cluster_scatter_seaborn.png` - Static cluster visualization
- `player_dashboard.html` - Interactive dashboard (open in browser)

## 📈 Dashboard Statistics

- **150+ Players** analyzed from top European leagues
- **6 League Sources** (Premier League, La Liga, Bundesliga, Serie A, Ligue 1, World Values)
- **3 Player Clusters** identified through KMeans algorithm
- **€78.6m** average market value across all players

## 🎯 Clustering Methodology

The dashboard uses the following features for clustering:
- Player age
- Market value
- Position encoding (GK, DEF, MID, FWD)
- Value-per-age ratio
- League strength proxy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This project is for educational and research purposes only. Transfermarkt data is used respectfully according to their terms of service. Please be mindful of web scraping ethics and rate limits.

## 🔗 Links

- [Live Dashboard](https://grisheet.github.io/transfermarkt-player-dashboard/)
- [Source Code](https://github.com/grisheet/transfermarkt-player-dashboard)
- [Issues](https://github.com/grisheet/transfermarkt-player-dashboard/issues)

---

Built with ❤️ using Python, Machine Learning, and Data Visualization
