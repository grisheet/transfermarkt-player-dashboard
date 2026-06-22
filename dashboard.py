#!/usr/bin/env python3
"""
Transfermarkt Player Performance Dashboard
Scrapes player data, performs clustering analysis, and generates interactive visualizations.
"""

import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRANSFERMARKT_URLS = {
    'world_values': 'https://www.transfermarkt.com/spieler-statistik/wertvollstespieler/marktwertetop',
    'premier_league': 'https://www.transfermarkt.com/premier-league/marktwerte/wettbewerb/GB1',
    'la_liga': 'https://www.transfermarkt.com/laliga/marktwerte/wettbewerb/ES1',
    'bundesliga': 'https://www.transfermarkt.com/bundesliga/marktwerte/wettbewerb/L1',
    'serie_a': 'https://www.transfermarkt.com/serie-a/marktwerte/wettbewerb/IT1',
    'ligue_1': 'https://www.transfermarkt.com/ligue-1/marktwerte/wettbewerb/FR1'
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def scrape_transfermarkt_data():
    """Scrape player data from Transfermarkt market value tables"""
    all_players = []
    
    for league, url in TRANSFERMARKT_URLS.items():
        print(f"Scraping {league}...")
        try:
            # For demo purposes - Transfermarkt requires more complex scraping
            # This is a simplified version showing the structure
            tables = pd.read_html(url, header=0)
            if tables:
                df = tables[0]
                df['league_source'] = league
                all_players.append(df)
        except Exception as e:
            print(f"Error scraping {league}: {e}")
            # Use sample data for demonstration
            sample_data = create_sample_data(league)
            all_players.append(sample_data)
    
    # Combine all data
    df_combined = pd.concat(all_players, ignore_index=True)
    return df_combined

def create_sample_data(league):
    """Create sample player data for demonstration"""
    import numpy as np
    
    players_data = {
        'world_values': [
            ('Lamine Yamal', 'Right Winger', 18, '€200.00m'),
            ('Erling Haaland', 'Centre-Forward', 25, '€200.00m'),
            ('Kylian Mbappé', 'Centre-Forward', 27, '€180.00m'),
            ('Pedri', 'Central Midfield', 23, '€150.00m'),
            ('Michael Olise', 'Right Winger', 24, '€130.00m'),
        ],
        'premier_league': [
            ('Bukayo Saka', 'Right Winger', 24, '€110.00m'),
            ('Declan Rice', 'Central Midfield', 27, '€120.00m'),
            ('William Saliba', 'Centre-Back', 25, '€100.00m'),
            ('Cole Palmer', 'Attacking Midfield', 24, '€100.00m'),
            ('Alexander Isak', 'Centre-Forward', 26, '€90.00m'),
        ],
        'la_liga': [
            ('Jude Bellingham', 'Attacking Midfield', 22, '€120.00m'),
            ('Vinicius Junior', 'Left Winger', 25, '€120.00m'),
            ('Florian Wirtz', 'Attacking Midfield', 23, '€100.00m'),
            ('João Neves', 'Central Midfield', 21, '€120.00m'),
        ],
        'bundesliga': [
            ('Jamal Musiala', 'Attacking Midfield', 23, '€100.00m'),
            ('Florian Wirtz', 'Attacking Midfield', 23, '€100.00m'),
        ],
        'serie_a': [
            ('Khvicha Kvaratskhelia', 'Left Winger', 24, '€110.00m'),
            ('Lautaro Martínez', 'Centre-Forward', 28, '€110.00m'),
            ('Strahinja Pavlović', 'Centre-Back', 25, '€40.00m'),
        ],
        'ligue_1': [
            ('Vitinha', 'Defensive Midfield', 25, '€100.00m'),
            ('Achraf Hakimi', 'Right-Back', 27, '€70.00m'),
        ]
    }
    
    if league in players_data:
        data = players_data[league]
        df = pd.DataFrame(data, columns=['player', 'position', 'age', 'market_value'])
        df['league_source'] = league
        return df
    else:
        return pd.DataFrame(columns=['player', 'position', 'age', 'market_value', 'league_source'])

def clean_and_engineer_features(df):
    """Clean data and engineer features for clustering"""
    # Clean market value
    df['market_value_m'] = df['market_value'].str.replace('€', '').str.replace('m', '').str.replace('k', '')
    df['market_value_m'] = pd.to_numeric(df['market_value_m'].str.replace('.', ''), errors='coerce')
    
    # Position grouping
    position_mapping = {
        'Goalkeeper': 'GK',
        'Centre-Back': 'DEF', 'Left-Back': 'DEF', 'Right-Back': 'DEF',
        'Defensive Midfield': 'MID', 'Central Midfield': 'MID', 'Attacking Midfield': 'MID',
        'Left Winger': 'FWD', 'Right Winger': 'FWD', 'Centre-Forward': 'FWD'
    }
    df['position_group'] = df['position'].map(position_mapping)
    
    # One-hot encode positions
    df['pos_gk'] = (df['position_group'] == 'GK').astype(int)
    df['pos_def'] = (df['position_group'] == 'DEF').astype(int)
    df['pos_mid'] = (df['position_group'] == 'MID').astype(int)
    df['pos_fwd'] = (df['position_group'] == 'FWD').astype(int)
    
    # Feature engineering
    df['value_per_age'] = df['market_value_m'] / df['age']
    
    # League strength proxy (simplified)
    league_strength = {
        'world_values': 5,
        'premier_league': 4,
        'la_liga': 4,
        'bundesliga': 4,
        'serie_a': 4,
        'ligue_1': 3
    }
    df['league_strength_proxy'] = df['league_source'].map(league_strength)
    
    return df

def perform_clustering(df, n_clusters=3):
    """Perform KMeans clustering on player features"""
    # Features for clustering
    feature_cols = ['age', 'market_value_m', 'pos_gk', 'pos_def', 'pos_mid', 'pos_fwd', 
                    'value_per_age', 'league_strength_proxy']
    
    X = df[feature_cols].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df['pca_1'] = pca_coords[:, 0]
    df['pca_2'] = pca_coords[:, 1]
    
    # Assign cluster names
    cluster_names = {0: 'Emerging Goalkeepers', 1: 'Emerging Goalkeepers', 2: 'Elite Goalkeepers'}
    
    # Better cluster naming based on average values
    cluster_avg_value = df.groupby('cluster')['market_value_m'].mean().sort_values(ascending=False)
    cluster_mapping = {
        cluster_avg_value.index[0]: 'Elite Goalkeepers',
        cluster_avg_value.index[1]: 'Established Goalkeepers', 
        cluster_avg_value.index[2]: 'Emerging Goalkeepers'
    }
    df['cluster_name'] = df['cluster'].map(cluster_mapping)
    
    return df

def create_seaborn_plot(df):
    """Create Seaborn cluster scatter plot"""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='cluster_name', 
                   style='position_group', s=100, alpha=0.7)
    plt.title('Player Clusters by Style Proxy (PCA)', fontsize=16)
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('cluster_scatter_seaborn.png', dpi=300, bbox_inches='tight')
    print("Seaborn plot saved: cluster_scatter_seaborn.png")

def create_plotly_dashboard(df):
    """Create interactive Plotly dashboard HTML"""
    # Summary stats
    n_players = len(df)
    n_leagues = df['league_source'].nunique()
    n_clusters = df['cluster'].nunique()
    avg_value = df['market_value_m'].mean()
    
    # Scatter plot
    fig_scatter = px.scatter(df, x='pca_1', y='pca_2', color='cluster_name',
                            hover_data=['player', 'position', 'age', 'market_value'],
                            title='Player clusters by style proxy',
                            labels={'pca_1': 'pca_1', 'pca_2': 'pca_2', 'cluster_name': 'cluster_name'})
    
    # Bar chart
    position_counts = df.groupby('position_group')['market_value_m'].mean().reset_index()
    fig_bar = px.bar(position_counts, x='position_group', y='market_value_m',
                    title='Average Market Value by Position Group')
    
    # Create dashboard layout
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Transfermarkt Player Performance Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #1a1a1a;
                color: #fff;
            }}
            .header {{
                text-align: center;
                padding: 20px;
            }}
            .stats {{
                display: flex;
                justify-content: space-around;
                margin: 30px 0;
            }}
            .stat-box {{
                background: #2a2a2a;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 32px;
                font-weight: bold;
                color: #4CAF50;
            }}
            .description {{
                background: #2a2a2a;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Transfermarkt Player Performance Dashboard</h1>
            <p class="description">Dashboard built from Transfermarkt market-value tables with Python, Pandas, Plotly, Seaborn, and KMeans. Clusters reflect playing-style proxies from position mix, age, league context, and market value features.</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <div>Players</div>
                <div class="stat-value">{n_players}</div>
            </div>
            <div class="stat-box">
                <div>Leagues / sources</div>
                <div class="stat-value">{n_leagues}</div>
            </div>
            <div class="stat-box">
                <div>Clusters</div>
                <div class="stat-value">{n_clusters}</div>
            </div>
            <div class="stat-box">
                <div>Avg market value</div>
                <div class="stat-value">€{avg_value:.1f}m</div>
            </div>
        </div>
        
        <div id="scatter"></div>
        <div id="bar"></div>
        
        <script>
            var scatterData = {fig_scatter.to_json()};
            var barData = {fig_bar.to_json()};
            Plotly.newPlot('scatter', scatterData.data, scatterData.layout);
            Plotly.newPlot('bar', barData.data, barData.layout);
        </script>
    </body>
    </html>
    """
    
    with open('player_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Interactive dashboard saved: player_dashboard.html")

def main():
    """Main pipeline"""
    print("=== Transfermarkt Player Performance Dashboard ===\n")
    
    # Step 1: Scrape data
    print("Step 1: Scraping Transfermarkt data...")
    df = scrape_transfermarkt_data()
    
    # Step 2: Clean and engineer features
    print("\nStep 2: Cleaning and engineering features...")
    df = clean_and_engineer_features(df)
    
    # Step 3: Save raw data
    df.to_csv('transfermarkt_players.csv', index=False)
    print("Raw data saved: transfermarkt_players.csv")
    
    # Step 4: Clustering
    print("\nStep 3: Performing clustering analysis...")
    df = perform_clustering(df, n_clusters=3)
    
    # Step 5: Save clustered data
    df.to_csv('transfermarkt_players_clustered.csv', index=False)
    print("Clustered data saved: transfermarkt_players_clustered.csv")
    
    # Step 6: Create visualizations
    print("\nStep 4: Creating visualizations...")
    create_seaborn_plot(df)
    create_plotly_dashboard(df)
    
    print("\n=== Pipeline complete! ===")
    print("\nOutputs generated:")
    print("  - transfermarkt_players.csv")
    print("  - transfermarkt_players_clustered.csv")
    print("  - cluster_scatter_seaborn.png")
    print("  - player_dashboard.html")
    print("\nOpen player_dashboard.html in your browser to view the interactive dashboard.")

if __name__ == "__main__":
    main()
