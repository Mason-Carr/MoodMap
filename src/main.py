import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import os

# save all figures to a figures folder so we can use them in the report
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

# download dataset from kaggle
print("Loading dataset...")
path = kagglehub.dataset_download("rodolfofigueroa/spotify-12m-songs")
df = pd.read_csv(f"{path}/tracks_features.csv")
print(f"Full dataset: {len(df):,} songs")

# 1.2M songs is too slow for some algorithms so sample 50k for training
df_sample = df.sample(n=50000, random_state=42)
print(f"Working sample: {len(df_sample):,} songs")

# these 9 features are the audio characteristics we care about for mood
features = ['danceability', 'energy', 'valence', 'tempo',
            'acousticness', 'instrumentalness', 'speechiness',
            'liveness', 'loudness']

X = df_sample[features].copy()

# normalize everything - tempo is like 60-200 but valence is 0-1,
# so we need to standardize before running any distance-based clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# try k from 2 to 15 and plot elbow + silhouette to find the best k
print("\nFinding optimal k...")
k_range = range(2, 16)
inertias = []
silhouettes = []
for k in k_range:
    print(f"  Testing k={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(k_range, inertias, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax2.plot(k_range, silhouettes, 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score vs k')
plt.tight_layout()
save_fig('elbow_silhouette.png')

# k=6 gives the best balance between silhouette score and interpretability
k = 6

# --- K-MEANS ---
print("\nRunning K-Means...")
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
df_sample['kmeans_cluster'] = kmeans_labels

# --- GMM ---
# trying GMM because it allows soft/probabilistic cluster assignments
print("Running GMM...")
gmm = GaussianMixture(n_components=k, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
df_sample['gmm_cluster'] = gmm_labels

# --- AGGLOMERATIVE ---
print("Running Agglomerative Clustering...")
agg = AgglomerativeClustering(n_clusters=k)
agg_labels = agg.fit_predict(X_scaled)
df_sample['agg_cluster'] = agg_labels

# --- DBSCAN ---
# eps=0.5 and min_samples=5 tuned by trial and error
print("Running DBSCAN...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
df_sample['dbscan_cluster'] = dbscan_labels
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

# compare all 4 algorithms
print("\n" + "=" * 50)
print("CLUSTERING COMPARISON")
print("=" * 50)
print("\nSilhouette Scores (higher is better):")
sil_kmeans = silhouette_score(X_scaled, kmeans_labels)
sil_gmm = silhouette_score(X_scaled, gmm_labels)
sil_agg = silhouette_score(X_scaled, agg_labels)
db_kmeans = davies_bouldin_score(X_scaled, kmeans_labels)
db_gmm = davies_bouldin_score(X_scaled, gmm_labels)
db_agg = davies_bouldin_score(X_scaled, agg_labels)

print(f"  K-Means:       {sil_kmeans:.4f}")
print(f"  GMM:           {sil_gmm:.4f}")
print(f"  Agglomerative: {sil_agg:.4f}")
print(f"  DBSCAN:        N/A ({n_clusters_dbscan} clusters, {100*n_noise/len(dbscan_labels):.1f}% noise)")

print("\nDavies-Bouldin Index (lower is better):")
print(f"  K-Means:       {db_kmeans:.4f}")
print(f"  GMM:           {db_gmm:.4f}")
print(f"  Agglomerative: {db_agg:.4f}")
print(f"  DBSCAN:        N/A")

# bar chart so its easier to see the differences visually
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
methods = ['K-Means', 'GMM', 'Agglomerative']
sil_scores = [sil_kmeans, sil_gmm, sil_agg]
db_scores = [db_kmeans, db_gmm, db_agg]

ax1.bar(methods, sil_scores, color=['steelblue', 'coral', 'seagreen'])
ax1.set_ylabel('Silhouette Score')
ax1.set_title('Silhouette Score (higher = better)')
for i, v in enumerate(sil_scores):
    ax1.text(i, v + 0.002, f'{v:.3f}', ha='center', fontweight='bold')

ax2.bar(methods, db_scores, color=['steelblue', 'coral', 'seagreen'])
ax2.set_ylabel('Davies-Bouldin Index')
ax2.set_title('Davies-Bouldin Index (lower = better)')
for i, v in enumerate(db_scores):
    ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
save_fig('clustering_comparison.png')

# PCA down to 2D so we can actually visualize the clusters
print("\nRunning PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# plot all 4 methods side by side in PCA space
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
scatter_data = [
    (kmeans_labels, 'K-Means Clustering'),
    (gmm_labels, 'GMM Clustering'),
    (agg_labels, 'Agglomerative Clustering'),
    (dbscan_labels, f'DBSCAN ({n_clusters_dbscan} clusters, {100*n_noise/len(dbscan_labels):.0f}% noise)'),
]
for ax, (labels, title) in zip(axes.flatten(), scatter_data):
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.4, s=5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
plt.suptitle('PCA Projection — All Clustering Methods', fontsize=14, y=1.01)
plt.tight_layout()
save_fig('pca_all_methods.png')

# t-SNE gives better cluster separation than PCA for high-dimensional data
print("Running t-SNE (takes a few minutes)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.4, s=5)
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
ax1.set_title('K-Means Clusters (PCA projection)')

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.4, s=5)
ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2')
ax2.set_title('K-Means Clusters (t-SNE projection)')
plt.tight_layout()
save_fig('pca_vs_tsne.png')

# correlation heatmap to see which features are related
correlation_matrix = df_sample[features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
save_fig('correlation_heatmap.png')

# look at the average feature values per cluster to label them
print("\nK-Means Cluster Profiles:")
cluster_profiles = df_sample.groupby('kmeans_cluster')[features].mean()
print(cluster_profiles.round(3))

# mood labels we came up with by looking at the cluster profiles
mood_labels = {
    0: 'Party / Feel-Good',
    1: 'Sad Acoustic Ballads',
    2: 'Ambient / Instrumental',
    3: 'Chill Hip-Hop / R&B',
    4: 'Hype / Workout / Rock',
    5: 'Live Recordings',
}

# normalize profiles to 0-1 so colors are easier to read in the heatmap
plt.figure(figsize=(12, 5))
cluster_profiles_norm = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
cluster_profiles_norm.index = [f"C{i}: {mood_labels[i]}" for i in cluster_profiles_norm.index]
sns.heatmap(cluster_profiles_norm, annot=True, cmap='YlOrRd', fmt='.2f', linewidths=0.5)
plt.title('K-Means Cluster Profiles (normalized feature averages)')
plt.tight_layout()
save_fig('cluster_profiles_heatmap.png')

# histograms of each feature split by cluster
print("\nGenerating feature histograms...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, feat in zip(axes.flatten(), features):
    for cluster_id in range(k):
        subset = df_sample[df_sample['kmeans_cluster'] == cluster_id][feat]
        ax.hist(subset, bins=40, alpha=0.5, label=f'C{cluster_id}', density=True)
    ax.set_title(feat)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
plt.suptitle('Feature Distributions by Cluster', fontsize=14)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
save_fig('feature_histograms.png')

# group by decade and see how mood features changed over time
print("\nRunning decade analysis...")
df_sample['year'] = pd.to_numeric(df_sample['year'], errors='coerce')
df_decade = df_sample.dropna(subset=['year']).copy()
df_decade['decade'] = (df_decade['year'] // 10 * 10).astype(int)
decade_mood = df_decade.groupby('decade')[['energy', 'valence', 'danceability', 'acousticness']].mean()
decade_mood = decade_mood[decade_mood.index >= 1950]  # ignore anything before 1950, data gets sparse

plt.figure(figsize=(12, 5))
for col in decade_mood.columns:
    plt.plot(decade_mood.index, decade_mood[col], marker='o', label=col)
plt.xlabel('Decade')
plt.ylabel('Average Feature Value')
plt.title('Music Mood Trends by Decade')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_fig('decade_trends.png')

# pull songs from the full 1.2M for each artist so we get more data points
print("\nRunning artist analysis...")
famous_artists = [
    'Nirvana', 'Eagles', 'Stevie Wonder', 'Elton John', 'Drake', 'Gorillaz'
]

df_artist_source = df.copy()
df_artist_source['kmeans_cluster'] = np.nan

import ast as _ast

# the artists column is stored as a list string like "['Drake', 'Future']"
# so we need to parse it properly
def extract_primary_artist(val):
    try:
        artists = _ast.literal_eval(val)
        for a in artists:
            if a.strip() in famous_artists:
                return a.strip()
    except:
        pass
    return None

df_artist_source['artist_name'] = df_artist_source['artists'].apply(extract_primary_artist)
df_artists = df_artist_source.dropna(subset=['artist_name']).copy()

if len(df_artists) > 0:
    X_artists = df_artists[features].dropna()
    df_artists = df_artists.loc[X_artists.index]
    # reuse the same scaler from training so the cluster assignments are consistent
    X_artists_scaled = scaler.transform(X_artists)
    df_artists['kmeans_cluster'] = kmeans.predict(X_artists_scaled)

    artist_clusters = df_artists.groupby('artist_name')['kmeans_cluster'].value_counts(normalize=True).unstack(fill_value=0)
    # make sure all 6 cluster columns are there even if an artist has 0 songs in some cluster
    for c in range(k):
        if c not in artist_clusters.columns:
            artist_clusters[c] = 0.0
    artist_clusters = artist_clusters[sorted(artist_clusters.columns)]
    artist_clusters.columns = [f"C{c}: {mood_labels[c]}" for c in artist_clusters.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    artist_clusters.plot(kind='bar', stacked=True, colormap='tab10', ax=ax)
    plt.title('Mood Cluster Distribution by Artist')
    plt.xlabel('Artist')
    plt.ylabel('Proportion of Songs')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    save_fig('artist_cluster_distribution.png')

    print(f"  Found {len(df_artists)} songs from target artists:")
    for name in famous_artists:
        n = (df_artists['artist_name'] == name).sum()
        print(f"    {name}: {n} songs")
else:
    print("  Warning: no target artists found")

# train a neural net to predict cluster labels from raw features
# if it can learn the clusters well it means they're real and separable
print("\nTraining Neural Network classifier...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, kmeans_labels, test_size=0.2, random_state=42, stratify=kmeans_labels
)

# 3 hidden layers, relu activation, early stopping so it doesn't overfit
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=200,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
nn_accuracy = accuracy_score(y_test, y_pred)
print(f"  Neural Network Accuracy: {nn_accuracy:.4f} ({nn_accuracy*100:.2f}%)")
print(classification_report(y_test, y_pred, target_names=[f"C{i}" for i in range(k)]))

# confusion matrix as heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f"C{i}" for i in range(k)],
            yticklabels=[f"C{i}" for i in range(k)])
plt.title(f'Neural Network Confusion Matrix\n(Accuracy: {nn_accuracy*100:.2f}%)')
plt.xlabel('Predicted Cluster')
plt.ylabel('True Cluster')
plt.tight_layout()
save_fig('nn_confusion_matrix.png')

# training loss over iterations
plt.figure(figsize=(8, 4))
plt.plot(mlp.loss_curve_, label='Training Loss')
best = getattr(mlp, 'best_loss_', None)
if best is not None:
    plt.axhline(y=best, color='r', linestyle='--', label='Best Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Neural Network Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_fig('nn_training_loss.png')

# run on the full 1.2M dataset to validate the model generalizes
print("\n" + "=" * 50)
print("Running K-Means on FULL dataset (1.2M songs)...")
print("=" * 50)
X_full = df[features].dropna()
X_full_scaled = scaler.fit_transform(X_full)
kmeans_full = KMeans(n_clusters=k, random_state=42, n_init=10)
full_labels = kmeans_full.fit_predict(X_full_scaled)
df_full_labeled = df.loc[X_full.index].copy()
df_full_labeled['mood_cluster'] = full_labels
df_full_labeled['mood_label'] = df_full_labeled['mood_cluster'].map(mood_labels)

sil_full = silhouette_score(X_full_scaled, full_labels, sample_size=10000, random_state=42)
db_full = davies_bouldin_score(X_full_scaled, full_labels)
print(f"Full dataset — Silhouette: {sil_full:.4f}, Davies-Bouldin: {db_full:.4f}")

# pie chart of how songs are distributed across mood clusters at full scale
cluster_counts = pd.Series(full_labels).value_counts().sort_index()
plt.figure(figsize=(8, 8))
plt.pie(cluster_counts, labels=[mood_labels[i] for i in cluster_counts.index],
        autopct='%1.1f%%', colors=plt.cm.tab10.colors[:k])
plt.title('Song Distribution Across Mood Clusters\n(Full 1.2M Dataset)')
plt.tight_layout()
save_fig('full_dataset_cluster_distribution.png')

print("\n" + "=" * 50)
print("ALL DONE — figures saved to:", os.path.abspath(OUTPUT_DIR))
print("=" * 50)
