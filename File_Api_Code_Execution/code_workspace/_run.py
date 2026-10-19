import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

df = pd.read_csv('streaming.csv')
# Categorical conversion
cat_cols = ['UserID','SubscriptionTier','TopGenre']
for c in cat_cols:
    df[c] = df[c].astype('category')

# Churn rates
churn_tier = df.groupby('SubscriptionTier')['Churned'].mean().reset_index()
churn_tier['Rate'] = churn_tier['Churned']*100

# Top 10 genres by count
top_genres = df['TopGenre'].value_counts().nlargest(10).index
churn_genre = df[df['TopGenre'].isin(top_genres)].groupby('TopGenre')['Churned'].mean().reset_index()
churn_genre['Rate'] = churn_genre['Churned']*100

# Feature importance via RandomForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X = pd.get_dummies(df.drop(columns=['UserID','Churned']), drop_first=True)
y = df['Churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=5)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)

# Correlation matrix (numeric + churn)
numeric_cols = ['TotalViewingHoursLastMonth','BingeWatchingSessionsLastMonth','NumberOfUniqueTitlesWatchedLastMonth','AverageSessionDurationMinutes','CustomerServiceInteractionsLastYear','MonthlyCost','Churned']
corr = df[numeric_cols].corr()

# Plot
plt.figure(figsize=(16,12))
# 1. Tier churn bar
ax1 = plt.subplot(2,2,1)
sns.barplot(data=churn_tier, x='SubscriptionTier', y='Rate', palette='Blues_d', ax=ax1)
ax1.set_title('Churn Rate by Subscription Tier')
ax1.set_ylabel('Churn Rate (%)')
# 2. Genre churn bar (sorted)
ax2 = plt.subplot(2,2,2)
churn_genre_sorted = churn_genre.sort_values('Rate', ascending=False)
sns.barplot(data=churn_genre_sorted, x='TopGenre', y='Rate', palette='OrRd', ax=ax2)
ax2.set_title('Churn Rate by Top Genres')
ax2.set_ylabel('Churn Rate (%)')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
# 3. Feature importance
ax3 = plt.subplot(2,2,3)
sns.barplot(x=importances.values, y=importances.index, palette='Greens_d', ax=ax3)
ax3.set_title('Top 12 Feature Importances (RandomForest)')
ax3.set_xlabel('Importance')
# 4. Correlation heatmap
ax4 = plt.subplot(2,2,4)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'shrink': .7}, ax=ax4)
ax4.set_title('Correlation Matrix (Numeric Variables)')

plt.tight_layout()
plt.savefig('churn_summary.png')
print('Plot saved')
