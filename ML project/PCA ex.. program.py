import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
data={ "Math":[80,90,70,60,85],
       "Sciene":[75,69,87,90,45],
       "English":[45,70,50,68,75],
       "Computer":[78,65,78,95,60]
 }
df=pd.DataFrame(data)
print(df)
scaler=StandardScaler()
X_scaled=scaler.fit_transform(df)
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)
pca_df=pd.DataFrame(X_pca,columns=["PC1","PC2"])
print("/n===PCA DATA===")
print(pca_df)
print("/n==== EXPLIAN VARIANCE====")
print(pca.explained_variance_ratio_)
print("/n Total Variance retained:",pca.explained_variance_ratio_.sum(()))