#%%
import numpy as np	
#%%
X = np.array([
	[1,2],
	[3,4],
	[5,6],
	[7,8],
	])
p = [0.25, 0.25, 0.25, 0.25]
indices = np.random.choice(X.shape[0], X.shape[0], p=p, replace=True)
print(indices)

X_samle = X[indices]
print(X_samle)
# %%
np.e
# %%
np.log(np.e)
# %%
X = np.array([1,2,3,4])
X * X
# %%
import math
math.isclose(1, 1.0000000001)
# %%
