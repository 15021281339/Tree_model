import numpy as np
data = np.array([[1,2,3,4,5,6],
                 [7,8,9,10,11,12]])
feat = 1
arr1 = data[np.nonzero(data[:, feat].astype(float) < 3)]
arr2 = data[np.nonzero(data[:, feat].astype(float) >= 1)]
print(np.nonzero(data[:, feat].astype(float) >= 1))
## np.nonzero获取非零元素的索引