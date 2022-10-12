import numpy as np

pose2d_error = []
ours_error = []

# open file and read the content in a list
with open('pose2d_mpjpe_z.txt', 'r') as fp:
    for line in fp:
        x = line[:-1]
        pose2d_error.append(float(x))
        
with open('best_1_mpjpe_z.txt', 'r') as fp:
    for line in fp:
        x = line[:-1]
        ours_error.append(float(x))
      
pose2d_error = np.array(pose2d_error)
ours_error = np.array(ours_error)
        
diff = pose2d_error - ours_error
idxs = diff.argsort()
idxs = idxs[::-1]
print(diff[idxs])
import pdb; pdb.set_trace()