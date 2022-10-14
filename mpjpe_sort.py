import numpy as np

pose2d_error_x = []
pose2d_error_y = []
pose2d_error_z = []
ours_error_x = []
ours_error_y = []
ours_error_z = []

# open file and read the content in a list
with open('pose2d_mpjpe_z.txt', 'r') as fp:
    for line in fp:
        x = line[:-1]
        pose2d_error_x.append(float(x))

with open('pose2d_mpjpe_y.txt', 'r') as fp:
    for line in fp:
        x = line[:-1]
        pose2d_error_y.append(float(x))

with open('pose2d_mpjpe_z.txt', 'r') as fp:
    for line in fp:
        x = line[:-1]
        pose2d_error_z.append(float(x))

with open('best_1_mpjpe_x.txt', 'r') as fp:
    for line in fp:
        x = line[:-1]
        ours_error_x.append(float(x))

with open('best_1_mpjpe_y.txt', 'r') as fp:
    for line in fp:
        x = line[:-1]
        ours_error_y.append(float(x))

with open('best_1_mpjpe_z.txt', 'r') as fp:
    for line in fp:
        x = line[:-1]
        ours_error_z.append(float(x))
      

pose2d_error_x = np.array(pose2d_error_x)
pose2d_error_y = np.array(pose2d_error_y)
pose2d_error_z = np.array(pose2d_error_z)
ours_error_x = np.array(ours_error_x)
ours_error_y = np.array(ours_error_y)
ours_error_z = np.array(ours_error_z)

pose2d_error_xy = np.sqrt(pose2d_error_x**2 + pose2d_error_y**2)
ours_error_xy = np.sqrt(ours_error_x**2 + ours_error_y**2)

mask =  (pose2d_error_xy<75) * (ours_error_xy<75)
    
diff = pose2d_error_z - ours_error_z
idxs = diff.argsort()
idxs = idxs[::-1]

num = 0
for i in idxs:
    if mask[i]:
        #print(diff[i])
        print(i, end=',')
        num += 1

    if num > 30:
        break

    
