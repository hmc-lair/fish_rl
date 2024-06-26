from collections import OrderedDict
import numpy as np

# class CentroidTracker:
#     def __init__(self, maxObjects=0, maxDisappeared=50):
#         self.nextObjectID = 0
#         self.objects = OrderedDict()
#         self.disappeared = OrderedDict()
#         self.maxDisappeared = maxDisappeared
#         self.maxObjects = maxObjects

#     def register(self, centroid):
#         # if len(self.objects)-1 == self.maxObjects:
#         #     raise ValueError
#         self.objects[self.nextObjectID] = centroid
#         self.disappeared[self.nextObjectID] = 0
#         self.nextObjectID += 1

#     def deregister(self, objectID):
#         del self.objects[objectID]
#         del self.disappeared[objectID]

#     def update(self, inputCentroids):
#         if len(inputCentroids) == 0:
#             for objectID in list(self.disappeared.keys()):
#                 self.disappeared[objectID] += 1

#                 if self.disappeared[objectID] > self.maxDisappeared:
#                     self.deregister(objectID)

#             return self.objects

#         if len(self.objects) == 0:
#             for centroid in inputCentroids:
#                 self.register(centroid)
#         else:
#             objectIDs = list(self.objects.keys())
#             objectCentroids = list(self.objects.values())

#             D = self.compute_distance(objectCentroids, inputCentroids)

#             rows = D.min(axis=1).argsort()
#             cols = D.argmin(axis=1)[rows]

#             usedRows = set()
#             usedCols = set()

#             for (row, col) in zip(rows, cols):
#                 if row in usedRows or col in usedCols:
#                     continue

#                 objectID = objectIDs[row]
#                 self.objects[objectID] = inputCentroids[col]
#                 self.disappeared[objectID] = 0

#                 usedRows.add(row)
#                 usedCols.add(col)

#             unusedRows = set(range(0, D.shape[0])).difference(usedRows)
#             unusedCols = set(range(0, D.shape[1])).difference(usedCols)

#             for row in unusedRows:
#                 objectID = objectIDs[row]
#                 self.disappeared[objectID] += 1

#                 if self.disappeared[objectID] > self.maxDisappeared:
#                     self.deregister(objectID)

#             for col in unusedCols:
#                 self.register(inputCentroids[col])

#         return self.objects

#     def compute_distance(self, objectCentroids, inputCentroids):
#         D = np.zeros((len(objectCentroids), len(inputCentroids)))

#         for i in range(len(objectCentroids)):
#             for j in range(len(inputCentroids)):
#                 D[i, j] = np.linalg.norm(np.array(objectCentroids[i]) - np.array(inputCentroids[j]))

#         return D

class CentroidTracker:
    def __init__(self, n):
        self.n = n
        self.objects = OrderedDict()
        self.all_objects = OrderedDict()
    
    def update(self, centroids):
        centroids = np.array(centroids)
        if len(self.all_objects) == 0:
            # Populate objects initially
            for i, centroid in enumerate(centroids[:self.n]):
                self.all_objects[i] = centroid
                self.objects[i] = centroid
        else:
            prev_centroid_ids = np.array(list(self.all_objects.keys()))  # Index is the row index, value is the object id
            remaining_centroid_ids = list(set(range(self.n)) - set(prev_centroid_ids))
            prev_centroids = np.array(list(self.all_objects.values()))
            self.objects = OrderedDict()
            D = CentroidTracker.compute_distances(prev_centroids, centroids)

            object_row_idxs = D.min(axis=1).argsort()
            centroid_col_idxs = D.argmin(axis=1)[object_row_idxs]
            used_cols = set()
            for new_row_idx, new_col_idx in zip(object_row_idxs, centroid_col_idxs):
                if new_col_idx in used_cols:
                    continue
                used_cols.add(new_col_idx)
                self.all_objects[prev_centroid_ids[new_row_idx]] = centroids[new_col_idx]
                self.objects[prev_centroid_ids[new_row_idx]] = centroids[new_col_idx]
            
            # Add any remaining centroids as new objects
            remaining_cols = list(set(range(len(centroids))) - used_cols)
            for new_object_id, new_col_idx in zip(remaining_centroid_ids, remaining_cols):
                self.all_objects[new_object_id] = centroids[new_col_idx]
                self.objects[new_object_id] = centroids[new_col_idx]
        return self.objects

    def compute_distances(prev_centroids, centroids):
        # Compute pairwise distances
        D = np.zeros((len(prev_centroids), len(centroids)))
        for i in range(len(prev_centroids)):
            for j in range(len(centroids)):
                D[i, j] = np.linalg.norm(prev_centroids[i] - centroids[j])
        return D

if __name__ == "__main__":
    pass
    # ct = CentroidTracker(3)
    # ct.update([(1, 0)])
    # print(ct.objects)
    # ct.update([(0, 0), (2, 0)])
    # print(ct.objects)
    # ct.update([(0, 0), (3, 0), (2, 0), (1, 0)])
    # print(ct.objects)
    # n = 3
    # objects = OrderedDict()
    # objects[0] = (0, 1)
    # objects[1] = (1, 2)
    # objects[2] = (2, 3)
    # prev_centroid_ids = np.array(list(objects.keys()))
    # remaining_centroid_ids = list(set(range(n)) - set(prev_centroid_ids))
    # prev_centroids = np.array(list(objects.values()))
    # centroids = np.array([[0, 1], [2, 3], [1, 2]])  # Index is the 1st col index, value is the centroid
    # D = CentroidTracker.compute_distances(prev_centroids, centroids)
    # object_row_idxs = D.min(axis=1).argsort()  # Index is the 1st row index, value is 2nd row index
    # centroid_col_idxs = D.argmin(axis=1)[object_row_idxs]
    # # new_objects = OrderedDict()
    # # for new_row_idx, new_col_idx in zip(object_row_idxs, centroid_col_idxs):
    # #     new_objects[prev_centroid_ids[new_row_idx]] = centroids[new_col_idx]
    # # for new_object_id, new_col_idx in zip(remaining_centroid_ids, centroid_col_idxs[len(object_row_idxs):]):
    # #     new_objects[new_object_id] = centroids[new_col_idx]
    # # for new_row_idx, new_col_idx in zip(object_row_idxs, centroid_col_idxs):
    # #     objects[new_row_idx] = centroids[new_col_idx]
