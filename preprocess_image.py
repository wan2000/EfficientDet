import math
import numpy as np
import cv2

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
    
def get_vec(A, B, dist):
    x1, y1 = A
    x2, y2 = B
    vec_proj = [y2 - y1, x1 - x2]
    len = math.sqrt(vec_proj[0] * vec_proj[0] + vec_proj[1] * vec_proj[1])
    unit_vec_proj = np.array(vec_proj) / len
    return np.array(unit_vec_proj * dist) + np.array([A, B])

def get_polygons(polygon, dist):
    # return new polygons have distance dist to polygon
    n = len(polygon)
    lines = []
    for i in range(n):
        A = polygon[i]
        B = polygon[(i + 1) % n]
        lines.append(get_vec(A, B, dist))
    new_polygon = []
    for i in range(len(lines)):
        line1 = lines[i]
        line2 = lines[(i + 1) % len(lines)]
        new_polygon.append(line_intersection(line1, line2))
    return new_polygon
  
def convert_image(image, polygon, dist, verbose = 0):
    img = image.copy()
    poly = get_polygons(polygon, dist)
    poly = np.array(poly, dtype = np.int32)
    
    if (verbose > 0): img = cv2.polylines(img, [poly], True, (0,0,0), 5)

    minX = maxX = poly[0][0]
    minY = maxY = poly[0][1]
    for pts in poly:
      minX = min(minX, pts[0])
      maxX = max(maxX, pts[0])
      minY = min(minY, pts[1])
      maxY = max(maxY, pts[1])
    minX = max(0, minX)
    maxX = min(img.shape[1], maxX)
    minY = max(0, minY)
    maxY = min(img.shape[0], maxY)
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, [poly], (255,255,255))
    img = np.bitwise_and(np.array(mask), np.array(img))
    img = img[minY:maxY + 1, minX: maxX + 1]
    return img