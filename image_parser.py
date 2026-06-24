import cv2
import numpy as np

COLOR_RANGES = {
    "R": ([0, 120, 70], [10, 255, 255]),
    "G": ([35, 80, 80], [85, 255, 255]),
    "B": ([100, 120, 70], [130, 255, 255]),
    "Y": ([20, 120, 120], [35, 255, 255]),
    "O": ([10, 120, 120], [20, 255, 255]),
}


def detect_color(hsv_cell):
    best = None
    best_score = 0

    for color, (low, high) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv_cell, np.array(low), np.array(high))
        score = np.sum(mask)

        if score > best_score and score > 1500:
            best = color
            best_score = score

    return best


def extract_grid(image_path, grid_size=5):
    img = cv2.imread(image_path)

    # speed: resize to reduce computation
    img = cv2.resize(img, (500, 500))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    rows = cols = grid_size

    h, w = img.shape[:2]
    cell_h = h // rows
    cell_w = w // cols

    board = [[None for _ in range(cols)] for _ in range(rows)]
    endpoints = {}

    for r in range(rows):
        for c in range(cols):

            cell = hsv[
                r * cell_h:(r + 1) * cell_h,
                c * cell_w:(c + 1) * cell_w
            ]

            color = detect_color(cell)

            if color:
                board[r][c] = color
                endpoints.setdefault(color, []).append((r, c))

    return board, endpoints, img