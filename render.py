import cv2

def draw_solution(image, paths, rows=5, cols=5):
    img = image.copy()

    h, w, _ = img.shape

    cell_h = h // rows
    cell_w = w // cols

    color_map = {
        "R": (255, 0, 0),
        "G": (0, 255, 0),
        "B": (0, 0, 255),
        "Y": (255, 255, 0),
        "O": (255, 165, 0)
    }

    for color, coords in paths.items():
        for i in range(len(coords) - 1):
            r1, c1 = coords[i]
            r2, c2 = coords[i + 1]

            x1 = c1 * cell_w + cell_w // 2
            y1 = r1 * cell_h + cell_h // 2

            x2 = c2 * cell_w + cell_w // 2
            y2 = r2 * cell_h + cell_h // 2

            cv2.line(
                img,
                (x1, y1),
                (x2, y2),
                color_map.get(color, (255, 255, 255)),
                4
            )

    return img