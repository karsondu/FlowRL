from image_parser import extract_grid
from solver import solve_flow
from render import draw_solution
import cv2

IMAGE_PATH = "puzzle.png"


def clean_endpoints(endpoints):
    cleaned = {}
    for color, pts in endpoints.items():
        if len(pts) >= 2:
            cleaned[color] = pts[:2]
    return cleaned


board, endpoints, img = extract_grid(IMAGE_PATH, grid_size=5)

print("RAW endpoints:", endpoints)

endpoints = clean_endpoints(endpoints)

print("CLEAN endpoints:", endpoints)

paths = solve_flow(board, endpoints)

output = draw_solution(img, paths)

cv2.imwrite("solved.png", output)

print("Saved solved.png")