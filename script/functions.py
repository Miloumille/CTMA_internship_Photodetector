import cv2
import numpy as np
from scipy.signal import find_peaks
import json
import logging


def crop_image(image):

    r, g, b = cv2.split(image)

    _, red_mask = cv2.threshold(r, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 30]

    filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:4]

    centroids = []
    for contour in filtered_contours:

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append([cX, cY])
            
    print(centroids)
    
    center_of_mass = np.mean(centroids, axis=0)

    top_left = [pt for pt in centroids if pt[0] < center_of_mass[0] and pt[1] < center_of_mass[1]]
    top_right = [pt for pt in centroids if pt[0] > center_of_mass[0] and pt[1] < center_of_mass[1]]
    bottom_left = [pt for pt in centroids if pt[0] < center_of_mass[0] and pt[1] > center_of_mass[1]]
    bottom_right = [pt for pt in centroids if pt[0] > center_of_mass[0] and pt[1] > center_of_mass[1]]

    top_left = min(top_left, key=lambda x: np.linalg.norm(np.array(x) - center_of_mass))
    top_right = min(top_right, key=lambda x: np.linalg.norm(np.array(x) - center_of_mass))
    bottom_left = min(bottom_left, key=lambda x: np.linalg.norm(np.array(x) - center_of_mass))
    bottom_right = min(bottom_right, key=lambda x: np.linalg.norm(np.array(x) - center_of_mass))

    src_pts = np.float32([top_left, top_right, bottom_left, bottom_right])

    offset_top_left = np.array([-15, -42])
    offset_top_right = np.array([15, -42])
    offset_bottom_left = np.array([-15, 52])
    offset_bottom_right = np.array([15, 52])

    src_pts[0] += offset_top_left
    src_pts[1] += offset_top_right
    src_pts[2] += offset_bottom_left 
    src_pts[3] += offset_bottom_right 

    dst_pts = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])

    red_mask = cv2.erode(red_mask, np.ones((3, 3), np.uint8), iterations=2)
    red_mask = cv2.dilate(red_mask, np.ones((3, 3), np.uint8), iterations=4)
    image[red_mask == 255, 0] = 0
    image[red_mask == 255, 2] = 0 

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_image = cv2.warpPerspective(image, matrix, (500, 500))
    return warped_image

def get_green(imageInputRGB, cell_data):
    imageInputRGB = cv2.GaussianBlur(imageInputRGB, (5, 5), 3)
    
    cell_data, grid_mask = build_grid(imageInputRGB,cell_data)
    overlayed_image = cv2.addWeighted(imageInputRGB, 0.8, grid_mask, 0.5, 0)

    for key in cell_data:
        
        green_channel = cell_data[key]["cell_image"][:, :, 1]
        hist, bin_edges = np.histogram(green_channel.ravel(), bins=64, range=(0, 256))
        peaks, _ = find_peaks(hist, height=10)
        first_x_peak = peaks[-1]
        peak_height = hist[peaks[-1]]
        threshold = peak_height * 0.8
        low_idx = np.where(hist[:first_x_peak] < threshold)[0][-1] if len(np.where(hist[:first_x_peak] < threshold)[0]) > 0 else 0
        low_intensity = bin_edges[low_idx]

        mask = (green_channel >= low_intensity)
        kernel = np.ones((3, 3), np.uint8)  
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        cell_data[key]["mask"] = mask
        cell_data[key]["mean_green"] = np.mean(green_channel[mask == 1])
        
        # ---------
        
        cell_coords = cell_data[key]["coordinates"]
        mean_green = cell_data[key]["mean_green"]
        grid_value = cell_data[key]["state"]
        if grid_value == -1:
            text_color = (238, 75, 43)
        elif grid_value == 1:
            text_color = (30,144,255)
        else:
            text_color = (255, 255, 255)

        text = f"{mean_green:.0f}" if mean_green is not None else "None"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_x = (cell_coords["x_start"] + (cell_coords["x_end"]) - 30) // 2
        text_y = (cell_coords["y_start"] + (cell_coords["y_end"]) + 50) // 2
        cv2.putText(overlayed_image, text, (text_x, text_y), font, 0.4, text_color, 1, cv2.FONT_HERSHEY_SIMPLEX)            
    
    return overlayed_image, cell_data, grid_mask

def build_grid(image_input,cell_data):
    height, width, _ = image_input.shape
    vertical_positions = [0, 35, 75, 120, 165, 205, 245, 285, 330, 375, 415, 460, 500]
    horizontal_positions = [0, 55, 120, 185, 250, 315, 380, 440, 500]

    lab_image  = cv2.cvtColor(image_input, cv2.COLOR_RGB2LAB)

    grid_mask = np.zeros_like(image_input, dtype=np.uint8)
    for x in vertical_positions:
        cv2.line(grid_mask, (x, 0), (x, height), (255, 255, 255), 1)
    for y in horizontal_positions:
        cv2.line(grid_mask, (0, y), (width, y), (255, 255, 255), 1)

    
    for row_idx in range(len(horizontal_positions) - 1): # Letters
        for col_idx in range(len(vertical_positions) - 1): # Numbers
            label = chr(65 + row_idx) + str(col_idx + 1)
            cell_coords = {
                'x_start': vertical_positions[col_idx],
                'x_end': vertical_positions[col_idx + 1],
                'y_start': horizontal_positions[row_idx],
                'y_end': horizontal_positions[row_idx + 1]
            }

            cell_image = image_input[cell_coords['y_start']:cell_coords['y_end'], cell_coords['x_start']:cell_coords['x_end']]
            cell_lab_image = lab_image[cell_coords['y_start']:cell_coords['y_end'], cell_coords['x_start']:cell_coords['x_end']]
            # Store data for the cell 
            cell_data[label]["coordinates"] = cell_coords
            cell_data[label]["cell_image"] = cell_image
            cell_data[label]["cell_lab_image"] = cell_lab_image

    return cell_data, grid_mask

def get_stat_results(input_image, cell_data, grid_mask):
    negative_mean_green_values = [cell["mean_green"] for cell in cell_data.values() if cell.get("state") == -1]
    mean_negative_green = sum(negative_mean_green_values) / len(negative_mean_green_values)
    sd_negative_green = np.std(negative_mean_green_values)
    treshold = mean_negative_green + 2 * sd_negative_green
    for key in cell_data:
        if cell_data[key]["mean_green"] <= treshold:
            cell_data[key]["result"] = 0
        else:
            cell_data[key]["result"] = 1
            
            
    overlayed_image = cv2.addWeighted(input_image, 0.8, grid_mask, 0.5, 0)

    for key in cell_data:
        cell_coords = cell_data[key]["coordinates"]
        cell_mean_green = cell_data[key]["mean_green"]
        cell_state = cell_data[key]["state"]
        cell_result = cell_data[key]["result"]
        if cell_state == -1:
            text_color = (238, 75, 43)
        elif cell_state == 1:
            text_color = (30,144,255)
        elif cell_result == 1:
            text_color = (0, 255, 0)
        else:
            text_color = (255, 255, 255)
        
        text = f"{cell_mean_green:.0f}" if cell_mean_green is not None else "None"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_x = (cell_coords["x_start"] + (cell_coords["x_end"]) - 30) // 2
        text_y = (cell_coords["y_start"] + (cell_coords["y_end"]) + 50) // 2
        cv2.putText(overlayed_image, text, (text_x, text_y), font, 0.4, text_color, 1, cv2.FONT_HERSHEY_SIMPLEX)

    return overlayed_image, cell_data

def get_json(cell_data):
    state_mapping = { -1: "negative control", 1: "positive control" }
    filtered_data = {}
    for key, value in cell_data.items():
        state_value = state_mapping.get(value["state"], None)
        filtered_data[key] = {
            "state": state_value,
            "mean_green": value["mean_green"],
            "result": value["result"],
            "comment": value["comment"]
        }
    return json.dumps(filtered_data, indent=4)

def build_log(json_data,cell_data):
    logging.basicConfig(
        filename='cell_data_processing.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode='w'
    )
    
    for key, value in cell_data.items():
        if value["state"] == 1 and value["result"] == 0:
            logging.info(f'Cell {key} has been indicated as posive control but has a negative result !')
            
    logging.info(f'\nFiltered JSON Output: {json_data}')
    return