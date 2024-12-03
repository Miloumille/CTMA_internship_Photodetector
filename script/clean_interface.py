import PySimpleGUI as sg
import cv2
import os
import pandas as pd
from functions import crop_image,get_green,get_stat_results,get_json,build_log

ROW_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
COLUMN_NUMBERS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
MAX_TEXT_LENGTH = 3
BUTTON_WIDTH = 3
BUTTON_HEIGHT = 1.2
GRID_ROWS = len(ROW_LABELS)
GRID_COLS = len(COLUMN_NUMBERS)
num_rectangles = 12
STATIC_IMAGE_PATH = "data/images_02_12/MW_G30_05_Extrafilter.jpg"


cell_data = {f"{row}{col}": {"state": None,"comment": ""} for row in ROW_LABELS for col in COLUMN_NUMBERS}
in_selection_mode_neg = False
in_selection_mode_pos = False

def pad_text(text, max_length):
    return text.ljust(max_length)

grid_layout = [
    [
        sg.Button(
            pad_text(f"{ROW_LABELS[row]}{COLUMN_NUMBERS[col]}", MAX_TEXT_LENGTH),
            size=(BUTTON_WIDTH, BUTTON_HEIGHT),
            pad=(0, 0),
            font=("Arial", 10),
            key=f"CELL-{row}-{col}",
            button_color=("black", "lightgray")
        )
        for col in range(GRID_COLS)
    ]
    for row in range(GRID_ROWS)
]

rectangles_layout = [
    [
        sg.Graph(
            canvas_size=(600, 200),
            graph_bottom_left=(0, 0),
            graph_top_right=(600, 200),
            key="-GRAPH-",
            enable_events=True,
        )
    ]
]

rectangles = []
def draw_rectangles():
    graph = window["-GRAPH-"]
    rect_width = 35
    rect_height = 300
    spacing = 15


    for i in range(num_rectangles):
        x1 = i * (rect_width + spacing)
        y1 = 10
        x2 = x1 + rect_width
        y2 = y1 + rect_height
        

        graph.draw_rectangle(
            (x1, y1),
            (x2, y2),
            line_color="black",
            fill_color="lightblue"
        )
        
        rectangles.append({
            'key': f"RECTANGLE-{i}",
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y1 + rect_height,
            'comment': ""
        })

under_grid_layout = [[sg.Button("Select ctrl -", size=(42, 1), pad=(20, 2), key="SELECT_BLANKS", button_color=("white", "navy")), sg.Button("Select ctrl +", size=(42, 1), pad=(20, 2), key="SELECT_POS", button_color=("white", "navy"))],
                     [sg.Button("Start", size=(42, 1), pad=(175, 2), key="START", button_color=("white", "navy"))]
                     ]

under_rect_layout = [[
        sg.Button("Select_blanks", size=(42, 1), pad=(20, 10), key="SELECT_BLANKS_rect", button_color=("white", "navy")),
        sg.Button("Start", size=(42, 1), pad=(20, 10), key="START_rect", button_color=("white", "navy"))
    ]]

layout = [
    [sg.Button("Grid", size=(43, 1.5), button_color=("white", "navy")), sg.Button("Plaques", size=(43, 1.5), button_color=("white", "navy"))],
    [sg.Column(grid_layout + under_grid_layout, key="-GRID-", pad=(15, 15), visible=False),
     sg.Image(filename='', key='-CAMERA-', visible=False, size=(500, 350),pad=(60,10)),
     sg.Column(rectangles_layout + under_rect_layout, key="-RECTANGLE_PAGE-", pad=(10, 15), visible=False)],
    [sg.Button("Process Image", key="-PROCESS_IMAGE-", size=(42, 1), button_color=("white", "navy"), pad=(170,0),visible=False)]
    ]

window = sg.Window(
    "Mobile LAB",
    layout,
    size=(640, 480),
    finalize=True
)

cap = None

while True:
    event, values = window.read(timeout=20)

    if event == sg.WINDOW_CLOSED:
        break

    if event == "Grid":
        window["-GRID-"].update(visible=True)
        window["-CAMERA-"].update(visible=False)
        window["-RECTANGLE_PAGE-"].update(visible=False)
        window["-PROCESS_IMAGE-"].update(visible=False)

    elif event == "Plaques":
        window["-RECTANGLE_PAGE-"].update(visible=True)
        window["-GRID-"].update(visible=False)
        window["-CAMERA-"].update(visible=False)
        window["-PROCESS_IMAGE-"].update(visible=False)
        draw_rectangles()

    elif event == "SELECT_BLANKS":
        
        in_selection_mode_neg = not in_selection_mode_neg

        if in_selection_mode_neg:
            print("Selection Mode NEG: ON")
            in_selection_mode_pos = False
            window["SELECT_POS"].update(button_color=("white", "navy"))
            window["SELECT_BLANKS"].update(button_color=("white", "gray"))
        else:
            print("Selection Mode NEG: OFF")
            window["SELECT_BLANKS"].update(button_color=("white", "navy"))
            
    elif event == "SELECT_POS":
        in_selection_mode_pos = not in_selection_mode_pos

        if in_selection_mode_pos:
            print("Selection Mode POS: ON")
            in_selection_mode_neg = False
            window["SELECT_POS"].update(button_color=("white", "lightblue"))
            window["SELECT_BLANKS"].update(button_color=("white", "navy"))
        else:
            print("Selection Mode POS: OFF")
            window["SELECT_POS"].update(button_color=("white", "navy"))

    elif event.startswith("CELL-"):
        _, row, col = event.split("-")
        row, col = int(row), int(col)
        cell_label = f"{ROW_LABELS[row]}{COLUMN_NUMBERS[col]}"

        if in_selection_mode_neg:
            if cell_data[cell_label]["state"] == None or cell_data[cell_label]["state"] == 1:
                cell_data[cell_label]["state"] = -1
                window[event].update(button_color=("black", "gray"))
            else:
                cell_data[cell_label]["state"] = None
                window[event].update(button_color=("black", "lightgray"))
                
        elif in_selection_mode_pos:
            if cell_data[cell_label]["state"] == None or cell_data[cell_label]["state"] == -1:
                cell_data[cell_label]["state"] = 1
                window[event].update(button_color=("black", "lightblue"))
            else:
                cell_data[cell_label]["state"] = None
                window[event].update(button_color=("black", "lightgray"))
                
        else:
            current_comment = cell_data[cell_label]["comment"]
            new_comment = sg.popup_get_text(
                f"Add/Edit comment for {cell_label}:",
                default_text=current_comment
            )
            if new_comment is not None:
                cell_data[cell_label]["comment"] = new_comment
                print(f"Comment for {cell_label} updated: {new_comment}")

    elif event == "START":
        window["-GRID-"].update(visible=False)
        window["-RECTANGLE_PAGE-"].update(visible=False)
        window["-CAMERA-"].update(visible=True)
        window["-PROCESS_IMAGE-"].update(visible=True,text = "Start", button_color=("white", "navy"))

#        if cap is None:
#            cap = cv2.VideoCapture(0)
#            if not cap.isOpened():
#                sg.popup_error("Failed to access the webcam. Please check your camera.")
#                break
#            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
        if os.path.exists(STATIC_IMAGE_PATH):
            image = cv2.imread(STATIC_IMAGE_PATH)
            image_resized = cv2.resize(image, (500, 350))
            imgbytes = cv2.imencode('.png', image_resized)[1].tobytes()
            window['-CAMERA-'].update(data=imgbytes)
        else:
            sg.popup_error(f"Image file not found: {STATIC_IMAGE_PATH}")
            
    elif event == "-PROCESS_IMAGE-":
        if any(cell["state"] == -1 for cell in cell_data.values()):
            window["-PROCESS_IMAGE-"].update(visible=False)
            imageInput = cv2.imread(STATIC_IMAGE_PATH)
            imageInputRGB = cv2.cvtColor(imageInput, cv2.COLOR_BGR2RGB)
            imageInputRGB = crop_image(imageInputRGB)
            final_image, cell_data, grid_mask = get_green(imageInputRGB,cell_data)
            final_image, cell_data = get_stat_results(imageInputRGB, cell_data, grid_mask)
            json = get_json(cell_data)
            build_log(json,cell_data)
            image_resized = cv2.resize(final_image, (500, 350))
            imgbytes = cv2.imencode('.png', cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))[1].tobytes()
            window['-CAMERA-'].update(data=imgbytes)
        else:
            window["-PROCESS_IMAGE-"].update(text="Please select negative control cells", button_color=("white", "red"))

            

    if cap:
        ret, frame = cap.read()
        if ret:
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['-CAMERA-'].update(data=imgbytes)
        else:
            sg.popup_error("Failed to capture video. Please ensure the webcam is working.")
            break
if cap:
    cap.release()
window.close()



