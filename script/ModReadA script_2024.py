import datetime
import PySimpleGUI as sg      
import time
#import picamera 
import cv2
import numpy as np
import pandas as pd
import os
sg.ChangeLookAndFeel('DarkAmber')      

    
layout = [      
    [sg.Text('ModReadA™ Manager ', size=(40, 1), justification='center', font=("Arial", 25), relief=sg.RELIEF_RIDGE)],    
    [sg.Frame(layout=[      
    [sg.Button('MULTI-WELL',font="Arial", size=(40,5),border_width=5), sg.Button('LFA',font="Arial", size=(40,5),border_width=5)]], title='Choose Test', title_color='white', title_location='n' ,font='Arial', relief=sg.RELIEF_RIDGE, tooltip='Use these to set flags')],
]      

window = sg.Window('ModReadA™ by B-LiFE', layout, default_element_size=(40, 1), grab_anywhere=False)     
 
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event=="Exit":
        break
    
    elif event =='MULTI-WELL':
        current_datetime = datetime.datetime.now()
        date_time_suffix = current_datetime.strftime('%m-%d_%H-%M-%S')
        filename = 'MW_' + date_time_suffix + '.jpeg'
       # with picamera.PiCamera() as camera:
          #  camera.resolution = (640, 480)
           # camera.framerate = 30
            # Wait for the automatic gain control to settle
           # time.sleep(2)
            # Now fix the values
          #  camera.contrast = 15
           # camera.saturation = 20
           # camera.sharpness = 15
           # camera.shutter_speed = 16000
           # camera.exposure_mode = 'snow'
           # camera.awb_mode = 'off'
          #  camera.awb_gains = (2 , 1.6)
           # filename = 'MW_' + date_time_suffix + '.jpeg'
           # camera.capture(filename)
        Img = "CLFA1.jpeg"
        imageInput = cv2.imread(Img)
        # Crop picture to fit the grid
        y=30
        x=0
        h=420
        w=640
        imageInput = imageInput[y:y+h, x:x+w]
        
        # ======================================================================================================================
        # Detect RED samples
        # Set range for red color and empty well
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                blue = imageInput[y, x, 0]
                green = imageInput[y, x, 1]
                red = imageInput[y, x, 2]
                print("Red: {}, Green: {}, Blue: {}".format(red,green,blue))              
                cv2.destroyAllWindows()  
                     
            lower_red = np.array([blue-20,green-30,red-1])
            upper_red = np.array([blue+20,green+10,red+5])
      
            
            mask1 = cv2.inRange(imageInput,lower_red, upper_red)
            mask2 = cv2.inRange(imageInput, (220,220,150), (255,255,255))
            
            ## Merge the mask and crop the red regions
            red_mask = cv2.bitwise_or(mask1, mask2 )
            kernal = np.ones((8, 8), "uint8")
            # Final mask
            red_mask = cv2.dilate(red_mask, kernal)
           
            #plt.imshow(res_red)
            # Creating contour to track red color
            contours, hierarchy = cv2.findContours(red_mask,
            									cv2.RETR_TREE,
            									cv2.CHAIN_APPROX_SIMPLE)
            
            cv2.drawContours(imageInput, contours, -1, (255, 255, 255), 2)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                imageResult = cv2.rectangle(imageInput, (x, y), (x + w, y + h), (255, 255, 255), 2)

                cv2.putText(imageResult, "neg", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

            
                    
            # ======================================================================================================================
            # The grid is 12 columns by 8 rows
            # Find for each cell if the software found some red

            # initialize final array with 0
            resultsArray = a = [[0] * 12 for i in range(8)]

            # size are in pixels and are the dimension of a square containing a sample
            GRID_SIZEy = 53
            GRID_SIZEx = 53
            #========================================================

            # The shape of an image is accessed by img.shape. It returns a tuple of the number of rows, columns,
            # and channels (if the image is color)
            rowPixels, columnPixelss, channels = imageInput.shape

            # contours are boxes around the red color
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 50:
                    x, y, w, h = cv2.boundingRect(contour)
                    for column in range(0, 12, 1):
                        for row in range(0, 8, 1):
                            startColumn = column * GRID_SIZEx
                            startRow = row * GRID_SIZEy
                            endColumn = (column + 1) * GRID_SIZEx
                            endRow = (row + 1) * GRID_SIZEy
                            if (startColumn < x < endColumn) and (startRow < y < endRow):
                                resultsArray[row][column] = 1
                                
            # print results
            for row in range(0, 8, 1):
            	print (resultsArray[row])
            
            df = pd.DataFrame(a)
            df = df.rename(index={0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}, columns={0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7", 7: "8", 8: "9", 9: "10", 10: "11", 11: "12"})
            df = df.replace({0: 'Pos', 1: 'Neg'})
            df.to_csv("Results.csv")
            df = pd.read_csv("Results.csv", sep=',', engine='python', header=None)
            data = df[1:].values.tolist()             
            header_list = df.iloc[0].tolist()
                    
            # ======================================================================================================================
            # Draw the grid
            # range(start, stop, step)
            # Why start at 3?
            for y in range(0, rowPixels - 1, GRID_SIZEy):
            # 	# line ( image, point 1, point 2, color, thickness, line type)
             	cv2.line(imageInput, (0, y), (columnPixelss, y), (0, 0, 255), 1, 1)
            #
            for x in range(0, columnPixelss - 1, GRID_SIZEx):
            # 	# line ( image, point 1, point 2, color, thickness, line type)
            	cv2.line(imageInput, (x, 0), (x, rowPixels), (0, 0, 255), 1, 1)
          
            # ======================================================================================================================
            imageRes= cv2.resize(imageInput, (0,0), fx=0.7, fy=0.7)
            cv2.imwrite("Results.png", imageRes)
            # to get the location of the current python file
            basedir = os.path.dirname(os.path.abspath(__file__))
            # to join it with the filename
            categorization_file = os.path.join(basedir,'Results.png')
            
            layout = [
                #[sg.Table(values=data, headings=header_list,col_widths=1, display_row_numbers=False, auto_size_columns=False, num_rows=min(25, len(data)), justification="center"),],
                [sg.Image("Results.png"), sg.Text("Analyse Successful!", size=(20,1), text_color="white", background_color="green", justification ="center", font=(20))]
                ]
            win = sg.Window('ModReader™ by B-LiFE', layout, modal=True, default_element_size=(20, 1), grab_anywhere=False)
            win.read(close=True)  
            event, values = window.read()
           
        cv2.namedWindow(winname='Select your well')
        cv2.setMouseCallback('Select your well', click_event)

        cv2.imshow('Select your well', imageInput)
        cv2.waitKey(0)
        cv2.destroyAllWindows()   
        
#FOR LFA SEGMENTATION HERE=============================================================================================================    

    elif event =='LFA':
        current_datetime = datetime.datetime.now()
        date_time_suffix = current_datetime.strftime('%m-%d_%H-%M-%S')
        filename = 'LFA_' + date_time_suffix + '.jpeg'
        #with picamera.PiCamera() as camera:
           # camera.resolution = (640, 480)
            #camera.framerate = 30
            #camera.rotation = 180
            # Wait for the automatic gain control to settle
           # time.sleep(2)
            # Now fix the values
           # camera.shutter_speed = 15000
           # camera.exposure_mode = 'backlight'
           # camera.awb_mode = 'off'
           # camera.awb_gains = (3.5,1.2)
           # camera.sharpness = 25
           # camera.contrast = 30
           # camera.brightness = 50
           # filename = 'LFA_' + date_time_suffix + '.jpeg'
           # camera.capture(filename)
        filename = 'CLFA1.jpeg'                              #### TO BE REMOVED
        imageInput = cv2.imread(filename)
        # Crop image
        y=162
        x=20
        h=90
        w=580
        imageInput = imageInput[y:y+h, x:x+w]
        
        #Detect band
        
        mask1 = cv2.inRange(imageInput, (0,0,30), (120,90,110))
        
        kernal = np.ones((10, 10), "uint8")
        red_mask = cv2.morphologyEx(mask1, cv2.MORPH_TOPHAT, kernal)
        
        kernal = np.ones((2, 2), "uint8")
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernal)
        
        #Contouring 
        contours, hierarchy = cv2.findContours(red_mask,
        									cv2.RETR_TREE,
        									cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(imageInput, contours, -1, (255,0,255),2)
        
        # ======================================================================================================================
        # The grid is 12 columns by 3 rows
        # Find for each cell if the software found some band
        
        # initialize final array with 0
        resultsArray = a = [[0] * 12 for i in range(3)]
        
        # size are in pixels and are the dimension of a square containing a sample
        GRID_SIZEy = 30
        GRID_SIZEx = 48
        #========================================================
        # The shape of an image is accessed by img.shape. It returns a tuple of the number of rows, columns,
        # and channels (if the image is color)
        rowPixels, columnPixelss, channels = imageInput.shape
        
        # contours are boxes around the red color
        for pic, contour in enumerate(contours):
        	area = cv2.contourArea(contour)
        	if area > 1:
        		x, y, w, h = cv2.boundingRect(contour)
        		for column in range(0, 12, 1):
        			for row in range(0, 3, 1):
        				startColumn = column * GRID_SIZEx
        				startRow = row * GRID_SIZEy
        				endColumn = (column + 1) * GRID_SIZEx
        				endRow = (row + 1) * GRID_SIZEy
        				if (startColumn < x < endColumn) and (startRow < y < endRow):
        					resultsArray[row][column] = 1
        # print results
        for row in range(0, 3, 1):
        	print (resultsArray[row])
        # Export Data    
        df = pd.DataFrame(a)
        df1 = df.rename(index={0: "Control", 1: "Target 1", 2: "Target 2",}, columns={0: "Sample 1", 1: "Sample 2", 2: "Sample 3", 3: "Sample 4", 4: "Sample 5", 5: "Sample 6", 6: "Sample 7", 7: "Sample 8", 8: "Sample 9", 9: "Sample 10", 10: "Sample 11", 11: "Sample 12"})
        df2 = df1.replace({1: 'Positive', 0: 'Negative'})
        df2.to_csv("Results.csv")
        df = pd.read_csv("Results.csv", sep=',', engine='python', header=None)
        data = df[1:].values.tolist()             
        header_list = df.iloc[0].tolist()
        # ======================================================================================================================
        # Draw the grid
        
        for y in range(0, rowPixels - 1, GRID_SIZEy):
        # 	# line ( image, point 1, point 2, color, thickness, line type)
         	cv2.line(imageInput, (0, y), (columnPixelss, y), (0, 0, 255), 1, 1)
        #
        for x in range(0, columnPixelss - 1, GRID_SIZEx):
        # 	# line ( image, point 1, point 2, color, thickness, line type)
        	cv2.line(imageInput, (x, 0), (x, rowPixels), (0, 0, 255), 1, 1)
              
        cv2.imwrite("Results.png", imageInput)
        # to get the location of the current python file
        basedir = os.path.dirname(os.path.abspath(__file__))
        # to join it with the filename
        categorization_file = os.path.join(basedir,'Results.png')
        
        layout = [
            #[sg.Table(values=data, headings=header_list,col_widths=1, display_row_numbers=False, auto_size_columns=False, num_rows=min(25, len(data)), justification="center"),],
            [sg.Image("Results.png"), sg.Text("Analyse Successful!", size=(20,1), text_color="white", background_color="green", justification ="center", font=(20))]
            ]
        win = sg.Window('ModReader™ by B-LiFE', layout, modal=True, default_element_size=(20, 1), grab_anywhere=False)
        win.read(close=True)  
        event, values = window.read()   
