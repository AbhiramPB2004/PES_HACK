from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import torch


app = FastAPI()
real_height = 15.0  
focal_length = 500  
model = torch.hub.load("ultralytics/yolov5", "yolov5s")
def calculate_distance(height_in_pixels):

    if height_in_pixels > 0:
        return (real_height * focal_length) / height_in_pixels
    return None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()
    while True:
        ret = []
        data = await websocket.receive_bytes()  
        np_array = np.frombuffer(data, np.uint8) 
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR) 
        results = model(image)
        for i in range(len(results.xyxy[0])):
            x1, y1, x2, y2, conf, class_id = results.xyxy[0][i]
            height_in_pixels = y2 - y1  
            distance = calculate_distance(height_in_pixels)
             # Ensure data is serializable
            label = f"{model.names[int(class_id)]}: {distance:.2f} cm" if distance else f"{model.names[int(class_id)]}: N/A"
            print(f"Distance: {distance} meters")
            print(f"Confidence: {conf}")
            print(f"Class ID: {class_id}")
            ret.append((x1.item(), y1.item(), x2.item(), y2.item(), conf.item(), class_id.item(), label)) 

        print(ret)
        await websocket.send_text(str(ret))  # Send as a string (you can format this as needed)

