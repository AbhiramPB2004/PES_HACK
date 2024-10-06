import asyncio
import websockets
import cv2
import numpy as np

async def test_websocket():
    uri = "ws://localhost:2000/ws"  # Replace with your FastAPI WebSocket URL

    async with websockets.connect(uri) as websocket:
        while True:
            try:
                cam = cv2.VideoCapture(0) 

                _, img_encoded = cv2.imencode('.jpg',cam.read()[1])  # Encode the image
                img_bytes = img_encoded.tobytes()

                # Send image to the WebSocket as binary data
                await websocket.send(img_bytes)

                # Receive the response from the server
                response = await websocket.recv()
                print("Received from server:", response)
                
                # Add a delay if needed
                await asyncio.sleep(1)  # Adjust the delay as needed

            except websockets.ConnectionClosed:
                print("Connection closed by the server")
                break

asyncio.get_event_loop().run_until_complete(test_websocket())
