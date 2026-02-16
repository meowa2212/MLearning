import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(loss_history: list) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker='o', color='b', linestyle='-')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.pause(5)
    plt.close("all")

def display(image: np.ndarray, title: np.ndarray) -> None:
    reshaped = image.reshape(28,28)
    plt.imshow(reshaped, cmap="gray")
    plt.colorbar()
    plt.axis("off")
    plt.title(np.argmax(title))
    plt.pause(3)
    plt.close("all")

def run_realtime_prediction(network):
    canvas = np.zeros((400, 400), dtype="uint8")
    window_name = "Draw! [c: Clear | d: Debug | q: Quit]"
    cv2.namedWindow(window_name)
    
    is_drawing = False

    def draw_event(event, x, y, flags, param):
        nonlocal is_drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            is_drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_drawing:
                # Use a thicker brush (20) to match MNIST stroke weight
                cv2.circle(canvas, (x, y), 20, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            is_drawing = False

    cv2.setMouseCallback(window_name, draw_event)

    while True:
        # 1. FIND THE DIGIT (Bounding Box)
        coords = cv2.findNonZero(canvas)
        
        final_28x28 = np.zeros((28, 28), dtype="uint8")

        if coords is not None:
            # Get the tightest rectangle around the drawing
            x, y, w, h = cv2.boundingRect(coords)
            
            # Extract the digit
            digit_only = canvas[y:y+h, x:x+w]
            
            # 2. RESIZE MAINTAINING ASPECT RATIO
            # We want the digit to fit in a 20x20 area (MNIST style)
            if w > h:
                new_w = 20
                new_h = int(h * (20 / w))
            else:
                new_h = 20
                new_w = int(w * (20 / h))
            
            # Ensure new_h/new_w are at least 1
            new_w, new_h = max(1, new_w), max(1, new_h)
            digit_resized = cv2.resize(digit_only, (new_w, new_h))
            
            # 3. CENTER IN 28x28
            # Calculate offsets to place the 20x20 (max) digit in the 28x28 box
            start_x = (28 - new_w) // 2
            start_y = (28 - new_h) // 2
            final_28x28[start_y:start_y+new_h, start_x:start_x+new_w] = digit_resized

        # 4. PREPARE FOR NETWORK
        model_input = final_28x28.reshape(784, 1) / 255.0

        # 5. PREDICT
        output = network.forward_pass(model_input)
        prediction = np.argmax(output)
        confidence = np.max(output) * 100

        # UI Visuals
        display_img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        cv2.putText(display_img, f"Prediction: {prediction} ({confidence:.1f}%)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show a small preview of what the network sees in the corner
        preview = cv2.resize(final_28x28, (100, 100))
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        display_img[300:400, 300:400] = preview_bgr
        cv2.rectangle(display_img, (300, 300), (399, 399), (255, 255, 255), 1)

        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'): canvas.fill(0)
        elif key == ord('d'):
            print(np.resize(output, (1,10)))
            display(model_input, output)

    cv2.destroyAllWindows()