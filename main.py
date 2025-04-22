import sys
import cv2
import base64
import os
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QTextEdit, QLabel, QLineEdit, QSplitter, 
                            QMessageBox)
from PyQt6.QtCore import Qt, QTimer, QEvent
from PyQt6.QtGui import QImage, QPixmap, QFont
from openai import OpenAI
import threading
import json
from dotenv import load_dotenv

class ImageAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        #Load tenvironment variables
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")

        # if not self.api_key:
        #     QMessageBox.critical(self,"API Key Error", "No API key found in the secret file,")
        #     sys.exit(1) 

        # Initialize UI properties
        self.setWindowTitle("Local Lens v1.2")
        self.setMinimumSize(1000, 700)
        
        # OpenAI API configuration
        self.api_key = "OPENAI_API_KEY"
        self.default_prompt = "Analyze the provided image. If it is a quiz question, answer it. Otherwise, explain what you see in the picture."
        self.setup_ui()
        
        # Initialize camera
        self.camera = None
        self.camera_id = 0
        self.frame = None
        self.captured_image = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Start camera
        self.start_camera()
        
    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Local Lens v1.2")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        main_layout.addWidget(title_label)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel (camera and controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)
        
        # Camera frame
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(480, 360)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("border: 2px solid #cccccc; border-radius: 5px; background-color: #000000;")
        left_layout.addWidget(self.camera_label)
        
        # Camera controls
        camera_controls = QHBoxLayout()
        self.capture_btn = QPushButton("Capture Image")
        self.capture_btn.setMinimumHeight(40)
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                font-weight: bold;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setMinimumHeight(40)
        self.reset_btn.clicked.connect(self.reset_capture)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 5px;
                font-weight: bold;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        camera_controls.addWidget(self.capture_btn)
        camera_controls.addWidget(self.reset_btn)
        left_layout.addLayout(camera_controls)
        
        # Add prompt input
        prompt_layout = QVBoxLayout()
        prompt_label = QLabel("Prompt:")
        prompt_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt for the AI here...")
        self.prompt_input.setText(self.default_prompt)
        self.prompt_input.setMinimumHeight(100)
        self.prompt_input.setStyleSheet("border: 1px solid #cccccc; border-radius: 5px; padding: 5px;")
        
        prompt_layout.addWidget(prompt_label)
        prompt_layout.addWidget(self.prompt_input)
        left_layout.addLayout(prompt_layout)
        
        # API Key input
        api_layout = QHBoxLayout()
        api_label = QLabel("OpenAI API Key:")
        api_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.api_input = QLineEdit()
        self.api_input.setPlaceholderText("Enter your OpenAI API key here...")
        self.api_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_input.setStyleSheet("border: 1px solid #cccccc; border-radius: 5px; padding: 5px;")
        
        api_layout.addWidget(api_label)
        api_layout.addWidget(self.api_input)
        left_layout.addLayout(api_layout)
        
        # Right panel (AI response)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)
        
        response_label = QLabel("AI Response:")
        response_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        right_layout.addWidget(response_label)
        
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 10px;
                background-color: #000000;
                font-size: 11pt;
            }
        """)
        right_layout.addWidget(self.response_text)
        
        # Analysis button
        self.analyze_btn = QPushButton("Analyze Image")
        self.analyze_btn.setMinimumHeight(50)
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12pt;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        right_layout.addWidget(self.analyze_btn)
        
        # Add save button
        self.save_btn = QPushButton("Save Results")
        self.save_btn.setMinimumHeight(40)
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border-radius: 5px;
                font-weight: bold;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        right_layout.addWidget(self.save_btn)
        
        # Add status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #666666; font-style: italic;")
        right_layout.addWidget(self.status_label)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([500, 500])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
    def start_camera(self):
        # Try to open the plugged-in camera first
        self.camera = cv2.VideoCapture(1)
        if self.camera.isOpened():
            self.camera_id = 1
        else:
            for cam in range(0,5):  # Check up to 5 camera indices
                self.camera = cv2.VideoCapture(cam)
                if self.camera.isOpened():
                    self.camera_id = cam
                    break
        if self.camera_id not in [0,1,2,3,4,5]:
            QMessageBox.critical(self, "Camera Error", "No camera found. Please check your camera connection.")
            return

        self.timer.start(30)  # Update every 30ms (33 fps)
        
    def update_frame(self):
        ret, self.frame = self.camera.read()
        if ret:
            # Convert frame to RGB format
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            
            # Convert to QImage and then to QPixmap
            img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            
            # Scale pixmap to fit the label while maintaining aspect ratio
            pixmap = pixmap.scaled(self.camera_label.width(), self.camera_label.height(), 
                                   Qt.AspectRatioMode.KeepAspectRatio)
            
            # Display the frame
            self.camera_label.setPixmap(pixmap)
        
    def capture_image(self):
        if self.frame is not None:
            self.captured_image = self.frame.copy()
            
            # Create 'img' folder if it doesn't exist
            img_dir = "img"
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            
            # Generate a sequential filename
            existing_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
            next_index = len(existing_files) + 1
            image_filename = f"image_{next_index:03d}.jpg"
            image_path = os.path.join(img_dir, image_filename)
            
            # Save the image
            cv2.imwrite(image_path, self.captured_image)
            
            # Store the path in curr_image
            self.curr_image = image_path
            
            # Update UI to show captured image
            frame_rgb = cv2.cvtColor(self.captured_image, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            pixmap = pixmap.scaled(self.camera_label.width(), self.camera_label.height(), 
                                Qt.AspectRatioMode.KeepAspectRatio)
            self.camera_label.setPixmap(pixmap)
            
            # Update button states
            self.capture_btn.setEnabled(False)
            self.reset_btn.setEnabled(True)
            self.analyze_btn.setEnabled(True)
            
            # Pause camera updates
            self.timer.stop()
            
            self.status_label.setText(f"Image captured and saved as {image_filename}! Ready for analysis.")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
    def reset_capture(self):
        # Clear captured image
        self.captured_image = None
        
        # Reset buttons
        self.capture_btn.setEnabled(True)
        self.reset_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        
        # Restart camera updates
        if not self.timer.isActive():
            self.timer.start(30)
        
        # Clear the response_text field
        self.response_text.clear()
        
        # Update status label
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("color: #666666; font-style: italic;")
    
    def analyze_image(self):
        # Check if API key is provided
        if not self.api_key:
            self.api_key = self.api_input.text().strip()
        if not self.api_key:
            QMessageBox.warning(self, "API Key Required", "Please enter your OpenAI API key.")
            return
            
        # Check if image is captured
        if self.captured_image is None:
            QMessageBox.warning(self, "No Image", "Please capture an image first.")
            return
            
        # Update UI
        self.analyze_btn.setEnabled(False)
        self.status_label.setText("Analyzing image...")
        self.status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        
        # Run analysis in a separate thread to avoid UI freezing
        prompt = self.generate_prompt()
        threading.Thread(target=self._run_analysis, args=(self.generate_prompt(),), daemon=True).start()

    def generate_prompt(self):
        # Get prompt
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            prompt = self.default_prompt
        return prompt
    
    #function to encode the image

    def encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _run_analysis(self, prompt):
        try:
            client = OpenAI()
            image_path = self.curr_image
            base64_image = self.encode_image(image_path)
            final_prompt = self.generate_prompt()

            response = client.responses.create(
                model="gpt-4o",
                input=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "input_text", "text": f"Hello. {final_prompt}"},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        ],
                    }
                ],
            )
            response_text = response.output_text
            # Append response to responses.txt
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("responses.txt", "a") as file:
                file.write(f"\n {timestamp} - {response_text}\n \n \n")

            # Update UI in the main thread
            QApplication.instance().postEvent(self, ResultEvent(response_text))

            
        except Exception as e:
            # Handle errors
            error_message = f"Error: {str(e)}"
            print(error_message)
            QApplication.instance().postEvent(self, ErrorEvent(error_message))
    
    def handle_result(self, result_text):
        self.response_text.setText(result_text)
        self.analyze_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.status_label.setText("Analysis complete!")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
    
    def handle_error(self, error_message):
        self.response_text.setText(f"ERROR: {error_message}")
        self.analyze_btn.setEnabled(True)
        self.status_label.setText("Analysis failed")
        self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
    
    def save_results(self):
        if not self.response_text.toPlainText():
            QMessageBox.warning(self, "No Results", "There are no results to save.")
            return
            
        # Create a folder for results if it doesn't exist
        results_dir = "analysis_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save the image
            image_path = os.path.join(results_dir, f"image_{timestamp}.jpg")
            cv2.imwrite(image_path, self.captured_image)
            
            # Save the analysis text
            text_path = os.path.join(results_dir, f"analysis_{timestamp}.txt")
            with open(text_path, 'w') as f:
                f.write(f" \n Prompt: {self.prompt_input.toPlainText()}\n\n")
                f.write(f"Analysis:\n{self.response_text.toPlainText()}")
                
            # Save a JSON file with all information
            json_path = os.path.join(results_dir, f"complete_{timestamp}.json")
            _, buffer = cv2.imencode('.jpg', self.captured_image)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            
            result_data = {
                "timestamp": timestamp,
                "prompt": self.prompt_input.toPlainText(),
                "response": self.response_text.toPlainText(),
                "image_path": image_path,
                "image_base64": encoded_image
            }
            
            with open(json_path, 'w') as f:
                json.dump(result_data, f, indent=2)
                
            QMessageBox.information(self, "Save Successful", 
                                   f"Results saved to folder '{results_dir}':\n"
                                   f"- Image: image_{timestamp}.jpg\n"
                                   f"- Analysis: analysis_{timestamp}.txt\n"
                                   f"- Complete data: complete_{timestamp}.json")
                                   
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save results: {str(e)}")
    
    def closeEvent(self, event):
        # Clean up resources when window is closed
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
        event.accept()


# Custom event for thread communication
class ResultEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
    
    def __init__(self, result_text):
        super().__init__(self.EVENT_TYPE)
        self.result_text = result_text


class ErrorEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
    
    def __init__(self, error_message):
        super().__init__(self.EVENT_TYPE)
        self.error_message = error_message


# Override event method to handle custom events
def event(self, event):
    if event.type() == ResultEvent.EVENT_TYPE:
        self.handle_result(event.result_text)
        return True
    elif event.type() == ErrorEvent.EVENT_TYPE:
        self.handle_error(event.error_message)
        return True
    return super(ImageAnalysisApp, self).event(event)

# Add the event method to the ImageAnalysisApp class
ImageAnalysisApp.event = event


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for a modern look
    
    # Set application-wide stylesheet
    app.setStyleSheet("""
    QMainWindow {
        background-color: #000000;  /* Set background color to black */
    }
    QLabel {
        font-size: 10pt;
        color: #ffffff;  /* Set text color to white for visibility */
    }
    QTextEdit, QLineEdit {
        font-size: 10pt;
        color: #ffffff;  /* Set text color to white for visibility */
        background-color: #333333;  /* Set input background to dark gray */
        border: 1px solid #555555;
    }
    QPushButton {
        font-size: 10pt;
        color: #ffffff;  /* Set button text color to white */
        background-color: #444444;  /* Set button background to dark gray */
        border: 1px solid #666666;
    }
    QPushButton:hover {
        background-color: #555555;  /* Slightly lighter gray on hover */
    }
    QPushButton:disabled {
        background-color: #222222;  /* Darker gray for disabled buttons */
        color: #777777;
    }
""")
    
    window = ImageAnalysisApp()
    window.show()
    sys.exit(app.exec())