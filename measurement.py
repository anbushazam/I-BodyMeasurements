import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import HRNET
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib


# Function to predict weight based on height, gender, and additional features
def predict_weight(height, gender, shoulder_breadth, waist, hip, chest,arm_length,leg_length):
    # Load the trained model and scaler
    model = joblib.load("weight_prediction_model_gb.pkl")
    scaler = joblib.load("scaler.pkl")
    
    # Prepare the input data
    gender_male = 1 if gender.lower() == 'male' else 0
    input_data = pd.DataFrame({
        'height': [height],
        'gender_male': [gender_male],
        'shoulder-breadth': [shoulder_breadth],
        'waist': [waist],
        'hip': [hip],
        'chest': [chest],
        'arm-length': [arm_length],
        'leg-length': [leg_length]
    })
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict the normalized weight
    predicted_weight_normalized = model.predict(input_data_scaled)
    
    # Denormalize the weight by multiplying it with the height
    predicted_weight = predicted_weight_normalized * height
    return predicted_weight

# Global variables to store image paths and height
front_image_path = None
side_image_path = None
angle_image_path = None
user_height_mm = None  # Store height in millimeters

class ImageDropLabel(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px dashed gray; padding: 10px;")
        self.setFixedSize(300, 200)
        self.setAcceptDrops(True)
        self.file_path = None  

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.file_path = urls[0].toLocalFile()  
            pixmap = QPixmap(self.file_path).scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(pixmap)
            self.setStyleSheet("border: 2px solid gray;")

class BodyMeasurementApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Body Measurement App')
        self.setGeometry(100, 100, 400, 600)

        layout = QVBoxLayout()
        
        self.front_label = ImageDropLabel('Drop Front View Here')
        self.side_label = ImageDropLabel('Drop Side View Here')
        self.angle_label = ImageDropLabel('Drop 45° View Here')
        self.gender_input = QLineEdit()
        self.gender_input.setPlaceholderText('Enter gender (male/female)')
        
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText('Enter height in cm')

        self.process_button = QPushButton('Process')
        self.process_button.clicked.connect(self.process_measurements)

        layout.addWidget(self.front_label)
        layout.addWidget(self.side_label)
        layout.addWidget(self.angle_label)
        layout.addWidget(self.height_input)
        layout.addWidget(self.process_button)
        layout.addWidget(self.gender_input)
        
        self.setLayout(layout)

    def process_measurements(self):
        if not all([self.front_label.file_path, self.side_label.file_path, self.angle_label.file_path]):
            QMessageBox.critical(self, "Error", "Please upload all three images!")
            return
    
        try:
            height_cm = float(self.height_input.text())
            user_height_mm = height_cm * 10
        except ValueError:
            QMessageBox.critical(self, "Error", "Enter a valid height in cm!")
            return
    
        gender = self.gender_input.text().strip().lower()
        if gender not in ["male", "female"]:
            QMessageBox.critical(self, "Error", "Enter gender as 'male' or 'female'")
            return
    
        front_image_path = self.front_label.file_path
        side_image_path = self.side_label.file_path
        angle_image_path = self.angle_label.file_path
    
        try:
            # Get measurements from HRNet
            (
                height, shoulder_width, torso_length, arm_length, leg_length,
                chest_circ, waist_circ, hip_circ
            ) = HRNET.body_measure(front_image_path, side_image_path, angle_image_path, user_height_mm)
    
            # Predict weight
            # if it's in another file
            predicted_weight_in_kg = predict_weight(
                height, gender, shoulder_width, waist_circ, hip_circ,
                chest_circ, arm_length, leg_length
            )
    
            # Display results
            result_text = (
                f"Predicted Weight: {predicted_weight_in_kg[0]:.2f} kg\n\n"
                f"Height: {height} cm\n"
                f"Shoulder Width: {shoulder_width} cm\n"
                f"Waist: {waist_circ} cm\n"
                f"Hip: {hip_circ} cm\n"
                f"Chest: {chest_circ} cm\n"
                f"Arm Length: {arm_length} cm\n"
                f"Leg Length: {leg_length} cm"
            )
    
            QMessageBox.information(self, "Predicted Weight & Measurements", result_text)
    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BodyMeasurementApp()
    window.show()
    app.exec_()  # Start UI loop

    # Now, HRNet will receive the selected image paths and height
    print("Running HRNet with these inputs:")
    print("Front Image:", front_image_path)
    print("Side Image:", side_image_path)
    print("45° Image:", angle_image_path)
    print("User Height (mm):", user_height_mm)


