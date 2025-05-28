# I-BodyMeasurements
AI powered Human body metrics

This study introduces an AI-powered body metrics system using HRNet for 
precise human pose estimation and regression models (linear, neural network) to 
accurately predict body measurements (height, chest, waist, hip, limbs) and 
weight. This approach overcomes traditional manual assessment errors by 
leveraging HRNet for key landmark identification and integrating demographic 
data for weight prediction. The COVID-19 pandemic has underscored its value, 
enabling remote healthcare, virtual consultations, automated weight tracking for 
early obesity diagnosis, and AI-driven fitness tools to combat sedentary 
lifestyles.Beyond healthcare, this technology benefits fitness and retail, notably 
enhancing online fashion shopping with precise measurement recommendations; 
its accuracy is validated on public datasets using MSE and RMSE. While 
challenges like posture variations and demographic biases persist, future work 
aims to refine feature extraction and integrate biometric attributes. This 
advancement in AI-driven body metrics is key to healthcare resilience during 
crises and drives innovation across multiple industries.

This  system utilizes HRNet, a high-resolution deep learning 
model, to extract key body measurements from images. The input consists of 
height images captured from the front side and at a 45-degree angle. Based on the 
output of HRNet, weight predictions are performed using a Gradient Boosting 
Regression algorithm(XGboost algorithms), trained on the Dagshub dataset, 
which contains 4000 data points. This system provides a robust and efficient 
approach to estimating body metrics, offering applications in healthcare, fitness 
tracking, and personalized clothing recommendations.


User Interface (UI): Built using PyQt5, it enables users to interact with 
the system through drag-and-drop image uploading and result display 
dialogs. 
• Pose Estimation Module: Uses HRNet to extract 2D human body 23 keypoints from input images. 
• Body Measurement Calculator: Computes body lengths and circumferences from detected keypoints. 
• Weight Prediction Model: A trained machine learning model (Gradient Boosting) predicts weight based on body part measurements. 
• Output Module: Displays the predicted weight to the user using a dialog box.

# TO EXECUTE THE APPLICATION FOLLOW THE STEPS:
 step 1:install all the neccessary python libraries and packages in environment or system ,check the requirement files
 step 2:clone and download the all the neccessary files from this page.
 step 3:in the HRNet.py and Meaurement.py chagne the output and input directory to any own folders.
 step 4:execute measurement.py in python terminal and then drag and provide the neccessary input images and data.

 # SAMPLE OUTPUT:
 This project able to get 95% and above accuracy in weight prediction even some times better than weighing scales.
 even use fully detection mode with accurate and stable full body images wihtout any artifacts ,good lighting and clear environment it can give BODY MEASUREMENTS 89% accurate
 if the input images contains irregular and inconsistencies in human subject angles,projections the output totatlly fails to give accurate or even relevant results.
 
 ![Screenshot 2025-05-26 110611](https://github.com/user-attachments/assets/f9143615-6587-4494-88a3-93b302941818)
 ![Screenshot 2025-05-26 113205](https://github.com/user-attachments/assets/0e21781a-e90c-4416-83f1-c75930a9b7ed)

 

 
