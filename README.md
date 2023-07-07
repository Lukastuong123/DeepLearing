# Weird AI Hackovic - GenAI Hackathon 2023 
Our work includes the solutions: 
* GEN AI Backend pipeline: including 2 version of the Generative AI pipeline (feneral and finetuned), question/answer dataset for Language AI, example and example_masked picture for Vision AI 
* Figma Wireframes: Pdf file of interface that we have for the app
* Streamlit Demo App: 
### INSTRUCTIONS ###
To run the streamlit app on your local machine, install streamlit with `pip3 install streamlit`. Install the required python modules with `pip3 install requirements.txt`. Next, cd to the directory of the main app script (chat-bot-demo-app), and run `streamlit run hello_world.py. The app will automatically open on your default browser. Input any desired prompt, change any of the input parameters like temperature and tokens, and hit Enter. To preview the image transformation, please use the provided sample_image.png to upload as an example.
### FILE CONTENTS ###
** hello_world.py: Main script that runs the app
** inference_image.py: Script to run image inference. Does not work locally unless running a system that has Nvidia-enabled GPU processing.
** compute_service_account: Credentials needed to connect to the GCP project and VertexAI model
** requirements.txt: Modules to install
** sample_image.png: Example image to use for frontend image upload
