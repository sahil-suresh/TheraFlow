import os
import sys
import json
import threading
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTextEdit, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QGridLayout, QSpacerItem, QSizePolicy, QListWidget, QListWidgetItem, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QEvent  
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap  
from vosk import Model, KaldiRecognizer
import pyaudio
from groq import Groq
import re
import pandas as pd

# Set the API key in your environment variable sahilursa
os.environ['GROQ_API_KEY'] = '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'

class Worker(QThread):
    # Define signals to communicate with the GUI
    transcribed_text_signal = pyqtSignal(str)
    diagnoses_signal = pyqtSignal(str, str)  # Will send diagnoses_list, treatment_plans
    clinical_questions_signal = pyqtSignal(str)
    layman_summary_signal = pyqtSignal(str)
    final_summary_signal = pyqtSignal(dict)
    # Will send clinical_questions

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.transcriptions = []
        self.transcriptions_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.client = Groq()
        # Initialize the Vosk model
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.audio_stream = None
        self.pyaudio_instance = pyaudio.PyAudio()
        
    def run(self):
        # Start the audio stream
        self.audio_stream = self.pyaudio_instance.open(format=pyaudio.paInt16, channels=1, rate=16000,
                                                       input=True, frames_per_buffer=8000)
        self.audio_stream.start_stream()

        # Initialize timers for API calls
        next_update_time_diagnoses = time.time() + 10  # For diagnoses and treatment plans
        next_update_time_questions = time.time() + 40  # For clinical questions

        try:
            while not self.stop_event.is_set():
                # Read data from the audio stream
                data = self.audio_stream.read(4000, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    result = self.recognizer.Result()
                    text = json.loads(result).get("text", "")
                    if text:
                        words = text.split()
                        with self.transcriptions_lock:
                            self.transcriptions.extend(words)
                        # Emit the transcribed text
                        self.transcribed_text_signal.emit(' '.join(words))
                else:
                    # Partial result can be used if needed
                    pass

                current_time = time.time()
                # Check if it's time to update diagnoses and treatment plans
                if current_time >= next_update_time_diagnoses:
                    self.update_diagnoses()
                    next_update_time_diagnoses += 10  # Schedule the next update in 10 seconds

                # Check if it's time to update clinical questions
                if current_time >= next_update_time_questions:
                    self.update_clinical_questions()
                    next_update_time_questions += 40  # Schedule the next update in 40 seconds

                # Sleep briefly to prevent high CPU usage
                time.sleep(0.01)

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Close the audio stream
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.pyaudio_instance.terminate()
            # At the end of the conversation, make a final API call to get a summary
            self.get_final_summary()

    def update_diagnoses(self):
        # Safely access the transcriptions
        with self.transcriptions_lock:
            conversation = ' '.join(self.transcriptions)
        if conversation.strip():
            # Get diagnoses and treatment plans from Groq API
            diagnoses_list, treatment_plans = self.get_diagnoses_and_treatments(conversation)
            # Emit the diagnoses and treatment plans
            self.diagnoses_signal.emit(diagnoses_list, treatment_plans)
        else:
            print("\nNo conversation to analyze yet.")

    def update_clinical_questions(self):
        # Safely access the transcriptions
        with self.transcriptions_lock:
            conversation = ' '.join(self.transcriptions)
        if conversation.strip():
            # Get clinical questions from Groq API
            clinical_questions = self.get_clinical_questions(conversation)
            # Emit the clinical questions
            self.clinical_questions_signal.emit(clinical_questions)
        else:
            print("\nNo conversation to analyze yet.")

    def get_diagnoses_and_treatments(self, conversation):
        # Initialize empty strings for each output
        diagnoses_list = ''
        treatment_plans = ''

        # First API call to get differential diagnoses
        response_diagnoses = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical diagnostic assistant listening to a conversation between a doctor and a patient. Based on the patient's statements, provide only a comma-separated list of three possible differential diagnoses ordered by likelihood. There should not be any other text, headers, titles, or explanations besides the list."
                },
                {
                    "role": "user",
                    "content": conversation
                }
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=100,
            top_p=1,
            stop=None,
            stream=False,
        )
        diagnoses_list = response_diagnoses.choices[0].message.content.strip()

        # Second API call to get treatment plans
        response_treatment = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical diagnostic assistant. Based on the patient's statements, provide only a comma-separated list of three accurate treatment plans or tests to order, if it is a pharmacological treatment include the drug name and generic brand name and if it is a test to order, be as specific as possible and order results by most appropriate. There should not be any other text, headers, titles, or explanations besides the list."
                },
                {
                    "role": "user",
                    "content": conversation
                }
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=200,
            top_p=1,
            stop=None,
            stream=False,
        )
        treatment_plans = response_treatment.choices[0].message.content.strip()

        # Return the two outputs
        return diagnoses_list, treatment_plans

    def get_clinical_questions(self, conversation):
        # Initialize empty string for clinical questions
        clinical_questions = ''

        # Third API call to get clinical questions
        response_questions = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical diagnostic assistant. Based on the patient's statements, provide a numbered list of three most important questions to ask the patient that have not been answered in the conversation ordered by importance. There should not be any other text, headers, titles, or explanations besides the list."
                },
                {
                    "role": "user",
                    "content": conversation
                }
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=150,
            top_p=1,
            stop=None,
            stream=False,
        )
        clinical_questions = response_questions.choices[0].message.content.strip()

        # Return the clinical questions
        return clinical_questions
    
    def get_final_summary(self):
        # Safely access the entire conversation
        with self.transcriptions_lock:
            conversation = ' '.join(self.transcriptions)
    
        if conversation.strip():
            # Final API call for summary extraction
            response_summary = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical assistant summarizing the conversation. Extract the following information as a comma-separated list with headers as follows in this order: Patient Name:, Current Diagnosis:, Treatment Plan:, and Medications:. There should be no other text or explanations besides the list and headers."
                    },
                    {
                        "role": "user",
                        "content": conversation
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.3,
                max_tokens=300,
                top_p=1,
                stop=None,
                stream=False,
            )
            summary = response_summary.choices[0].message.content.strip()
            
            layman_summary = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical assistant summarizing the conversation. Based on the conversation explain the Current Diagnosis, Treatment Plan, and Medications in layman's terms to someone with little clinical knowledge and only output just the explanation."
                    },
                    {
                        "role": "user",
                        "content": conversation
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.3,
                max_tokens=300,
                top_p=1,
                stop=None,
                stream=False,
            )
            laymansummary = layman_summary.choices[0].message.content.strip()
            print(summary)
            print(laymansummary)
            self.layman_summary_signal.emit(laymansummary)
            # Use regex to capture each section based on the keywords
            patient_name_match = re.search(r"Patient Name:\s*(.*?)(?=Current Diagnosis:|Treatment Plan:|Medications:|$)", summary, re.DOTALL)
            diagnoses_match = re.search(r"Current Diagnosis:\s*(.*?)(?=Treatment Plan:|Medications:|Patient Name:|$)", summary, re.DOTALL)
            treatment_plan_match = re.search(r"Treatment Plan:\s*(.*?)(?=Medications:|Patient Name:|Current Diagnosis:|$)", summary, re.DOTALL)
            medications_match = re.search(r"Current Medications:\s*(.*?)(?=Patient Name:|Current Diagnosis:|Treatment Plan:|$)", summary, re.DOTALL)
    
            # Extract the matched text or default to "Not provided" if not found
            patient_name = patient_name_match.group(1).strip() if patient_name_match else "Not provided"
            current_diagnoses = diagnoses_match.group(1).strip() if diagnoses_match else "Not provided"
            treatment_plan = treatment_plan_match.group(1).strip() if treatment_plan_match else "Not provided"
            current_medications = medications_match.group(1).strip() if medications_match else "Not provided"
    
            self.final_summary_signal.emit({
                'Patient Name': patient_name,
                'Current Diagnosis': current_diagnoses,
                'Treatment Plan': treatment_plan,
                'Current Medications': current_medications
            })


    def stop(self):
        self.stop_event.set()

class MainWindow(QMainWindow):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker 
        self.installEventFilter(self)

        self.setWindowTitle("TheraFlow")
        self.resize(900, 1400)  # Increased window height
        
        # Initialize the set to keep track of clinical questions
        self.clinical_questions_set = set()

        # Set the color palette for the application
        self.setStyleSheet("""
            QWidget {
                background-color: #114B19;  /* Main background color */
            }
            QTextEdit {
                background-color: #ffffff;  /* White text edit background */
                border: 1px solid #ffffff;  /* White border */
                border-radius: 5px;
                padding: 5px;
                font-size: 20px;
                font-family: 'Segoe UI';
                color: #000000;  /* Text color to match main background */
            }
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #ffffff;  /* White text for labels */
                font-family: 'Segoe UI';
            }
            QGroupBox {
                border: 1px solid #ffffff;  /* White border for group box */
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                color: #ffffff;  /* White title text for group box */
            }
            QListWidget {
                background-color: #ffffff;  /* White background for list widget */
                border: 1px solid #ffffff;  /* White border */
                border-radius: 5px;
                padding: 5px;
                font-size: 20px;
                font-family: 'Segoe UI';
                color: #000000;  /* Text color to match main background */
            }
        """)


        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        logo_path = os.path.join(os.getcwd(), "Logo1.png")
    
        # Logo image in place of the title
        logo_label = QLabel()
        logo_pixmap = QPixmap(logo_path)
        if logo_pixmap.isNull():
            print("Failed to load Logo.png. Make sure it's in the current working directory.")
        else:
            logo_label.setPixmap(logo_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(logo_label)

        # Create group boxes for each section
        self.create_transcription_section()
        self.create_diagnoses_section()
        self.create_treatment_section()
        self.create_questions_section()
        self.create_layman_summary_section()

        # Set stretch factors to allocate more space to Clinical Questions
        self.layout.setStretch(1, 1)  # Transcription section
        self.layout.setStretch(2, 1)  # Diagnoses section
        self.layout.setStretch(3, 1)  # Treatment section
        self.layout.setStretch(4, 3) 
        self.layout.setStretch(5, 3)# Clinical Questions section

        # Add a spacer at the bottom
        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
    def create_layman_summary_section(self):
        summary_group = QGroupBox("Summary for Patient")
        summary_layout = QVBoxLayout()
        summary_group.setLayout(summary_layout)

        self.layman_summary_text_edit = QTextEdit()
        self.layman_summary_text_edit.setReadOnly(True)
        summary_layout.addWidget(self.layman_summary_text_edit)

        self.layout.addWidget(summary_group)

    # Add the method to update the layman summary
    def update_layman_summary(self, summary_text):
        self.layman_summary_text_edit.setPlainText(summary_text)

    def create_transcription_section(self):
        transcription_group = QGroupBox("Transcribed Conversation")
        transcription_layout = QVBoxLayout()
        transcription_group.setLayout(transcription_layout)

        self.transcribed_text_edit = QTextEdit()
        self.transcribed_text_edit.setReadOnly(True)
        transcription_layout.addWidget(self.transcribed_text_edit)

        self.layout.addWidget(transcription_group)

    def create_diagnoses_section(self):
        diagnoses_group = QGroupBox("Differential Diagnoses")
        diagnoses_layout = QVBoxLayout()
        diagnoses_group.setLayout(diagnoses_layout)

        self.diagnoses_text_edit = QTextEdit()
        self.diagnoses_text_edit.setReadOnly(True)
        self.diagnoses_text_edit.setFixedHeight(50)
        diagnoses_layout.addWidget(self.diagnoses_text_edit)

        self.layout.addWidget(diagnoses_group)

    def create_treatment_section(self):
        treatment_group = QGroupBox("Treatment Plans")
        treatment_layout = QVBoxLayout()
        treatment_group.setLayout(treatment_layout)

        self.treatment_plans_text_edit = QTextEdit()
        self.treatment_plans_text_edit.setReadOnly(True)
        self.treatment_plans_text_edit.setFixedHeight(50)
        treatment_layout.addWidget(self.treatment_plans_text_edit)

        self.layout.addWidget(treatment_group)

    def create_questions_section(self):
        questions_group = QGroupBox("Clinical Questions")
        questions_layout = QVBoxLayout()
        questions_group.setLayout(questions_layout)

        self.clinical_questions_list_widget = QListWidget()
        self.clinical_questions_list_widget.itemChanged.connect(self.cross_out_item)
        self.clinical_questions_list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        questions_layout.addWidget(self.clinical_questions_list_widget)

        self.layout.addWidget(questions_group)

    def parse_clinical_questions(self, clinical_questions):
        # Parses the numbered list into individual questions
        questions = []
        lines = clinical_questions.strip().split('\n')
        for line in lines:
            # Remove leading number and dot
            question = line.strip()
            # Find the first dot followed by a space
            index = question.find('. ')
            if index != -1:
                question = question[index+2:].strip()
            else:
                # Try with just a dot
                index = question.find('.')
                if index != -1:
                    question = question[index+1:].strip()
            if question:
                questions.append(question)
        return questions

    def cross_out_item(self, item):
        if item.checkState() == Qt.Checked:
            font = item.font()
            font.setStrikeOut(True)
            item.setFont(font)
        else:
            font = item.font()
            font.setStrikeOut(False)
            item.setFont(font)

    def update_transcribed_text(self, text):
        # Append the transcribed text to the text edit
        self.transcribed_text_edit.append(text)

    def update_diagnoses(self, diagnoses_list, treatment_plans):
        # Update the diagnoses and treatment plans
        self.diagnoses_text_edit.setPlainText(diagnoses_list)
        self.treatment_plans_text_edit.setPlainText(treatment_plans)

    def update_clinical_questions(self, clinical_questions):
        # Parse and update clinical questions
        questions = self.parse_clinical_questions(clinical_questions)
        for question in questions:
            if question not in self.clinical_questions_set:
                self.clinical_questions_set.add(question)
                item = QListWidgetItem(question)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.clinical_questions_list_widget.addItem(item)
                
    def eventFilter(self, source, event):
        # Detect if the Ctrl key is pressed
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Control:
            self.stop_recording()  # Call the stop method
            return True  # Mark the event as handled
        return super().eventFilter(source, event)

    def stop_recording(self):
        # Stop the worker thread and print transcribed text
        self.worker.stop()  # Stop the recording worker
        self.worker.wait()  # Wait for the worker to finish
        # Print the final transcribed text from the text box
        print("Transcribed Text:", self.transcribed_text_edit.toPlainText())
        
    def show_final_summary(self, data):
        # Create a pop-up window
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Final Summary")
        msg_box.setIcon(QMessageBox.Information)

        # Format the message
        message = (
            f"Patient Name: {data['Patient Name']}\n"
            f"Current Diagnosis: {data['Current Diagnosis']}\n"
            f"Treatment Plan: {data['Treatment Plan']}\n"
            f"Current Medications: {data['Current Medications']}"
        )
        msg_box.setText(message)

        # Show the pop-up
        msg_box.exec_()

        # Save the data as a CSV file
        from datetime import datetime

        # Create a DataFrame
        df = pd.DataFrame([data])

        # Get current datetime
        now = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Clean the patient name for use in filename
        safe_patient_name = re.sub(r'[^\w\s-]', '', data['Patient Name']).strip().replace(' ', '_')

        # Construct the filename
        filename = f"{safe_patient_name}_{now}.csv"

        # Save the DataFrame as a CSV file
        df.to_csv(filename, index=False)

def main():
    # Get the current working directory
    current_dir = os.getcwd()

    # Find the model directory in the current working directory
    model_dir = None
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path) and item.startswith('vosk-model'):
            model_dir = item_path
            break

    if model_dir is None:
        print("Vosk model not found in the current directory.")
        sys.exit(1)
    else:
        model_path = model_dir

    app = QApplication(sys.argv)

    # Initialize and start the worker thread
    worker = Worker(model_path)
    window = MainWindow(worker)  # Pass worker to MainWindow

    worker.transcribed_text_signal.connect(window.update_transcribed_text)
    worker.diagnoses_signal.connect(window.update_diagnoses)
    worker.clinical_questions_signal.connect(window.update_clinical_questions)
    worker.layman_summary_signal.connect(window.update_layman_summary)
    worker.final_summary_signal.connect(window.show_final_summary) 
    worker.start()

    window.show()

    def on_app_quit():
        worker.stop()
        worker.wait()

    app.aboutToQuit.connect(on_app_quit)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
