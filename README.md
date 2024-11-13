# TheraFlow
## A real-time clinical decision assistant

TheraFlow aims to enhance clinical decision-making and patient communication by providing real-time support to clinicians during patient encounters. It addresses the limitations of traditional Point-of-Care (POC) tools by using artificial intelligence (AI) to automate the process of generating differential diagnoses, treatment plans, and clinical questions without requiring manual input from the clinician. The application achieves this through a combination of speech recognition, natural language processing, and a user-friendly graphical interface.

Upon starting TheraFlow, the application listens to and transcribes the patient encounter. Periodically, the program provides real-time updates to the text transcript of the conversation and additionally prompts a large language model (LLM) to provide the top diagnoses, treatments, and next questions to ask the patient — all based on the current conversation history. To end the conversation, press "Control" or "Command" (on Macs). At this point, the application prints a summary of key points from the encounter.

### Usage
To install the required dependencies, navigate to the TheraFlow directory and run the following command:
```
pip install -r requirements.txt
```

Then, to start the application, run the follwing command:
```
python DatathonApp.py
```
To end the conversation, press "Control" or "Command" (on Macs).

<br/>

***Note:*** Some users, especially those using a Mac M1 Pro, may additionally need to import the ```sounddevice``` library prior to importing ```pyaudio``` at the beginning of ```DatathonApp.py```:
```
# other dependencies...

import sounddevice
import pyaudio

# more dependencies...
```
