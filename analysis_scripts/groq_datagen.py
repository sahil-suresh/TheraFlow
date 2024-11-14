import pandas as pd
import re
from datetime import datetime, timedelta
import os
from groq import Groq  # Assuming Groq client is correctly set up

# Set up API key in environment variable for Groq
os.environ['GROQ_API_KEY'] = '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'
client = Groq()

dialogue1 = """[doctor] hi , martha . how are you ? [patient] i'm doing okay . how are you ? [doctor] i'm doing okay . so , i know the nurse told you about dax . i'd like to tell dax a little bit about you , okay ?[patient] okay . [doctor] martha is a 50-year-old female with a past medical history significant for congestive heart failure , depression and hypertension who presents for her annual exam . so , martha , it's been a year since i've seen you . how are you doing ? [patient] i'm doing well . i've been traveling a lot recently since things have , have gotten a bit lighter . and i got my , my vaccine , so i feel safer about traveling . i've been doing a lot of hiking . uh , went to washington last weekend to hike in northern cascades, like around the mount baker area . [doctor] nice . that's great . i'm glad to hear that you're staying active , you know . i , i just love this weather . i'm so happy the summer is over . i'm definitely more of a fall person . [patient] yes , fall foliage is the best . [doctor] yeah . um , so tell me , how are you doing with the congestive heart failure ? how are you doing watching your diet ? i know we've talked about watching a low sodium diet . are you doing okay with that ? [patient] i've been doing well with that . i resisted , as much , as i could , from the tater tots , you know , the soft pretzels , the salty foods that i , i love to eat . and i've been doing a really good job . [doctor] okay , all right . well , i'm glad to hear that . and you're taking your medication ? [patient] yes .
 [doctor] okay , good . and any symptoms like chest pains , shortness of breath , any swelling in your legs ?
 [patient] no , not that i've noticed .
 [doctor] okay , all right . and then in terms of your depression , i know that we tried to stay off of medication in the past because you're on medications for your other problems . how are you doing ? and i know that you enrolled into therapy . is that helping ? or-
 [patient] yeah , it's been helping a lot . i've been going every week , um , for the past year since my last annual exam . and that's been really helpful for me .
 [doctor] okay . so , no , no issues , no feelings of wanting to harm yourself or hurt others ?
 [patient] no , nothing like that .
 [doctor] okay , all right . and then in terms of your high blood pressure , i know that you and i have kind of battled in the past with you remembering to take some of your blood pressure medications . how are you doing with that ?
 [patient] i'm still forgetting to take my blood pressure medication . and i've noticed when work gets more stressful , my blood pressure goes up .
 [doctor] okay . and , and so how has work going for you ?
 [patient] it's been okay . it's been a lot of long hours , late nights . a lot of , um , you know , fiscal year end data that i've been having to pull . so , a lot of responsibility , which is good . but with the responsibility comes the stress .
 [doctor] yeah , okay , all right . i understand . um , all right . well , i know that you did a review of system sheet when you checked in with the nurse . i know that you were endorsing some nasal congestion from some of the fall pollen and allergies . any other symptoms , nausea or vomiting , abdominal pain , anything like that ?
 [patient] no , nothing like that .
 [doctor] no , okay , all right . well , i'm gon na go ahead and do a quick physical exam , okay ?
 [patient] okay .
 [doctor] hey , dragon , show me the blood pressure . so , yeah , looking at your blood pressure today here in the office , it is a little elevated . you know , it could just , you could just be nervous . uh , let's look at some of the past readings . hey , dragon , show me the blood pressure readings . hey , dragon , show me the blood pressure readings . here we go . uh , so they are running on the higher side . um , y- you know , i , i do think that , you know , i'd like to see you take your medication a little bit more , so that we can get that under control a little bit better , okay ?
 [patient] okay .
 [doctor] so , i'm just gon na check out your heart and your lungs . and you know , let you know what i find , okay ?
 [patient] okay .
 [doctor] okay . so , on your physical examination , you know , everything looks good . on your heart exam , i do appreciate a three out of six systolic ejection murmur , which i've heard in the past , okay ? and on your lower extremities , i do appreciate one plus pitting edema , so you do have a little bit of fluid in your legs , okay ?
 [patient] okay .
 [doctor] let's go ahead , i wan na look at some of your results , okay ? hey , dragon , show me the echocardiogram . so , this is the result of the echocardiogram that we did last year . it showed that you have that low-ish pumping function of your heart at about 45 % . and it also sh- shows some mitral regurgitation , that's that heart murmur that i heard , okay ?
 [doctor] um , hey , dragon , show me the lipid panel . so , looking at your lipid panel from last year , you know , everything , your cholesterol was like , a tiny bit high . but it was n't too , too bad , so i know you're trying to watch your diet . so , we'll repeat another one this year , okay ?
 [patient] okay .
 [doctor] um , so i wan na just go over a little bit about my assessment and my plan for you , okay ? so , for your first problem your congestive heart failure , um , i wan na continue you on your current medications . but i do wan na increase your lisinopril to 40 milligrams a day , just because your blood pressure's high . and you know , you are retaining a little bit of fluid . i also wan na start you on some lasix , you know , 20 milligrams a day . and have you continue to watch your , your diet , okay ?
 [patient] okay .
 [doctor] i also wan na repeat another echocardiogram , okay ?
 [patient] all right .
 [doctor] hey , dragon , order an echocardiogram . from a depression standpoint , it sounds like you're doing really well with that . so , i'm , i'm really happy for you . i'm , i'm glad to see that you're in therapy and you're doing really well . i do n't feel the need to start you on any medications this year , unless you feel differently .
 [patient] no , i feel the same way .
 [doctor] okay , all right . and then for your last problem your hypertension , you know , again i , i , i think it's out of control . but we'll see , i think , you know , i'd like to see you take the lisinopril as directed , okay ? uh , i want you to record your blood pressures within the patient , you know , take your blood pressure every day . record them to me for like , about a week , so i have to see if we have to add another agent , okay ? 'cause we need to get that under better control for your heart failure to be more successful , okay ?
 [patient] okay .
 [doctor] do you have any questions ? , and i forgot . for your annual exam , you're due for a mammogram , so we have to schedule for that , as well , okay ?
 [patient] okay .
 [doctor] okay . do you have any questions ?
 [patient] can i take all my pills at the same time ?
 [doctor] yeah .
 [patient] 'cause i've been trying to take them at different times of the day , 'cause i did n't know if it was bad to take them all at once or i should separate them . i do n't know .
 [doctor] yeah . you can certainly take them , you know , all at the same time , as long , as yeah , they're all one scale . you can take them all at the same time . just set an alarm-
 [patient] okay .
 [doctor] . some time during the day to take them , okay ?
 [patient] that might help me remember better .
 [doctor] all right . that sounds good . all right , well , it's good to see you .
 [patient] good seeing you too .
 [doctor] hey , dragon , finalize the note ."""

# Remove role tags from the dialogue
cleaned_dialogue = re.sub(r'\[.*?\]', '', dialogue1).strip()

# Define the speaking rate: average words per 15 seconds
WORDS_PER_30_SECONDS = 60  # Adjust as needed for accuracy

# Split cleaned dialogue by words
words = cleaned_dialogue.split()

# Initialize an empty list to store results as dictionaries
data = []

# Set the initial timestamp
initial_timestamp = datetime(2023, 1, 1, 0, 0, 0)  # Arbitrary start time

# Function to get diagnoses, treatments, and questions for a given conversation
def get_diagnoses_and_treatments(conversation, client):
    # Initialize empty strings for each output
    diagnoses_list = ''
    treatment_plans = ''
    clinical_questions = ''

    # First API call to get differential diagnoses
    response_diagnoses = client.chat.completions.create(
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
    response_treatment = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a medical diagnostic assistant. Based on the patient's statements, provide only a comma-separated list of three accurate treatment plans, if it is a pharmacological treatment include the drug name and generic brand name, be as specific as possible and order results by most appropriate. There should not be any other text, headers, titles, or explanations besides the list."
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

    # Third API call to get clinical questions
    response_questions = client.chat.completions.create(
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

    # Return the three outputs
    return diagnoses_list, treatment_plans, clinical_questions

# Loop over the words list to build segments by word count (30 words per segment)
growing_conversation = ""
for i in range(0, len(words), WORDS_PER_30_SECONDS):
    # Construct the current segment of conversation up to the current point
    segment = ' '.join(words[:i + WORDS_PER_30_SECONDS])
    added_segment = ' '.join(words[i:i + WORDS_PER_30_SECONDS])
    
    # Calculate the current timestamp based on the segment index
    current_timestamp = initial_timestamp + timedelta(seconds=30 * (i // WORDS_PER_30_SECONDS))
    
    # Get diagnoses, treatments, and questions for the current state of conversation
    diagnoses, treatment, questions = get_diagnoses_and_treatments(segment.strip(), client)
    
    # Append results as a dictionary to the data list
    data.append({
        "Timestamp": current_timestamp.strftime("%H:%M:%S"),
        "Conversation": added_segment.strip(),
        "Diagnoses": diagnoses,
        "Treatment Plans": treatment,
        "Clinical Questions": questions
    })

# Convert list of dictionaries to a DataFrame
df = pd.DataFrame(data)
# Final API call to get a summary of the full conversation
response_summary = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a medical assistant summarizing the conversation. Extract the following information as a comma-separated list in this order: Patient Name, Current Diagnoses and diagnostic history, Treatment/Plan/Next steps, and Current Medications. There should be no other text, headers, or explanations besides the list."
        },
        {
            "role": "user",
            "content": cleaned_dialogue  # Full conversation
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

df.loc[0, "Full Conversation Summary"] = summary


# Save the DataFrame to a CSV file with a datetime-stamped filename
now = datetime.now()
datetime_string = now.strftime("%Y-%m-%d %H-%M-%S")
filename = f"growing_conversation_output_{datetime_string}.csv"
df.to_csv(filename, index=False)

print(f"File saved as {filename}")
