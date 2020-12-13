import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import gtts
from gtts import gTTS
from playsound import playsound
import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "AplhaBot"
phoneNumber = 0
print("Let's chat! Type 'quit' to exit")

while True:
    sentence = input('You: ')
    if sentence == 'quit' or sentence == 'exit':
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    # print(f"Phone Number: {phoneNumber}")

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    # print(tag, prob)

    dummy_data = [
        "No balance dues!",
        "Balance due is Rs.432",
        "Balance overdue of Rs. 1253"
      ]

    # Text to Speech
    language = 'en'
    res = ''

    if prob.item() > 0.80:
        if (tag == 'purpose' and phoneNumber != 0):
            print(f"{bot_name}: {random.choice(dummy_data)}")

        elif (tag == 'purpose' and phoneNumber == 0):
            phoneNumber = int(input('Enter the mobile number: '))
            print(f"{bot_name}:Thanks, fetching details for you!")
            print(f"{bot_name}: {random.choice(dummy_data)}")


        else:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    res = random.choice(intent['responses'])

                    # Audio part
                    # myobj = gtts.gTTS(res, slow=False)
                    # myobj.save("audio.mp3")
                    # playsound("audio.mp3")

                    print(f"{bot_name}: {res}")
                # print(prob.item(), tag , intent["tag"])

    else:
        # myobj = gtts.gTTS("I do not understand", slow=False)
        # myobj.save("audio.mp3")
        # playsound("audio.mp3")
        print(f"{bot_name}: I do not understand....")
        # print(prob.item(), tag, intent["tag"])


