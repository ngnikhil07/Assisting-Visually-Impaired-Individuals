import pyttsx3

engine= pyttsx3.init()
rate=engine.getProperty("rate")
voices=engine.getProperty("voices")

kick="Brother do you eat chicken" 
engine.setProperty("rate",120)
engine.setProperty("voice",voices[1].id)
engine.say(kick)

engine.runAndWait()