from google import genai

client = genai.Client(api_key="AIzaSyBH2NUrsP1S2RCv7odQzHikeDdHZhk3Ttk")

print("--- AVAILABLE MODELS ---")
for model in client.models.list():
    print(f"Name: {model.name} | Actions: {model.supported_actions}")