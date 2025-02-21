import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from diffusers import StableDiffusionPipeline
import torch

# Create the main application window
app = ctk.CTk()  
app.geometry("532x632")
app.title("Text to Image")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), fg_color="black", placeholder_text="Enter your prompt here")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

# Model setup (move to GPU if available, otherwise to CPU)
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(modelid)

if device == "cuda":
    pipe.to(device)

def generate():
    prompt_text = prompt.get()

   
    image = pipe(prompt_text, guidance_scale=8.5).images[0]

    
    image = image.convert("RGB")

    # Save and display the image
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)
    lmain.image = img 

# Create the CTkButton widget for generating the image
trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 20), text_color="black", fg_color="blue", command=generate)
trigger.place(x=206, y=60)

# Run the application
app.mainloop()