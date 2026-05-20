# Fast API
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates # UI 
from fastapi.responses import HTMLResponse 
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI app
app = FastAPI(title = "Text Summarizer App", description="An API to summarize text using T5 model", version="1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load T5 model and tokenizer
MODEL_NAME = "vaishnavirathi/SummarixAI"
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Device
if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

model.to(device)

# Templating
templates = Jinja2Templates(directory="templates")

# Input Schema for dialogue => string
class DialogueInput(BaseModel):
    dialogue: str

# Cleaning Data
def clean_data(text):
  text = re.sub(r"\r\n", " ", text)  # Remove Lines
  text = re.sub(r"\s+", " ", text)  # Remove Spaces
  text = re.sub(r"<.*?>", " ", text)  # Remove HTMLTags <p> <h1>
  text = text.strip().lower()
  return text

# Summarization
def summarize_dialogue(dialogue : str) -> str:
  dialogue = clean_data(dialogue) # Clean

  # Tokenize
  inputs = tokenizer(
      dialogue,
      padding="max_length",
      max_length=512,
      truncation=True,
      return_tensors="pt"
  ).to(device)

  # Generate the summary => token ids
  model.to(device)
  targets = model.generate(
      input_ids=inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      max_length=150,
      num_beams=4,
      early_stopping=True
  )

  # token ids convert to text/summary => decoding
  # EOS → End of Sentence (sometimes also used as End of Sequence in NLP models)
  # SEP → Separator
  summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
  return summary

# API Endpoint
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

from fastapi import Request

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )




