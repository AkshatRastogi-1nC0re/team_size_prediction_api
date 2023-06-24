import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

class ScoringItem(BaseModel):
    Class: int
    TES: int
    TIS: int
    CLE: int
    PSS: int
    PCS: int
    HC: int
    RAS: int

with open('best_model_team_rf.pkl', 'rb') as file:
    model = pickle.load(file)

@app.post("/")
async def scoring_endpoint(item: ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {"prediction" : str(yhat)}