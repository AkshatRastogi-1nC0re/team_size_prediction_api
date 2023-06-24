import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import math

app = FastAPI()

class ScoringItem(BaseModel):
    TES: int
    TIS: int
    CLE: int
    PSS: int
    PCS: int
    HC: int
    RAS: int
    Class: int

with open('best_model_team_rf.pkl', 'rb') as file:
    model = pickle.load(file)

@app.post("/")
async def scoring_endpoint(item: ScoringItem):
    print(list(item.dict().values()))
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    print(df)
    yhat = model.predict(df)
    return {"prediction" : str(int(math.ceil(yhat)))}