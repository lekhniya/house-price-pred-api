import io
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException,UploadFile,File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel ,Field

app=FastAPI()

model=joblib.load("house_model.joblib")
features=joblib.load("house_features.joblib")


#input Schema
class HouseFeatures(BaseModel):
    MedInc: float =Field(gt=0,description="Media Income of the Neighbourhood")
    HouseAge : float=Field(gt=0,description="Average age of houses in the block")
    AveRooms : float = Field(gt=0,description="Average number of Rooms")
    AveBedrms : float =Field(gt=0,description="Average number of Bedrooms")
    Population : float = Field(gt=0, description = "Total population ")
    AveOccup : float=Field(gt=0,description="Average number of Occupation") 
    Latitude : float = Field(gt=32,le=42,description="Latitude")
    Longitude : float = Field(ge=-125,le=-114,description = "Longitude")
    
    
    
#home
@app.get("/")
def home():
    return{"message":"California House prediction api",
           "status":"running",
           "endpoint":"send POST request to /predict"
        }
@app.get("/health")
def health():
    return {
        "status":"running",
        "model":"RandomForestRegressor",
        "features":features,
        "avg_error":"$39,000"
    }
    
#prediction
@app.post('/predict')
def predict(house:HouseFeatures):
    try:
        input_data=pd.DataFrame([{
            "MedInc" : house.MedInc,
            "HouseAge":house.HouseAge,
            "AveRooms":house.AveRooms,
            "AveBedrms":house.AveBedrms,
            "Population":house.Population,
            "AveOccup":house.AveOccup,
            "Latitude":house.Latitude,
            "Longitude":house.Longitude
        }])
        
        predicted=model.predict(input_data)[0]
        price_usd = predicted*100000
        return{
           "predicted_price":f"${price_usd:,.0f}",
           "predicted_price_short":f'${predicted:,.2f} hundred thousands',
           "fidance_range":f"${price_usd - 39000:,.0f} to ${price_usd + 39000:,.0f}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed : {str(e)}"
        )


@app.post("/predict-file")
async def predict_file(file:UploadFile=File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="please upload a CSV File only"
        )
    contents = await file.read()
    #b'name,age'\nSagar,25\nRahul
    df=pd.read_csv(io.BytesIO(contents))
    
    required_columns=[
        'MedInc','HouseAge','AveRooms','AveBedrms','Population'
        ,'AveOccup','Latitude','Longitude'
    ]
    missing_columns=[col for col in required_columns
                     if col not in df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f'These columns are missing from your file{missing_columns}'
        )
    
    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail='The uploaded file has no data rows'
        )
        
    try:
        predictions=model.predict(df[required_columns])
        df["predicted_price_usd"] = predictions * 100000
        df['predicted_price_usd']=df["predicted_price_usd"].apply(lambda x:f"${x:,.0f}")  
        output=df.to_csv(index=False)
        
        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "content-Disposition":"attachment; filename=prediction.csv"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'prediction failed : {str(e)}'
        )