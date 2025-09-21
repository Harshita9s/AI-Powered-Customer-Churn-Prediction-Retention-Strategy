from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import shap
import json
import aiofiles
from emergentintegrations.llm.chat import LlmChat, UserMessage
import asyncio
import io

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global variables to store model and data
current_model = None
current_data = None
current_shap_explainer = None
feature_importance = None

# Define Models
class ChurnDataUpload(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    data_summary: Dict[str, Any]
    
class ModelTrainingResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_type: str
    accuracy: float
    precision: float
    recall: float
    auc: float
    feature_importance: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str  # Low, Medium, High
    top_factors: List[Dict[str, Any]]

class RecommendationRequest(BaseModel):
    model_summary: Dict[str, Any]
    top_risk_customers: List[Dict[str, Any]]

class RetentionStrategy(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategies: List[Dict[str, str]]
    executive_summary: str
    expected_impact: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Utility functions
def prepare_sample_data():
    """Create sample customer churn data for demo purposes"""
    np.random.seed(42)
    n_customers = 1000
    
    data = {
        'customer_id': [f'CUST_{i:04d}' for i in range(1, n_customers + 1)],
        'tenure': np.random.normal(24, 12, n_customers).clip(1, 60),
        'monthly_charges': np.random.normal(65, 20, n_customers).clip(20, 120),
        'total_charges': [],
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.4, 0.4, 0.2]),
        'tech_support': np.random.choice(['Yes', 'No'], n_customers),
        'online_security': np.random.choice(['Yes', 'No'], n_customers),
        'device_protection': np.random.choice(['Yes', 'No'], n_customers),
        'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]),
        'partner': np.random.choice(['Yes', 'No'], n_customers),
        'dependents': np.random.choice(['Yes', 'No'], n_customers),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_customers),
        'multiple_lines': np.random.choice(['Yes', 'No'], n_customers),
        'churn': []
    }
    
    # Calculate total charges based on tenure and monthly charges
    for i in range(n_customers):
        total_charge = data['tenure'][i] * data['monthly_charges'][i] + np.random.normal(0, 100)
        data['total_charges'].append(max(0, total_charge))
    
    # Generate churn based on logical patterns
    for i in range(n_customers):
        churn_probability = 0.1  # Base probability
        
        # Higher churn for month-to-month contracts
        if data['contract_type'][i] == 'Month-to-month':
            churn_probability += 0.3
        elif data['contract_type'][i] == 'One year':
            churn_probability += 0.1
            
        # Higher churn for electronic check payments
        if data['payment_method'][i] == 'Electronic check':
            churn_probability += 0.15
            
        # Higher churn for high monthly charges
        if data['monthly_charges'][i] > 80:
            churn_probability += 0.2
            
        # Lower churn for customers with tech support
        if data['tech_support'][i] == 'Yes':
            churn_probability -= 0.1
            
        # Lower churn for customers with partners/dependents
        if data['partner'][i] == 'Yes' or data['dependents'][i] == 'Yes':
            churn_probability -= 0.05
            
        # Higher churn for short tenure
        if data['tenure'][i] < 12:
            churn_probability += 0.2
            
        churn_probability = max(0.05, min(0.8, churn_probability))
        data['churn'].append(1 if np.random.random() < churn_probability else 0)
    
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    # Create a copy to avoid modifying original
    processed_df = df.copy()
    
    # Handle categorical variables
    categorical_columns = ['contract_type', 'payment_method', 'internet_service', 
                          'tech_support', 'online_security', 'device_protection',
                          'partner', 'dependents', 'paperless_billing', 'multiple_lines']
    
    for col in categorical_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].map({'Yes': 1, 'No': 0}) if processed_df[col].dtype == 'object' and set(processed_df[col].unique()).issubset({'Yes', 'No', None}) else processed_df[col]
    
    # One-hot encode remaining categorical variables
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
    
    return processed_df

async def generate_recommendations(model_summary: Dict, top_risk_customers: List[Dict]) -> str:
    """Generate AI-powered retention recommendations using GPT-5"""
    try:
        # Initialize the LLM chat
        chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=f"churn_analysis_{uuid.uuid4()}",
            system_message="""You are a senior McKinsey consultant specializing in customer retention strategy. 
            You analyze churn prediction models and create actionable, executive-level retention recommendations.
            Your responses should be professional, data-driven, and focused on ROI."""
        ).with_model("openai", "gpt-5")
        
        prompt = f"""
        CHURN ANALYSIS BRIEF:
        
        MODEL PERFORMANCE:
        - Accuracy: {model_summary.get('accuracy', 0):.2%}
        - Precision: {model_summary.get('precision', 0):.2%} 
        - Recall: {model_summary.get('recall', 0):.2%}
        - AUC: {model_summary.get('auc', 0):.2%}
        
        TOP CHURN DRIVERS:
        {json.dumps(model_summary.get('feature_importance', {}), indent=2)}
        
        HIGH-RISK CUSTOMER PROFILE:
        {json.dumps(top_risk_customers[:3], indent=2)}
        
        DELIVERABLE REQUEST:
        Create an executive summary with 3 targeted retention strategies. For each strategy:
        1. Strategic rationale (why this approach)
        2. Target customer segment 
        3. Specific tactical implementation
        4. Expected impact estimate
        5. Resource requirements
        
        Format as professional consulting brief, suitable for C-suite presentation.
        """
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        return response
        
    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")
        return f"Error generating recommendations: {str(e)}"

# API Routes
@api_router.get("/")
async def root():
    return {"message": "AI-Powered Customer Churn Prediction API"}

@api_router.get("/sample-data")
async def get_sample_data():
    """Generate and return sample customer data"""
    try:
        sample_df = prepare_sample_data()
        
        # Convert to JSON-serializable format
        sample_data = sample_df.to_dict(orient='records')
        
        # Store in global variable for use
        global current_data
        current_data = sample_df
        
        # Generate data summary
        summary = {
            'total_customers': len(sample_df),
            'churn_rate': sample_df['churn'].mean(),
            'avg_tenure': sample_df['tenure'].mean(),
            'avg_monthly_charges': sample_df['monthly_charges'].mean(),
            'columns': list(sample_df.columns)
        }
        
        return {
            'data': sample_data[:100],  # Return first 100 rows for preview
            'summary': summary,
            'total_rows': len(sample_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating sample data: {str(e)}")

@api_router.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """Upload CSV file for churn analysis"""
    try:
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Store in global variable
        global current_data
        current_data = df
        
        # Generate data summary
        summary = {
            'total_customers': len(df),
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        if 'churn' in df.columns:
            summary['churn_rate'] = df['churn'].mean() if df['churn'].dtype in ['int64', 'float64'] else df['churn'].value_counts(normalize=True).to_dict()
        
        # Save upload record to database
        upload_record = ChurnDataUpload(
            filename=file.filename,
            data_summary=summary
        )
        
        await db.data_uploads.insert_one(upload_record.dict())
        
        return {
            'message': 'Data uploaded successfully',
            'summary': summary,
            'upload_id': upload_record.id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading data: {str(e)}")

@api_router.post("/train-model")
async def train_model():
    """Train churn prediction model"""
    try:
        global current_model, current_data, current_shap_explainer, feature_importance
        
        if current_data is None:
            raise HTTPException(status_code=400, detail="No data available. Please upload data or use sample data first.")
        
        # Preprocess the data
        processed_df = preprocess_data(current_data)
        
        # Separate features and target
        if 'churn' not in processed_df.columns:
            raise HTTPException(status_code=400, detail="Target column 'churn' not found in data")
        
        X = processed_df.drop(['churn', 'customer_id'], axis=1, errors='ignore')
        y = processed_df['churn']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred))
        recall = float(recall_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_pred_proba))
        
        # Get feature importance
        feature_names = X.columns.tolist()
        importance_scores = model.feature_importances_
        feature_importance = dict(zip(feature_names, [float(score) for score in importance_scores]))
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Initialize SHAP explainer
        current_shap_explainer = shap.TreeExplainer(model)
        
        # Store the trained model
        current_model = model
        
        # Save training results to database
        training_result = ModelTrainingResult(
            model_type="XGBoost",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            auc=auc,
            feature_importance=feature_importance
        )
        
        await db.training_results.insert_one(training_result.dict())
        
        return {
            'message': 'Model trained successfully',
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'auc': auc
            },
            'feature_importance': feature_importance,
            'training_id': training_result.id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@api_router.get("/predictions")
async def get_predictions():
    """Get churn predictions for all customers"""
    try:
        global current_model, current_data, current_shap_explainer
        
        if current_model is None or current_data is None:
            raise HTTPException(status_code=400, detail="No trained model available. Please train model first.")
        
        # Preprocess the data
        processed_df = preprocess_data(current_data)
        
        # Prepare features
        feature_cols = [col for col in processed_df.columns if col not in ['churn', 'customer_id']]
        X = processed_df[feature_cols]
        
        # Make predictions
        predictions = current_model.predict_proba(X)[:, 1]
        
        # Calculate SHAP values for explanations
        shap_values = current_shap_explainer.shap_values(X)
        
        # Create prediction results
        results = []
        for i, (idx, row) in enumerate(processed_df.iterrows()):
            customer_id = row.get('customer_id', f'Customer_{i}')
            churn_prob = predictions[i]
            
            # Determine risk level
            if churn_prob >= 0.7:
                risk_level = "High"
            elif churn_prob >= 0.4:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            # Get top factors (SHAP values)
            customer_shap = shap_values[i]
            feature_contributions = list(zip(feature_cols, customer_shap))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_factors = []
            for feature, contribution in feature_contributions[:3]:
                factor_impact = "Increases" if contribution > 0 else "Decreases"
                top_factors.append({
                    'feature': feature,
                    'impact': factor_impact,
                    'value': float(abs(contribution)),
                    'current_value': row.get(feature, 'N/A')
                })
            
            results.append({
                'customer_id': customer_id,
                'churn_probability': float(churn_prob),
                'risk_level': risk_level,
                'top_factors': top_factors
            })
        
        # Sort by churn probability (highest first)
        results.sort(key=lambda x: x['churn_probability'], reverse=True)
        
        return {
            'predictions': results,
            'summary': {
                'total_customers': len(results),
                'high_risk': sum(1 for r in results if r['risk_level'] == 'High'),
                'medium_risk': sum(1 for r in results if r['risk_level'] == 'Medium'),
                'low_risk': sum(1 for r in results if r['risk_level'] == 'Low'),
                'avg_churn_probability': float(sum(r['churn_probability'] for r in results) / len(results))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")

@api_router.post("/generate-recommendations")
async def generate_retention_recommendations():
    """Generate AI-powered retention recommendations"""
    try:
        global current_model, feature_importance
        
        if current_model is None:
            raise HTTPException(status_code=400, detail="No trained model available. Please train model first.")
        
        # Get model performance metrics (from previous training)
        training_results = await db.training_results.find().sort([("timestamp", -1)]).limit(1).to_list(1)
        if not training_results:
            raise HTTPException(status_code=400, detail="No training results found.")
        
        latest_result = training_results[0]
        
        # Get top risk customers
        predictions_response = await get_predictions()
        top_risk_customers = predictions_response['predictions'][:5]  # Top 5 risk customers
        
        # Generate recommendations using AI
        model_summary = {
            'accuracy': latest_result['accuracy'],
            'precision': latest_result['precision'],
            'recall': latest_result['recall'],
            'auc': latest_result['auc'],
            'feature_importance': latest_result['feature_importance']
        }
        
        ai_recommendations = await generate_recommendations(model_summary, top_risk_customers)
        
        # Save recommendations to database
        retention_strategy = RetentionStrategy(
            strategies=[],  # Will be parsed from AI response
            executive_summary=ai_recommendations,
            expected_impact="Based on model predictions and strategic interventions"
        )
        
        await db.retention_strategies.insert_one(retention_strategy.dict())
        
        return {
            'recommendations': ai_recommendations,
            'model_performance': model_summary,
            'top_risk_customers': top_risk_customers,
            'strategy_id': retention_strategy.id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@api_router.get("/dashboard-summary")
async def get_dashboard_summary():
    """Get summary statistics for dashboard"""
    try:
        global current_data
        
        if current_data is None:
            # Return sample data summary
            sample_df = prepare_sample_data()
            current_data = sample_df
        
        # Basic statistics
        total_customers = len(current_data)
        churn_rate = current_data['churn'].mean() if 'churn' in current_data.columns else 0
        
        # Get latest training results
        training_results = await db.training_results.find().sort([("timestamp", -1)]).limit(1).to_list(1)
        model_performance = None
        if training_results:
            result = training_results[0]
            model_performance = {
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'auc': result['auc']
            }
        
        # Get risk distribution
        if current_model is not None:
            predictions_data = await get_predictions()
            risk_distribution = {
                'high_risk': predictions_data['summary']['high_risk'],
                'medium_risk': predictions_data['summary']['medium_risk'],
                'low_risk': predictions_data['summary']['low_risk']
            }
        else:
            risk_distribution = {'high_risk': 0, 'medium_risk': 0, 'low_risk': 0}
        
        return {
            'total_customers': total_customers,
            'churn_rate': churn_rate,
            'model_performance': model_performance,
            'risk_distribution': risk_distribution,
            'feature_importance': feature_importance or {},
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating dashboard summary: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
