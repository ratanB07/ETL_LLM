from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def determine_problem_type(y):
    """Determine if classification or regression based on target variable"""
    unique_values = y.nunique()
    if unique_values < 10 or y.dtype == 'object':
        return 'classification'
    return 'regression'

def create_feature_importance_plot(model, feature_names):
    """Create and return base64 encoded feature importance plot"""
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title("Feature Importances")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('ascii')
    plt.close()
    return plot_data


@app.route('/deploy_to_alteryx')
def deploy_to_alteryx():
    return send_file('deploy_to_alteryx.html')
@app.route('/ai_catalog/search', methods=['POST'])
def ai_catalog_search():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        # In a real implementation, this would query your LLM and data catalog
        # For demo purposes, we'll return simulated results
        
        # Simulate processing delay
        time.sleep(1)
        
        # Generate simulated response
        response = {
            'status': 'success',
            'query': query,
            'results': [
                {
                    'name': 'customer_transactions',
                    'description': 'Contains customer transaction records with purchase amounts, dates, and product categories.',
                    'source': 'PostgreSQL',
                    'tags': ['customers', 'transactions', 'sales'],
                    'quality_score': 0.95,
                    'last_updated': '2023-07-15T14:30:00Z'
                },
                {
                    'name': 'product_inventory',
                    'description': 'Current inventory levels across all warehouses with product SKUs and locations.',
                    'source': 'Excel',
                    'tags': ['products', 'inventory', 'supply-chain'],
                    'quality_score': 0.82,
                    'last_updated': '2023-07-14T09:15:00Z'
                }
            ],
            'semantic_relationships': [
                {
                    'source': 'customer_transactions',
                    'target': 'product_inventory',
                    'relationship': 'transactions reference products in inventory',
                    'strength': 0.75
                }
            ],
            'ai_insights': [
                'The customer_transactions dataset shows strong seasonal patterns in Q4.',
                'Inventory data has some inconsistencies that should be reconciled.'
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/ai_catalog/ask', methods=['POST'])
def ai_catalog_ask():
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        # Simulate LLM processing
        time.sleep(2)
        
        # Generate context-aware response
        response = {
            'status': 'success',
            'question': question,
            'answer': "Based on your question about customer data, I've identified that the customer_transactions dataset would be most relevant. It contains purchase history that can be analyzed for patterns and trends.",
            'confidence': 0.88,
            'suggested_datasets': ['customer_transactions', 'customer_demographics'],
            'follow_up_questions': [
                "Would you like to see customer segmentation analysis?",
                "Are you interested in time-based purchase patterns?"
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        target_column = data['target_column']
        
        # Preprocess data
        df = df.dropna()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Encode categorical features
        for col in categorical_cols:
            if col != target_column:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
        # Prepare target
        y = df[target_column]
        problem_type = determine_problem_type(y)
        
        if problem_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X = df.drop(columns=[target_column])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train model
        if problem_type == 'classification':
            model = RandomForestClassifier(random_state=42)
        else:
            model = RandomForestRegressor(random_state=42)
            
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        if problem_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
        else:
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        
        # Create feature importance plot
        feature_importance = create_feature_importance_plot(model, X.columns)
        
        return jsonify({
            'status': 'success',
            'problem_type': problem_type,
            'target_column': target_column,
            'feature_importance': feature_importance,
            **metrics
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(port=5001)