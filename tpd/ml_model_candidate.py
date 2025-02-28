import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import joblib
import pickle


class ITCandidateML:
    """
    ML-based IT candidate evaluation system that learns from historical data
    to predict role suitability for production support, maintenance, and new development.
    """
    
    def __init__(self, model_type='classification'):
        """
        Initialize the ML model for IT candidate evaluation.
        
        Args:
            model_type (str): Either 'classification' or 'regression'
        """
        self.model_type = model_type
        self.feature_columns = [
            'technical_knowledge', 'communication_skills', 'problem_solving',
            'team_collaboration', 'enterprise_experience', 'stress_management',
            'response_time', 'documentation_skills', 'system_knowledge',
            'innovation_capability', 'learning_agility', 'code_quality',
            'architecture_understanding'
        ]
        
        # For classification, define classes
        self.role_classes = ['Production Support', 'Maintenance', 'New Development']
        
        # For regression, define target columns
        self.target_columns = ['ps_score', 'm_score', 'nd_score']
        
        # Initialize preprocessing
        self.scaler = StandardScaler()
        
        # Initialize models
        if model_type == 'classification':
            # Model to predict the most suitable role
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            # Model to predict scores for all three roles
            base_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model = MultiOutputRegressor(base_regressor)
    
    def generate_synthetic_training_data(self, n_samples=1000):
        """
        Generate synthetic training data based on domain knowledge.
        In a real-world scenario, this would be replaced with historical data.
        
        Args:
            n_samples: Number of synthetic samples to generate
            
        Returns:
            DataFrame containing synthetic training data
        """
        np.random.seed(42)  # For reproducibility
        
        # Generate random scores for each parameter
        data = {}
        for feature in self.feature_columns:
            data[feature] = np.random.uniform(1, 10, n_samples)
        
        df = pd.DataFrame(data)
        
        # Calculate 'ground truth' scores based on our original weighted formulas
        # Production Support score
        df['ps_score'] = (
            0.15 * df['technical_knowledge'] +
            0.15 * df['communication_skills'] +
            0.15 * df['problem_solving'] +
            0.05 * df['team_collaboration'] +
            0.10 * df['enterprise_experience'] +
            0.15 * df['stress_management'] +
            0.15 * df['response_time'] +
            0.10 * df['system_knowledge']
        )
        
        # Maintenance score
        df['m_score'] = (
            0.15 * df['technical_knowledge'] +
            0.10 * df['communication_skills'] +
            0.15 * df['problem_solving'] +
            0.10 * df['team_collaboration'] +
            0.15 * df['enterprise_experience'] +
            0.05 * df['stress_management'] +
            0.10 * df['documentation_skills'] +
            0.20 * df['system_knowledge']
        )
        
        # New Development score
        df['nd_score'] = (
            0.15 * df['technical_knowledge'] +
            0.10 * df['communication_skills'] +
            0.15 * df['problem_solving'] +
            0.10 * df['team_collaboration'] +
            0.05 * df['enterprise_experience'] +
            0.15 * df['innovation_capability'] +
            0.10 * df['learning_agility'] +
            0.10 * df['code_quality'] +
            0.10 * df['architecture_understanding']
        )
        
        # Add some random noise to make it more realistic
        df['ps_score'] += np.random.normal(0, 0.2, n_samples)
        df['m_score'] += np.random.normal(0, 0.2, n_samples)
        df['nd_score'] += np.random.normal(0, 0.2, n_samples)
        
        # Clip scores to 0-10 range
        df['ps_score'] = np.clip(df['ps_score'], 0, 10)
        df['m_score'] = np.clip(df['m_score'], 0, 10)
        df['nd_score'] = np.clip(df['nd_score'], 0, 10)
        
        # For classification, determine the most suitable role
        df['best_role'] = df[['ps_score', 'm_score', 'nd_score']].idxmax(axis=1)
        df['best_role'] = df['best_role'].map({
            'ps_score': 'Production Support',
            'm_score': 'Maintenance', 
            'nd_score': 'New Development'
        })
        
        return df
    
    def preprocess_data(self, X):
        """Scale the features"""
        return self.scaler.transform(X)
    
    def train(self, data=None, hyperparameter_tuning=False):
        """
        Train the ML model using either provided data or synthetic data.
        
        Args:
            data (DataFrame, optional): Training data. If None, synthetic data is generated.
            hyperparameter_tuning (bool): Whether to perform grid search for optimal parameters
        """
        if data is None:
            data = self.generate_synthetic_training_data()
        
        # Extract features and targets
        X = data[self.feature_columns]
        
        if self.model_type == 'classification':
            y = data['best_role']
        else:
            y = data[self.target_columns]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning if requested
        if hyperparameter_tuning:
            self._perform_hyperparameter_tuning(X_train_scaled, y_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        if self.model_type == 'classification':
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            print(f"Model Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(report)
            
        else:  # Regression
            y_pred = self.model.predict(X_test_scaled)
            mse = [mean_squared_error(y_test[col], y_pred[:, i]) 
                  for i, col in enumerate(self.target_columns)]
            r2 = [r2_score(y_test[col], y_pred[:, i]) 
                 for i, col in enumerate(self.target_columns)]
            
            for i, role in enumerate(self.target_columns):
                print(f"{role} - MSE: {mse[i]:.4f}, R²: {r2[i]:.4f}")
        
        # Save feature importance
        if hasattr(self.model, 'feature_importances_'):
            self._analyze_feature_importance(X.columns)
        elif hasattr(self.model, 'estimators_'):
            for i, est in enumerate(self.model.estimators_):
                if hasattr(est, 'feature_importances_'):
                    print(f"\nFeature importance for {self.target_columns[i]}:")
                    self._analyze_feature_importance(X.columns, est.feature_importances_)
    
    def _perform_hyperparameter_tuning(self, X_train, y_train):
        """Perform grid search for optimal hyperparameters"""
        if self.model_type == 'classification':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='accuracy'
            )
        else:
            base_regressor = RandomForestRegressor(random_state=42)
            param_grid = {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': [None, 10, 20],
                'estimator__min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(
                MultiOutputRegressor(base_regressor),
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error'
            )
        
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        self.model = grid_search.best_estimator_
    
    def _analyze_feature_importance(self, feature_names, importances=None):
        """Analyze and visualize feature importance"""
        if importances is None:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            else:
                return
        
        # Create a dataframe of feature importances
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print("Feature Importance:")
        print(feature_importance)
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
    
    def predict_role(self, candidate_data):
        """
        Predict the most suitable role for a candidate.
        
        Args:
            candidate_data: Dictionary or DataFrame with candidate parameters
            
        Returns:
            Dictionary with prediction results
        """
        if isinstance(candidate_data, dict):
            # Convert single candidate dictionary to DataFrame
            df = pd.DataFrame([candidate_data])
        else:
            df = candidate_data.copy()
        
        # Ensure all required features are present
        for feature in self.feature_columns:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Extract features in the correct order
        X = df[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'classification':
            # Predict the most suitable role
            role_prediction = self.model.predict(X_scaled)
            role_probs = self.model.predict_proba(X_scaled)
            
            # Create result dictionary for each candidate
            results = []
            for i, prediction in enumerate(role_prediction):
                result = {
                    'predicted_role': prediction,
                    'confidence': {
                        self.role_classes[j]: prob 
                        for j, prob in enumerate(role_probs[i])
                    }
                }
                results.append(result)
            
        else:  # Regression
            # Predict scores for all three roles
            scores = self.model.predict(X_scaled)
            
            # Create result dictionary for each candidate
            results = []
            for i in range(len(X)):
                # Get predicted scores
                ps_score, m_score, nd_score = scores[i]
                
                # Determine the most suitable role based on highest score
                roles = ['Production Support', 'Maintenance', 'New Development']
                role_scores = [ps_score, m_score, nd_score]
                best_role_idx = np.argmax(role_scores)
                
                # Get fitness level
                def get_fitness_level(score):
                    if score >= 8.5:
                        return "Excellent fit"
                    elif score >= 7.0:
                        return "Strong fit"
                    elif score >= 5.5:
                        return "Moderate fit"
                    elif score >= 4.0:
                        return "Weak fit"
                    else:
                        return "Poor fit"
                
                result = {
                    'Production Support': {
                        'score': round(ps_score, 2),
                        'classification': get_fitness_level(ps_score)
                    },
                    'Maintenance': {
                        'score': round(m_score, 2),
                        'classification': get_fitness_level(m_score)
                    },
                    'New Development': {
                        'score': round(nd_score, 2),
                        'classification': get_fitness_level(nd_score)
                    },
                    'Most Suitable Role': roles[best_role_idx]
                }
                
                # Check for versatility
                max_score = max(role_scores)
                close_roles = [role for role, score in zip(roles, role_scores) 
                              if abs(score - max_score) <= 0.5]
                
                if len(close_roles) > 1:
                    result['Versatility'] = "Versatile: Suitable for multiple roles"
                else:
                    result['Versatility'] = f"Specialized: Best suited for {roles[best_role_idx]}"
                
                results.append(result)
        
        # If single candidate, return just the first result
        if len(results) == 1:
            return results[0]
        return results
    
    def visualize_prediction(self, candidate_name, prediction):
        """
        Generate a visualization of the prediction results.
        
        Args:
            candidate_name: Name of the candidate
            prediction: Prediction dictionary from predict_role method
            
        Returns:
            Matplotlib figure
        """
        if self.model_type == 'classification':
            # For classification, create a bar chart of confidence scores
            roles = list(prediction['confidence'].keys())
            confidence = list(prediction['confidence'].values())
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(roles, confidence, color=['#3498db', '#2ecc71', '#e74c3c'])
            
            # Highlight the predicted role
            predicted_idx = roles.index(prediction['predicted_role']) if prediction['predicted_role'] in roles else -1
            if predicted_idx >= 0:
                bars[predicted_idx].set_color('gold')
                bars[predicted_idx].set_edgecolor('black')
            
            plt.ylim(0, 1)
            plt.xlabel('IT Roles')
            plt.ylabel('Confidence Score (0-1)')
            plt.title(f'ML Prediction: Role Suitability for {candidate_name}')
            
            # Add score labels on top of bars
            for bar, score in zip(bars, confidence):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{score:.2f}", ha='center', va='bottom')
            
        else:  # Regression
            # For regression, create a bar chart of predicted scores
            roles = ["Production Support", "Maintenance", "New Development"]
            scores = [prediction[role]['score'] for role in roles]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(roles, scores, color=['#3498db', '#2ecc71', '#e74c3c'])
            
            # Highlight the predicted role
            predicted_idx = roles.index(prediction['Most Suitable Role'])
            bars[predicted_idx].set_color('gold')
            bars[predicted_idx].set_edgecolor('black')
            
            plt.ylim(0, 10)
            plt.xlabel('IT Roles')
            plt.ylabel('Predicted Suitability Score (0-10)')
            plt.title(f'ML Prediction: Role Suitability for {candidate_name}')
            
            # Add score labels on top of bars
            for bar, score in zip(bars, scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(score), ha='center', va='bottom')
                
            # Add classification labels
            for i, role in enumerate(roles):
                plt.text(i, scores[i] - 0.5, prediction[role]['classification'],
                        ha='center', va='bottom', color='white', fontweight='bold')
        
        plt.tight_layout()
        return plt
    
    def save_model(self, filename='it_candidate_ml_model'):
        """Save the trained model, scaler, and metadata"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'role_classes': self.role_classes,
            'target_columns': self.target_columns
        }
        
        with open(f"{filename}.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}.pkl")
    
    @classmethod
    def load_model(cls, filename='it_candidate_ml_model.pkl'):
        """Load a saved model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        instance = cls(model_type=model_data['model_type'])
        
        # Restore the model attributes
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_columns = model_data['feature_columns']
        instance.role_classes = model_data['role_classes']
        instance.target_columns = model_data['target_columns']
        
        return instance


# Example usage
if __name__ == "__main__":
    # Create and train a regression-based ML model
    print("Training regression model...")
    ml_regressor = ITCandidateML(model_type='regression')
    ml_regressor.train(hyperparameter_tuning=False)
    ml_regressor.save_model('it_candidate_regressor')
    
    # Create and train a classification-based ML model
    print("\nTraining classification model...")
    ml_classifier = ITCandidateML(model_type='classification')
    ml_classifier.train(hyperparameter_tuning=False)
    ml_classifier.save_model('it_candidate_classifier')
    
    # Example candidate: Jane Doe
    jane = {
        'technical_knowledge': 8.0,
        'communication_skills': 7.0,
        'problem_solving': 9.0,
        'team_collaboration': 8.0,
        'enterprise_experience': 6.0,
        'stress_management': 9.0,
        'response_time': 8.0,
        'documentation_skills': 7.0,
        'system_knowledge': 7.0,
        'innovation_capability': 9.0,
        'learning_agility': 8.0,
        'code_quality': 8.0,
        'architecture_understanding': 7.0
    }
    
    # Evaluate Jane with regression model
    jane_results_reg = ml_regressor.predict_role(jane)
    print("\nRegression Model Results for Jane Doe:")
    for role, data in jane_results_reg.items():
        if role not in ["Most Suitable Role", "Versatility"]:
            print(f"{role}: Score = {data['score']}, Classification = {data['classification']}")
    print(f"Most Suitable Role: {jane_results_reg['Most Suitable Role']}")
    print(f"Versatility: {jane_results_reg['Versatility']}")
    
    # Create visualization
    plt_reg = ml_regressor.visualize_prediction("Jane Doe", jane_results_reg)
    plt_reg.savefig('jane_ml_regression.png')
    
    # Evaluate Jane with classification model
    jane_results_cls = ml_classifier.predict_role(jane)
    print("\nClassification Model Results for Jane Doe:")
    print(f"Predicted Role: {jane_results_cls['predicted_role']}")
    print(f"Confidence Scores: {jane_results_cls['confidence']}")
    
    # Create visualization
    plt_cls = ml_classifier.visualize_prediction("Jane Doe", jane_results_cls)
    plt_cls.savefig('jane_ml_classification.png')
    
    # Example: Batch prediction with multiple candidates
    candidates = pd.DataFrame([
        {
            'name': 'Jane Doe',
            'technical_knowledge': 8.0,
            'communication_skills': 7.0,
            'problem_solving': 9.0,
            'team_collaboration': 8.0,
            'enterprise_experience': 6.0,
            'stress_management': 9.0,
            'response_time': 8.0,
            'documentation_skills': 7.0,
            'system_knowledge': 7.0,
            'innovation_capability': 9.0,
            'learning_agility': 8.0,
            'code_quality': 8.0,
            'architecture_understanding': 7.0
        },
        {
            'name': 'John Smith',
            'technical_knowledge': 7.0,
            'communication_skills': 9.0,
            'problem_solving': 7.0,
            'team_collaboration': 9.0,
            'enterprise_experience': 8.0,
            'stress_management': 8.0,
            'response_time': 9.0,
            'documentation_skills': 8.0,
            'system_knowledge': 9.0,
            'innovation_capability': 6.0,
            'learning_agility': 7.0,
            'code_quality': 7.0,
            'architecture_understanding': 6.0
        }
    ])
    
    # Get feature columns only
    candidate_features = candidates.drop('name', axis=1)
    
    # Make predictions
    batch_results = ml_regressor.predict_role(candidate_features)
    
    print("\nBatch Prediction Results:")
    for i, name in enumerate(candidates['name']):
        print(f"\n{name} - Most Suitable Role: {batch_results[i]['Most Suitable Role']}")
        for role in ["Production Support", "Maintenance", "New Development"]:
            print(f"  {role}: {batch_results[i][role]['score']}")