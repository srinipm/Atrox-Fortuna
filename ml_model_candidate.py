import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class TechRoleAptitudeModel:
    """
    A model to predict whether an individual is more suited to be a developer, tester,
    release manager, DevOps engineer, or PMO based on various cognitive, personality, and skill parameters.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the model.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('logistic_regression' or 'random_forest')
        """
        self.model_type = model_type
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
        else:
            self.model = RandomForestClassifier(random_state=42)
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
        
        # Define role mappings (for interpretation)
        self.role_mapping = {
            0: 'Developer',
            1: 'Tester',
            2: 'Release Manager',
            3: 'DevOps Engineer',
            4: 'PMO'
        }
        self.reverse_role_mapping = {v: k for k, v in self.role_mapping.items()}
    
    def preprocess_data(self, data):
        """
        Preprocess the input data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data with features and target variable
            
        Returns:
        --------
        X : numpy.ndarray
            Scaled feature matrix
        y : numpy.ndarray
            Target variable array
        """
        # Separate features and target
        X = data.drop('role_preference', axis=1)
        y = data['role_preference']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train(self, data, test_size=0.2):
        """
        Train the model on the provided data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Training data with features and target variable
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        accuracy : float
            Model accuracy on test data
        """
        X, y = self.preprocess_data(data)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate feature importance
        if self.model_type == 'random_forest':
            self.feature_importance = self.model.feature_importances_
        else:
            self.feature_importance = np.abs(self.model.coef_[0])
        
        # Evaluate on test data
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.report = classification_report(y_test, y_pred)
        self.conf_matrix = confusion_matrix(y_test, y_pred)
        
        return accuracy
    
    def predict(self, features):
        """
        Predict the role suitability for new individuals.
        
        Parameters:
        -----------
        features : pandas.DataFrame or numpy.ndarray
            Features of new individuals
            
        Returns:
        --------
        predictions : numpy.ndarray
            Predicted roles (0 for developer, 1 for tester)
        probabilities : numpy.ndarray
            Probability of each class
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet. Call train() first.")
        
        # Scale the features
        if isinstance(features, pd.DataFrame):
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        return predictions, probabilities
    
    def visualize_feature_importance(self, feature_names):
        """
        Visualize the importance of each feature in the model.
        
        Parameters:
        -----------
        feature_names : list
            Names of the features
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet. Call train() first.")
        
        # Sort features by importance
        indices = np.argsort(self.feature_importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance for Developer vs Tester Prediction')
        plt.bar(range(len(indices)), self.feature_importance[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        return plt

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic data for model demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    data : pandas.DataFrame
        Synthetic data with features and target variable
    """
    np.random.seed(42)
    
    # Define parameters and their distributions
    parameters = {
        # Cognitive abilities (Range: 0-100)
        'analytical_thinking': np.random.normal(70, 15, n_samples),
        'attention_to_detail': np.random.normal(70, 15, n_samples),
        'creativity': np.random.normal(70, 15, n_samples),
        'problem_solving': np.random.normal(70, 15, n_samples),
        'abstract_reasoning': np.random.normal(70, 15, n_samples),
        'pattern_recognition': np.random.normal(70, 15, n_samples),
        'working_memory': np.random.normal(70, 15, n_samples),
        
        # Technical skills (Range: 0-100)
        'programming_knowledge': np.random.normal(70, 15, n_samples),
        'debugging_ability': np.random.normal(70, 15, n_samples),
        'systems_thinking': np.random.normal(70, 15, n_samples),
        'technical_documentation': np.random.normal(70, 15, n_samples),
        'code_review_skill': np.random.normal(70, 15, n_samples),
        'algorithm_design': np.random.normal(70, 15, n_samples),
        'data_structure_knowledge': np.random.normal(70, 15, n_samples),
        
        # Personality traits (Range: 0-100)
        'patience': np.random.normal(70, 15, n_samples),
        'adaptability': np.random.normal(70, 15, n_samples),
        'methodical_approach': np.random.normal(70, 15, n_samples),
        'persistence': np.random.normal(70, 15, n_samples),
        'openness_to_feedback': np.random.normal(70, 15, n_samples),
        'conscientiousness': np.random.normal(70, 15, n_samples),
        'risk_tolerance': np.random.normal(70, 15, n_samples),
        'stress_tolerance': np.random.normal(70, 15, n_samples),
        
        # Soft skills (Range: 0-100)
        'communication_skills': np.random.normal(70, 15, n_samples),
        'teamwork': np.random.normal(70, 15, n_samples),
        'empathy': np.random.normal(70, 15, n_samples),
        'conflict_resolution': np.random.normal(70, 15, n_samples),
        'time_management': np.random.normal(70, 15, n_samples),
        'leadership': np.random.normal(70, 15, n_samples),
        'stakeholder_management': np.random.normal(70, 15, n_samples),
        'negotiation_skills': np.random.normal(70, 15, n_samples),
        
        # Experience factors (Range: 0-10 years, scaled to 0-100)
        'years_coding_experience': np.random.gamma(3, 1, n_samples) * 10,
        'years_testing_experience': np.random.gamma(3, 1, n_samples) * 10,
        'years_release_management': np.random.gamma(3, 1, n_samples) * 10,
        'years_devops_experience': np.random.gamma(3, 1, n_samples) * 10,
        'years_project_management': np.random.gamma(3, 1, n_samples) * 10,
        'project_complexity_experience': np.random.normal(60, 20, n_samples),
        
        # Education & aptitude test scores (Range: 0-100)
        'formal_cs_education': np.random.normal(70, 20, n_samples),
        'logical_reasoning_score': np.random.normal(70, 15, n_samples),
        'math_aptitude': np.random.normal(70, 15, n_samples),
        
        # DevOps specific skills (Range: 0-100)
        'infrastructure_knowledge': np.random.normal(70, 15, n_samples),
        'cloud_services_knowledge': np.random.normal(70, 15, n_samples),
        'automation_skills': np.random.normal(70, 15, n_samples),
        'ci_cd_knowledge': np.random.normal(70, 15, n_samples),
        'containerization_knowledge': np.random.normal(70, 15, n_samples),
        'monitoring_skills': np.random.normal(70, 15, n_samples),
        'security_knowledge': np.random.normal(70, 15, n_samples),
        
        # Release Management specific skills (Range: 0-100)
        'change_management': np.random.normal(70, 15, n_samples),
        'release_planning': np.random.normal(70, 15, n_samples),
        'deployment_coordination': np.random.normal(70, 15, n_samples),
        'risk_management': np.random.normal(70, 15, n_samples),
        'version_control_knowledge': np.random.normal(70, 15, n_samples),
        
        # PMO specific skills (Range: 0-100)
        'project_planning': np.random.normal(70, 15, n_samples),
        'resource_allocation': np.random.normal(70, 15, n_samples),
        'budget_management': np.random.normal(70, 15, n_samples),
        'reporting_skills': np.random.normal(70, 15, n_samples),
        'process_improvement': np.random.normal(70, 15, n_samples),
        'organizational_skills': np.random.normal(70, 15, n_samples),
        'strategic_thinking': np.random.normal(70, 15, n_samples),
    }
    
    # Create DataFrame
    data = pd.DataFrame(parameters)
    
    # Clip values to be between 0 and 100
    for col in data.columns:
        data[col] = np.clip(data[col], 0, 100)
    
    # Define more comprehensive rules for role preference
    # Developers tend to have higher scores in certain areas
    dev_score = (
        # Cognitive abilities
        1.5 * data['analytical_thinking'] +
        0.8 * data['attention_to_detail'] +
        1.4 * data['creativity'] +
        1.5 * data['problem_solving'] +
        1.3 * data['abstract_reasoning'] +
        1.1 * data['pattern_recognition'] +
        1.0 * data['working_memory'] +
        
        # Technical skills
        1.7 * data['programming_knowledge'] +
        1.0 * data['debugging_ability'] +
        1.3 * data['systems_thinking'] +
        0.7 * data['technical_documentation'] +
        1.2 * data['code_review_skill'] +
        1.6 * data['algorithm_design'] +
        1.5 * data['data_structure_knowledge'] +
        
        # Personality traits
        0.6 * data['patience'] +
        1.0 * data['adaptability'] +
        0.7 * data['methodical_approach'] +
        0.9 * data['persistence'] +
        0.8 * data['openness_to_feedback'] +
        0.9 * data['conscientiousness'] +
        1.2 * data['risk_tolerance'] +
        0.8 * data['stress_tolerance'] +
        
        # Soft skills
        0.8 * data['communication_skills'] +
        0.7 * data['teamwork'] +
        0.6 * data['empathy'] +
        0.7 * data['conflict_resolution'] +
        0.9 * data['time_management'] +
        
        # Experience & Education
        1.3 * data['years_coding_experience'] +
        0.4 * data['years_testing_experience'] +
        1.0 * data['project_complexity_experience'] +
        1.1 * data['formal_cs_education'] +
        1.2 * data['logical_reasoning_score'] +
        1.4 * data['math_aptitude']
    )
    
    # Testers tend to have higher scores in certain areas
    tester_score = (
        # Cognitive abilities
        1.0 * data['analytical_thinking'] +
        1.8 * data['attention_to_detail'] +
        0.9 * data['creativity'] +
        1.2 * data['problem_solving'] +
        0.9 * data['abstract_reasoning'] +
        1.4 * data['pattern_recognition'] +
        1.1 * data['working_memory'] +
        
        # Technical skills
        0.9 * data['programming_knowledge'] +
        1.7 * data['debugging_ability'] +
        1.1 * data['systems_thinking'] +
        1.3 * data['technical_documentation'] +
        1.4 * data['code_review_skill'] +
        0.7 * data['algorithm_design'] +
        0.8 * data['data_structure_knowledge'] +
        
        # Personality traits
        1.5 * data['patience'] +
        0.9 * data['adaptability'] +
        1.6 * data['methodical_approach'] +
        1.3 * data['persistence'] +
        1.1 * data['openness_to_feedback'] +
        1.4 * data['conscientiousness'] +
        0.7 * data['risk_tolerance'] +
        1.1 * data['stress_tolerance'] +
        
        # Soft skills
        1.2 * data['communication_skills'] +
        1.0 * data['teamwork'] +
        1.1 * data['empathy'] +
        1.0 * data['conflict_resolution'] +
        1.2 * data['time_management'] +
        
        # Experience & Education
        0.7 * data['years_coding_experience'] +
        1.5 * data['years_testing_experience'] +
        0.9 * data['project_complexity_experience'] +
        0.8 * data['formal_cs_education'] +
        1.0 * data['logical_reasoning_score'] +
        0.9 * data['math_aptitude']
    )
    
    # Add some noise to make the data more realistic
    noise = np.random.normal(0, 50, n_samples)
    dev_score += noise
    tester_score -= noise
    
    # Assign role preference (0 for developer, 1 for tester)
    data['role_preference'] = (tester_score > dev_score).astype(int)
    
    return data

def evaluate_model(data, model_type='random_forest'):
    """
    Evaluate the model on the provided data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to evaluate the model on
    model_type : str
        Type of model to use
        
    Returns:
    --------
    model : DevTesterAptitudeModel
        Trained model
    accuracy : float
        Model accuracy
    """
    # Create and train the model
    model = DevTesterAptitudeModel(model_type=model_type)
    accuracy = model.train(data)
    
    # Print evaluation metrics
    print(f"Model: {model_type}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(model.report)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(model.conf_matrix)
    
    # Visualize feature importance
    plt = model.visualize_feature_importance(data.drop('role_preference', axis=1).columns)
    plt.show()
    
    return model, accuracy

def predict_individual(model, individual_data):
    """
    Predict the role suitability for a new individual.
    
    Parameters:
    -----------
    model : DevTesterAptitudeModel
        Trained model
    individual_data : dict
        Dictionary with individual's parameter values
        
    Returns:
    --------
    role : str
        Predicted role ('Developer' or 'Tester')
    confidence : float
        Confidence in the prediction (probability)
    suitability_report : dict
        Detailed breakdown of suitability for each role
    """
    # Convert individual data to DataFrame
    df = pd.DataFrame([individual_data])
    
    # Make prediction
    pred, prob = model.predict(df)
    
    # Get the result
    if pred[0] == 0:
        role = 'Developer'
        confidence = prob[0][0]
    else:
        role = 'Tester'
        confidence = prob[0][1]
    
    # Create a more detailed suitability report
    suitability_report = {
        'primary_role': role,
        'confidence': confidence,
        'developer_probability': prob[0][0],
        'tester_probability': prob[0][1],
        'strength_areas': [],
        'improvement_areas': []
    }
    
    # Identify strengths and areas for improvement based on parameter thresholds
    developer_params = ['analytical_thinking', 'creativity', 'problem_solving', 
                        'programming_knowledge', 'algorithm_design', 'data_structure_knowledge',
                        'abstract_reasoning', 'risk_tolerance', 'years_coding_experience', 'math_aptitude']
    
    tester_params = ['attention_to_detail', 'patience', 'methodical_approach', 
                    'debugging_ability', 'technical_documentation', 'code_review_skill',
                    'persistence', 'conscientiousness', 'years_testing_experience', 'pattern_recognition']
    
    common_params = ['systems_thinking', 'communication_skills', 'teamwork', 
                    'stress_tolerance', 'time_management', 'openness_to_feedback']
    
    # Set thresholds for strengths and improvements
    strength_threshold = 80
    improvement_threshold = 60
    
    # Check which role is primary
    if role == 'Developer':
        # For developers, check developer params as strengths
        for param in developer_params:
            if param in individual_data and individual_data[param] >= strength_threshold:
                suitability_report['strength_areas'].append(param)
            elif param in individual_data and individual_data[param] < improvement_threshold:
                suitability_report['improvement_areas'].append(param)
                
        # Check common params
        for param in common_params:
            if param in individual_data and individual_data[param] >= strength_threshold:
                suitability_report['strength_areas'].append(param)
            elif param in individual_data and individual_data[param] < improvement_threshold:
                suitability_report['improvement_areas'].append(param)
    else:
        # For testers, check tester params as strengths
        for param in tester_params:
            if param in individual_data and individual_data[param] >= strength_threshold:
                suitability_report['strength_areas'].append(param)
            elif param in individual_data and individual_data[param] < improvement_threshold:
                suitability_report['improvement_areas'].append(param)
                
        # Check common params
        for param in common_params:
            if param in individual_data and individual_data[param] >= strength_threshold:
                suitability_report['strength_areas'].append(param)
            elif param in individual_data and individual_data[param] < improvement_threshold:
                suitability_report['improvement_areas'].append(param)
    
    return role, confidence, suitability_report

def main():
    """
    Main function to demonstrate the model.
    """
    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data(1000)
    
    # Show data summary
    print("\nData Summary:")
    print(data.describe())
    
    # Check class distribution
    print("\nClass Distribution:")
    print(data['role_preference'].value_counts())
    
    # Evaluate the model
    print("\nEvaluating model...")
    model, _ = evaluate_model(data)
    
    # Demonstrate prediction for individuals
    print("\nPrediction examples:")
    
    # Example 1: Developer-leaning profile
    dev_example = {
        'analytical_thinking': 85,
        'attention_to_detail': 65,
        'creativity': 80,
        'problem_solving': 90,
        'debugging_ability': 75,
        'communication_skills': 70,
        'programming_knowledge': 85,
        'patience': 60,
        'systems_thinking': 80,
        'adaptability': 75,
        'methodical_approach': 65,
        'persistence': 70
    }
    
    role, confidence = predict_individual(model, dev_example)
    print(f"Example 1: Predicted as {role} with {confidence:.2%} confidence")
    
    # Example 2: Tester-leaning profile
    tester_example = {
        'analytical_thinking': 70,
        'attention_to_detail': 90,
        'creativity': 65,
        'problem_solving': 75,
        'debugging_ability': 85,
        'communication_skills': 75,
        'programming_knowledge': 70,
        'patience': 85,
        'systems_thinking': 75,
        'adaptability': 70,
        'methodical_approach': 90,
        'persistence': 80
    }
    
    role, confidence = predict_individual(model, tester_example)
    print(f"Example 2: Predicted as {role} with {confidence:.2%} confidence")

if __name__ == "__main__":
    main()