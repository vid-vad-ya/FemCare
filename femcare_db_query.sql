-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Assessments table
CREATE TABLE assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    taken_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    risk_pct FLOAT NOT NULL,
    risk_tier VARCHAR(20) NOT NULL,
    symptom_inputs JSONB NOT NULL
);

-- SHAP explanations table
CREATE TABLE shap_explanations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID REFERENCES assessments(id) ON DELETE CASCADE,
    feature_name VARCHAR(100) NOT NULL,
    shap_value FLOAT NOT NULL
);
