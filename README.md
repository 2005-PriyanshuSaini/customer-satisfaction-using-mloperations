# Customer Satisfaction MLOps Project

This project implements a comprehensive MLOps framework for customer satisfaction prediction, focusing on enhancing the accuracy, reliability, and actionability of customer experience insights. By integrating advanced machine learning techniques with operational best practices, this system overcomes traditional limitations in customer satisfaction analysis.

## Overview

Traditional customer satisfaction metrics like Net Promoter Score (NPS) often suffer from methodological limitations and operational challenges. This project addresses these issues by implementing a robust MLOps pipeline that handles the complete lifecycle of machine learning models from development to deployment and continuous monitoring.

The framework significantly improves prediction accuracy (from 50-60% with conventional methods to 70-85% with our approach) by implementing novel techniques such as NPS bias classification and continuous model adaptation.

## Key Features

- **Multi-Source Data Ingestion**: Automated collection of structured and unstructured data from NPS surveys, transaction logs, social media sentiment, and customer service transcripts
- **Advanced Feature Engineering**: Implementation of NPS Bias metric to segment customers into positively and negatively biased cohorts
- **Synthetic Data Generation**: Uses Gaussian copulas to create synthetic samples that maintain original data distributions
- **Automated Model Monitoring**: Continuous tracking of model performance with drift detection and automated retraining
- **Interpretability Tools**: Methods for explaining model predictions in business-relevant terms
- **Business Intelligence Integration**: Comprehensive integration with existing BI tools

## Technical Architecture

The project implements a ZenML pipeline architecture with the following components:

1. **Data Pipeline**: Handles data ingestion, validation, and preprocessing
2. **Feature Engineering**: Extracts and transforms features from raw data
3. **Model Training**: Trains and validates machine learning models
4. **Model Deployment**: Deploys models to production environments
5. **Monitoring**: Continuously evaluates model performance and detects drift

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/customer_satisfaction_mlops.git
cd customer_satisfaction_mlops

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation

```bash
python src/data_preparation.py
```

### Model Training

```bash
python src/model_training.py
```

### Model Evaluation

```bash
python src/model_evaluation.py
```

### Running the Web Application

```bash
python app.py
```

Visit `http://127.0.0.1:5001/` in your browser to access the web interface.

### Model Monitoring

```bash
python src/model_monitoring.py
```

## Dataset

This project uses the Telco Customer Churn dataset, which contains information about customers of a telecommunications company and whether they churned. The dataset includes various features such as:

- Customer demographics (gender, age, partners, dependents)
- Account information (tenure, contract type, payment method)
- Services signed up for (phone, internet, online security, etc.)
- Charges (monthly and total)

## Results and Impact

Our framework demonstrates significant improvements over traditional approaches:

- **Increased Prediction Accuracy**: From 50-60% with conventional methods to 70-85% with our approach
- **Enhanced F1-Score**: Improved to >70% across all bias categories
- **Targeted Interventions**: Different priorities identified for customer segments
  - Negatively biased cohorts prioritize call center improvements (β=0.67)
  - Positively biased cohorts prioritize network data quality (β=0.42)

## Future Directions

- **NLP Integration**: Analyzing free-text comments to uncover hidden reasons for dissatisfaction
- **Federated Learning**: Implementing cross-industry benchmarking with data privacy
- **Real-time Analytics**: Integrating edge computing for low-latency insights during customer interactions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
## License

This project is licensed under the [MIT License](LICENSE) © 2025 Priyanshu-Saini, Shiva-Gupta, Deyant-Kashyap, Uchit-Yadav, Shagun-Verma

## Acknowledgements

This project is based on research conducted by Shiva Gupta, Deyant Kashyap, Priyanshu Saini, Shagun Verma, Uchit Yadav, and Amandeep Kaur at Chandigarh University, as detailed in their paper "Customer Satisfaction using MLOps."

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/55140311/6f5d2805-b83d-4665-b12e-b3a4b758682b/Research-Paper.pdf
[2] https://github.com/Shyamkadiwar/MLOPS-END-TO-END-PROJECT-customer-satisfaction
[3] https://mlops-coding-course.fmind.dev/6.%20Sharing/6.2.%20Readme.html
[4] https://github.com/virajbhutada/telecom-customer-churn-prediction/blob/main/README.md
[5] https://mlops-coding-course.fmind.dev/7.%20Observability/4.%20Costs-KPIs.html
[6] https://github.com/mlops-guide/mlops-template/blob/main/README.md
[7] https://github.com/ayush714/customer-satisfaction-mlops
[8] https://dagshub.com/htahir1/zenfiles/pulls/93/files?page=0&path=llm-finetuning%2FREADME.md
[9] https://dagshub.com/htahir1/zenfiles/pulls/93/files?page=0&path=llm-lora-finetuning%2FREADME.md
[10] https://github.com/tarasowski/customer-satisfaction-machine-learning
[11] https://www.scribd.com/document/627199286/ML-Ops
[12] https://dagshub.com/iampraveens/Customer-Churn-Prediction-MLOps
[13] https://microsoft.github.io/azureml-ops-accelerator/4-Migrate/dstoolkit-mlops-base/
[14] https://github.com/praj2408/Customer-Satisfaction-Analysis-Project
[15] https://www.mdpi.com/2076-3417/11/19/8861
[16] https://github.com/ignaciofls/MLOps_telco
[17] https://github.com/Azure/mlops-project-template
[18] https://dagshub.com/htahir1/zenfiles/pulls/93/files?page=0&path=customer-churn%2FREADME.md
[19] https://www.linkedin.com/pulse/building-robust-churn-prediction-system-telecom-using-sdajc
[20] https://knowledge.dataiku.com/latest/solutions/retail/solution-analyze-customer-reviews.html
[21] http://www.ir.juit.ac.in:8080/jspui/bitstream/123456789/11507/1/Production-Ready%20Customer%20Satisfaction%20Prediction%20using%20MLOps.pdf

