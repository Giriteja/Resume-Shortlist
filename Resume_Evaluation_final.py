import PyPDF2
import pandas as pd
from typing import List, Dict, Optional
import anthropic
import json
import os
from dataclasses import dataclass
from pathlib import Path
import streamlit as st
from logging import getLogger
import plotly.graph_objects as go
import plotly.express as px

logger = getLogger(__name__)

def get_evaluation_prompt(question_type: str, question: Dict, response: str) -> str:
    """
    Generate appropriate evaluation prompt based on question type
    """
    if question_type == 'technical':
        return f"""
        Technical Question: {question['question']}
        
        Candidate's Response: {response}
        
        Expected Answer Points:
        {chr(10).join('- ' + point for point in question['expected_answer_points'])}
        
        Please evaluate the technical response on the following parameters (score out of 10 for each):
        1. Technical Accuracy: Correctness of technical concepts and implementation details
        2. Problem Solving: Approach to solving the technical problem and methodology
        3. Code Quality: If code is involved, evaluate its efficiency, readability, and best practices
        4. Technical Depth: Understanding of underlying concepts and technical implications
        5. Implementation Feasibility: Practicality and feasibility of the proposed solution
        
        Provide scores and brief justification for each parameter, then calculate an overall score (out of 10).
        Format your response as a JSON object with these exact keys: technical_accuracy_score, problem_solving_score, code_quality_score, technical_depth_score, implementation_feasibility_score, overall_score, and feedback.
        """
    else:  # behavioral
        return f"""
        Behavioral Question: {question['question']}
        
        Candidate's Response: {response}
        
        Expected Answer Points:
        {chr(10).join('- ' + point for point in question['expected_answer_points'])}
        
        Please evaluate the behavioral response on the following parameters (score out of 10 for each):
        1. Relevance & Clarity: How well the response addresses the question and clarity of expression
        2. Experience & Skills: Demonstration of relevant experience, skills, and achievements
        3. Motivation & Enthusiasm: Level of genuine interest, motivation, and positive attitude
        4. Cultural Fit: Alignment with company values and team culture
        5. Professional Growth: Evidence of career progression and learning mindset
        
        For questions about previous experience or projects, also consider:
        - Specific role and responsibilities
        - Challenges faced and solutions implemented
        - Impact and outcomes achieved
        - Technologies and methodologies used
        
        Provide scores and brief justification for each parameter, then calculate an overall score (out of 10).
        Format your response as a JSON object with these exact keys: relevance_clarity_score, experience_skills_score, motivation_enthusiasm_score, cultural_fit_score, professional_growth_score, overall_score, and feedback.
        """

def evaluate_answer(client: anthropic.Client, question: Dict, response: str, question_type: str) -> Dict:
    """
    Evaluate a single answer using Claude API with type-specific criteria
    """
    evaluation_prompt = get_evaluation_prompt(question_type, question, response)
    
    message = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        temperature=0,
        messages=[{
            "role": "user",
            "content": evaluation_prompt
        }]
    )
    
    try:
        evaluation = message.content[0].text
        return eval(evaluation)
    except Exception as e:
        st.error(f"Error parsing Claude's response: {e}")
        if question_type == 'technical':
            return {
                "technical_accuracy_score": 0,
                "problem_solving_score": 0,
                "code_quality_score": 0,
                "technical_depth_score": 0,
                "implementation_feasibility_score": 0,
                "overall_score": 0,
                "feedback": "Error in evaluation"
            }
        else:
            return {
                "relevance_clarity_score": 0,
                "experience_skills_score": 0,
                "motivation_enthusiasm_score": 0,
                "cultural_fit_score": 0,
                "professional_growth_score": 0,
                "overall_score": 0,
                "feedback": "Error in evaluation"
            }


class TestGenerator:
    def __init__(self, client):
        self.client = client

    @st.cache_data
    def generate_technical_questions(_self, domain: str, role_description: str) -> List[Dict]:
        prompt = f"""Generate 5 technical interview questions for the {domain} domain.
        Role context: {role_description}
        
        Return the questions in this JSON format:
        {{
            "questions": [
                {{
                    "question": "technical question here",
                    "type": "technical",
                    "expected_answer_points": ["key point 1", "key point 2", "key point 3"]
                }}
            ]
        }}
        
        Make questions specific to the domain and varying in difficulty."""

        try:
            response = _self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            return json.loads(response.content[0].text)["questions"]
        except Exception as e:
            st.error(f"Error generating technical questions: {str(e)}")
            return []

    def get_behavioral_questions(_self) -> List[Dict]:
        return [
            {
                "question": "Tell me about yourself",
                "type": "behavioral",
                "expected_answer_points": ["background", "relevant experience", "key achievements", "career goals"]
            },
            {
                "question": "Why do you want to join our company?",
                "type": "behavioral",
                "expected_answer_points": ["company values alignment", "growth opportunities", "industry interest", "specific company achievements"]
            },
            {
                "question": "Where do you see yourself in 5 years?",
                "type": "behavioral",
                "expected_answer_points": ["career progression", "skill development", "leadership goals", "industry impact"]
            }
        ]

def show_online_test(domain: str, role_description: str, candidate_name: str):
    st.title("Online Assessment")
    st.write(f"Candidate: {candidate_name}")
    st.write(f"Domain: {domain}")
    
    # Initialize test generator
    test_generator = TestGenerator(anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"]))
    
    # Generate questions
    technical_questions = test_generator.generate_technical_questions(domain, role_description)
    behavioral_questions = test_generator.get_behavioral_questions()
    
    # Combine all questions
    all_questions = technical_questions + behavioral_questions
    
    # Create a form for the test
    with st.form("online_test_form"):
        responses = {}
        
        st.subheader("Technical Questions")
        for i, q in enumerate(technical_questions, 1):
            st.write(f"\n{i}. {q['question']}")
            responses[f"technical_{i}"] = st.text_area(
                f"Your answer for question {i}",
                key=f"tech_{i}",
                height=150,
                label_visibility="collapsed"
            )
        
        st.subheader("Behavioral Questions")
        for i, q in enumerate(behavioral_questions, 1):
            st.write(f"\n{i}. {q['question']}")
            responses[f"behavioral_{i}"] = st.text_area(
                f"Your answer for behavioral question {i}",
                key=f"beh_{i}",
                height=150,
                label_visibility="collapsed"
            )
        
        submitted = st.form_submit_button("Submit Test")
        if submitted:
            # Save responses
            if 'test_responses' not in st.session_state:
                st.session_state.test_responses = {}
            st.session_state.test_responses[candidate_name] = {
                'responses': responses,
                'questions': all_questions
            }
            st.success("Test submitted successfully!")
            st.balloons()

# Add these functions to your main code
def show_test_results(candidate_name: str, client: anthropic.Client):
    """
    Display test results with type-specific evaluation scores
    """
    if 'test_responses' in st.session_state and candidate_name in st.session_state.test_responses:
        st.subheader("Test Responses and Evaluation")
        responses = st.session_state.test_responses[candidate_name]
        
        total_score = 0
        question_count = 0
        
        for q_type in ['technical', 'behavioral']:
            st.write(f"\n{q_type.title()} Questions:")
            questions = [q for q in responses['questions'] if q['type'] == q_type]
            
            for i, question in enumerate(questions, 1):
                with st.expander(f"Question {i}: {question['question']}"):
                    response_key = f"{q_type}_{i}"
                    response = responses['responses'][response_key]
                    
                    st.write("Response:")
                    st.write(response)
                    
                    st.write("\nExpected Answer Points:")
                    for point in question['expected_answer_points']:
                        st.write(f"- {point}")
                    
                    if 'evaluations' not in responses:
                        responses['evaluations'] = {}
                    
                    if response_key not in responses['evaluations']:
                        with st.spinner("Evaluating response..."):
                            evaluation = evaluate_answer(client, question, response, q_type)
                            responses['evaluations'][response_key] = evaluation
                    else:
                        evaluation = responses['evaluations'][response_key]
                    
                    st.write("\nEvaluation Scores:")
                    col1, col2, col3 = st.columns(3)
                    
                    if q_type == 'technical':
                        with col1:
                            st.metric("Technical Accuracy", f"{evaluation['technical_accuracy_score']}/10")
                            st.metric("Problem Solving", f"{evaluation['problem_solving_score']}/10")
                        with col2:
                            st.metric("Code Quality", f"{evaluation['code_quality_score']}/10")
                            st.metric("Technical Depth", f"{evaluation['technical_depth_score']}/10")
                        with col3:
                            st.metric("Implementation Feasibility", f"{evaluation['implementation_feasibility_score']}/10")
                    else:  # behavioral
                        with col1:
                            st.metric("Relevance & Clarity", f"{evaluation['relevance_clarity_score']}/10")
                            st.metric("Experience & Skills", f"{evaluation['experience_skills_score']}/10")
                        with col2:
                            st.metric("Motivation & Enthusiasm", f"{evaluation['motivation_enthusiasm_score']}/10")
                            st.metric("Cultural Fit", f"{evaluation['cultural_fit_score']}/10")
                        with col3:
                            st.metric("Professional Growth", f"{evaluation['professional_growth_score']}/10")
                    
                    st.metric("Overall Score", f"{evaluation['overall_score']}/10")
                    st.write("\nDetailed Feedback:")
                    st.write(evaluation['feedback'])
                    
                    total_score += evaluation['overall_score']
                    question_count += 1
        
        if question_count > 0:
            final_score = total_score / question_count
            st.subheader("Final Evaluation")
            st.metric("Average Score Across All Questions", f"{final_score:.1f}/10")
            
        st.session_state.test_responses[candidate_name] = responses

class DynamicDomainEvaluator:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.domain_metrics = {}
        self.selected_metrics = {}
        self.weights = {}

    @st.cache_data
    def get_domain_metrics(_self, domain: str, role_description: str = "") -> Dict[str, str]:
        """Get metrics for any domain using Claude - cached version"""
        prompt = f"""As an expert HR professional, generate relevant evaluation metrics for the {domain} domain.
        Additional role context: {role_description}
        
        Generate 8-10 key evaluation metrics that are most important for this domain. 
        Each metric should have a name and detailed description.
        
        Return the metrics in this exact JSON format:
        {{
            "metric_name": "detailed description",
            "metric_name2": "detailed description2"
        }}
        
        The metric names should be in snake_case and descriptions should be comprehensive.
        Focus on domain-specific technical and professional competencies."""

        try:
            response = _self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            metrics = json.loads(response.content[0].text)
            return metrics
        except Exception as e:
            st.error(f"Error generating metrics: {str(e)}")
            return {}

    def set_evaluation_metrics(self, metrics: Dict[str, float]):
        """Set the metrics and weights for evaluation"""
        self.selected_metrics = {k: v for k, v in metrics.items() if v > 0}
        total_weight = sum(self.selected_metrics.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.selected_metrics.items()}
        else:
            self.weights = {}

class ClaudeResumeEvaluator:
    def __init__(self, api_key: str, domain_metrics: Dict[str, str] = None, weights: Dict[str, float] = None):
        """Initialize the resume evaluator with API key and domain metrics."""
        if not api_key:
            raise ValueError("API key cannot be empty")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.domain_metrics = domain_metrics or {}
        self.weights = weights or {}

    @staticmethod
    def extract_text_from_pdf(pdf_file) -> Optional[str]:
        """Extract text content from PDF file."""
        if pdf_file is None:
            return None
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            return "\n".join(text)
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            return None

    def _create_evaluation_prompt(self, resume_text: str, job_description: str) -> str:
        """Create the evaluation prompt for Claude using dynamic metrics."""
        metric_format = ','.join(f'"{metric}": {{"score": 0.0, "justification": "explanation"}}' 
                               for metric in self.domain_metrics.keys())
        
        return f"""As an expert HR professional, evaluate this resume against the job description.
        For each criterion, provide a numerical score between 0.0 and 1.0 and a brief justification.
        
        Job Description:
        {job_description}

        Resume:
        {resume_text}

        Criteria to evaluate:
        {json.dumps(self.domain_metrics, indent=2)}

        Return your evaluation in this exact JSON format:
        {{
            {metric_format}
        }}
        """

    def get_claude_evaluation(self, resume_text: str, job_description: str) -> Optional[Dict]:
        """Get evaluation from Claude API."""
        if not resume_text or not job_description:
            return None

        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0,
                system="You are an expert HR professional. Provide objective, evidence-based evaluations.",
                messages=[
                    {
                        "role": "user",
                        "content": self._create_evaluation_prompt(resume_text, job_description),
                    }
                ],
            )

            try:
                evaluation = json.loads(response.content[0].text)
                if not all(field in evaluation for field in self.domain_metrics.keys()):
                    logger.error("Missing required fields in Claude's response")
                    return None
                return evaluation
            except json.JSONDecodeError:
                logger.error("Failed to parse Claude's response as JSON")
                return None

        except Exception as e:
            logger.error(f"Error getting Claude evaluation: {str(e)}")
            return None

    def calculate_total_score(self, evaluation: Dict) -> float:
        """Calculate weighted total score from evaluation results."""
        if not evaluation:
            return 0.0

        total_score = 0.0
        for criterion, weight in self.weights.items():
            try:
                criterion_data = evaluation.get(criterion, {})
                if isinstance(criterion_data, dict):
                    score = float(criterion_data.get("score", 0.0))
                    total_score += score * weight
            except (TypeError, ValueError) as e:
                logger.error(f"Error calculating score for {criterion}: {str(e)}")
                continue

        return round(total_score, 3)

    def evaluate_resume(self, resume_file, job_description: str) -> Optional[Dict]:
        """Evaluate a single resume."""
        resume_text = self.extract_text_from_pdf(resume_file)
        if not resume_text:
            return None

        evaluation = self.get_claude_evaluation(resume_text, job_description)
        if evaluation:
            evaluation["total_score"] = self.calculate_total_score(evaluation)
            evaluation["resume_file"] = getattr(resume_file, "name", "Unknown")
            return evaluation
        return None

    def evaluate_multiple_resumes(self, resume_files: List, job_description_file) -> pd.DataFrame:
        """Evaluate multiple resumes."""
        if not resume_files or not job_description_file:
            return pd.DataFrame()

        job_description = self.extract_text_from_pdf(job_description_file)
        if not job_description:
            return pd.DataFrame()

        results = []
        for resume_file in resume_files:
            result = self.evaluate_resume(resume_file, job_description)
            if result:
                results.append(result)

        if not results:
            return pd.DataFrame()

        return self._format_results(pd.DataFrame(results))

    def _format_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format and organize results DataFrame including both scores and justifications."""
        if df.empty:
            return df
        
        # Extract scores and justifications from nested dictionaries
        for criterion in self.domain_metrics.keys():
            if criterion in df.columns:
                df[f"{criterion}_score"] = df[criterion].apply(
                    lambda x: float(x.get("score", 0)) if isinstance(x, dict) else 0.0
                )
                df[f"{criterion}_justification"] = df[criterion].apply(
                    lambda x: x.get("justification", "") if isinstance(x, dict) else ""
                )
        
        # Organize columns
        score_cols = ["total_score"] + [f"{c}_score" for c in self.domain_metrics.keys()]
        justification_cols = [f"{c}_justification" for c in self.domain_metrics.keys()]
        basic_cols = ["resume_file"]
        
        # Arrange columns and sort by total score
        final_cols = basic_cols + score_cols + justification_cols
        result_df = df[final_cols].sort_values("total_score", ascending=False)
        
        # Round numerical columns
        for col in score_cols:
            result_df[col] = result_df[col].round(3)
        
        return result_df
        
def main():
    st.set_page_config(page_title="Dynamic Resume Evaluator", layout="wide")
    st.title("Dynamic Resume Evaluator")

    # Initialize session state variables if they don't exist
    if 'show_test' not in st.session_state:
        st.session_state.show_test = False
    if 'test_candidate' not in st.session_state:
        st.session_state.test_candidate = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Summary"

    api_key = st.secrets["ANTHROPIC_API_KEY"]
    
    if not api_key:
        st.error("Please set the ANTHROPIC_API_KEY in environment variables or Streamlit secrets.")
        return

    domain_evaluator = DynamicDomainEvaluator(api_key)

    # Domain input
    col1, col2 = st.columns(2)
    with col1:
        domain = st.text_input(
            "Enter Job Domain",
            help="e.g., 'Software Engineering', 'Data Science', 'Digital Marketing'"
        )
    with col2:
        role_description = st.text_area(
            "Role Description (Optional)",
            help="Add specific details about the role to get more targeted evaluation metrics"
        )

    # Generate metrics when domain is entered
    if domain:
        domain_metrics = domain_evaluator.get_domain_metrics(domain, role_description)
            
        if domain_metrics:
            st.subheader("Customize Evaluation Metrics")
            st.info("Adjust weights (0-1) for each metric. Set to 0 to exclude a metric.")
            
            # Display metrics in two columns
            col1, col2 = st.columns(2)
            metric_weights = {}
            
            for i, (metric, description) in enumerate(domain_metrics.items()):
                with col1 if i < len(domain_metrics)/2 else col2:
                    weight = st.slider(
                        f"{metric.replace('_', ' ').title()}",
                        0.0,
                        1.0,
                        0.5,
                        0.1,
                        help=description
                    )
                    metric_weights[metric] = weight

            # Set evaluation metrics
            domain_evaluator.set_evaluation_metrics(metric_weights)

            # File uploaders
            st.subheader("Upload Documents")
            col1, col2 = st.columns(2)
            with col1:
                job_description_file = st.file_uploader(
                    "Upload Job Description (PDF)",
                    type="pdf"
                )
            with col2:
                resume_files = st.file_uploader(
                    "Upload Resumes (PDF)",
                    type=['pdf'],
                    accept_multiple_files=True
                )

            if job_description_file and resume_files and any(metric_weights.values()):
                if st.button("Evaluate Resumes"):
                    # Create resume evaluator with domain metrics and weights
                    resume_evaluator = ClaudeResumeEvaluator(
                        api_key,
                        domain_metrics=domain_metrics,
                        weights=domain_evaluator.weights
                    )
                    
                    with st.spinner("Evaluating resumes... This may take a few minutes."):
                        try:
                            results_df = resume_evaluator.evaluate_multiple_resumes(
                                resume_files,
                                job_description_file
                            )
                            
                            if not results_df.empty:
                                # Store results in session state
                                st.session_state.results_df = results_df
                                st.session_state.domain = domain
                                st.session_state.role_description = role_description

                        except Exception as e:
                            st.error(f"An error occurred during evaluation: {str(e)}")

            # Check if we have results to display
            if hasattr(st.session_state, 'results_df'):
                # Create tabs
                tabs = ["Summary", "Detailed Analysis", "Online Test"]
                tab1, tab2, tab3 = st.tabs(tabs)

                with tab1:
                    st.dataframe(
                        st.session_state.results_df[[col for col in st.session_state.results_df.columns 
                                  if not col.endswith('_justification')]],
                        use_container_width=True
                    )
                    
                    # Add online test section
                    st.subheader("Online Test Section")
                    selected_candidate = st.selectbox(
                        "Select candidate for test",
                        st.session_state.results_df['resume_file'].tolist()
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Take Test"):
                            st.session_state.show_test = True
                            st.session_state.test_candidate = selected_candidate
                            st.session_state.current_tab = "Online Test"
                            st.rerun()
                            
                    with col2:
                        if st.button("View Results"):
                            if 'test_responses' in st.session_state and selected_candidate in st.session_state.test_responses:
                                show_test_results(selected_candidate,api_key)
                            else:
                                st.warning("No test results available for this candidate yet.")
                
                with tab2:
                    for _, row in st.session_state.results_df.iterrows():
                        with st.expander(f"Detailed Analysis - {row['resume_file']}"):
                            for metric in domain_metrics:
                                st.subheader(metric.replace('_', ' ').title())
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    st.metric(
                                        "Score",
                                        f"{row[f'{metric}_score']:.2f}"
                                    )
                                with col2:
                                    st.write(row[f'{metric}_justification'])
                
                with tab3:
                    if st.session_state.show_test:
                        show_online_test(
                            st.session_state.domain,
                            st.session_state.role_description,
                            st.session_state.test_candidate
                        )
                    else:
                        st.info("Please select a candidate and click 'Take Test' in the Summary tab to start the assessment.")

if __name__ == "__main__":
    main()
