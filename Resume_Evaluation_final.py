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


class DynamicDomainEvaluator:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.domain_metrics = {}
        self.selected_metrics = {}
        self.weights = {}

    def get_domain_metrics(self, domain: str, role_description: str = "") -> Dict[str, str]:
        """Get metrics for any domain using Claude"""
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
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            metrics = json.loads(response.content[0].text)
            self.domain_metrics = metrics
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

    def _create_evaluation_prompt(self, resume_text: str, job_description: str) -> str:
        """Create domain-specific evaluation prompt"""
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
            {','.join(f'"{metric}": {{"score": 0.0, "justification": "explanation"}}' for metric in self.selected_metrics.keys())}
        }}

        Provide specific evidence from the resume to justify each score.
        """


@dataclass
class EvaluationCriteria:
    """Defines evaluation criteria and weights"""

    weights: Dict[str, float] = None

    def __post_init__(self):
        self.weights = {
            "subject_expertise": 0.25,        # Critical importance for content accuracy and standards alignment
            "pedagogical_knowledge": 0.20,    # Essential for understanding effective educational approaches
            "content_development": 0.15,      # Key for evaluating and creating materials
            "educational_background": 0.10,   # Formal qualifications and credentials
            "assessment_expertise": 0.10,     # Important for evaluating learning outcomes
            "digital_literacy": 0.05,         # Necessary for modern content formats
            "differentiation_skills": 0.05,   # Valuable for inclusive content
            "curriculum_alignment": 0.05,     # Important for standards compliance
            "review_experience": 0.05         # Previous experience in similar role
        }


class ClaudeResumeEvaluator:
    def __init__(self, api_key: str):
        """Initialize the resume evaluator with API key."""
        if not api_key:
            raise ValueError("API key cannot be empty")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.criteria = EvaluationCriteria()

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
        """Create the evaluation prompt for Claude."""
        criteria_descriptions = {
    "subject_expertise": "Deep knowledge and understanding of K-12 curriculum standards, learning objectives, and subject matter across relevant grade levels. Demonstrated ability to evaluate content accuracy and age-appropriateness.",
    
    "pedagogical_knowledge": "Understanding of effective teaching methodologies, learning theories, and instructional design principles. Ability to assess content structure, scaffolding, and differentiation strategies.",
    
    "content_development": "Experience in creating, reviewing, or adapting educational materials for K-12 students. Proven track record of developing engaging, standards-aligned content.",
    
    "educational_background": "Relevant academic qualifications in education, curriculum development, or subject-specific areas. Teaching credentials or certifications are valued.",
    
    "assessment_expertise": "Skills in evaluating learning outcomes, creating assessment items, and understanding various assessment formats suitable for K-12 learners.",
    
    "digital_literacy": "Proficiency in reviewing and preparing content for multiple formats including digital platforms, 3D models review and feedback,interactive materials, and multimedia resources.",
    
    "differentiation_skills": "Ability to review and suggest modifications for diverse learners, including special education, gifted students, and English language learners.",
    
    "curriculum_alignment": "Experience in mapping content to state/national standards,NCERT sylabus ensuring vertical alignment across grade levels, and maintaining scope and sequence.",
    
    "review_experience": "Track record of providing constructive feedback, maintaining quality standards, and collaborating with content development teams."
}

        return f"""As an expert HR professional, evaluate this resume against the job description.
        For each criterion, provide a numerical score between 0.0 and 1.0 and a brief justification.
        
        Job Description:
        {job_description}

        Resume:
        {resume_text}

        Criteria to evaluate:
        {json.dumps(criteria_descriptions, indent=2)}

        Return your evaluation in this exact JSON format:
        {{
            "subject_expertise" : {{"score": 0.0, "justification": "explanation"}},
            "pedagogical_knowledge": {{"score": 0.0, "justification": "explanation"}},
            "content_development": {{"score": 0.0, "justification": "explanation"}},
            "educational_background": {{"score": 0.0, "justification": "explanation"}},
            "communication_skills": {{"score": 0.0, "justification": "explanation"}},
            "assessment_expertise": {{"score": 0.0, "justification": "explanation"}},
            "digital_literacy": {{"score": 0.0, "justification": "explanation"}},
            "differentiation_skills": {{"score": 0.0, "justification": "explanation"}},
            "curriculum_alignment": {{"score": 0.0, "justification": "explanation"}},
            "review_experience": {{"score": 0.0, "justification": "explanation"}}
        }}
        """

    def get_claude_evaluation(
        self, resume_text: str, job_description: str
    ) -> Optional[Dict]:
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
                        "content": self._create_evaluation_prompt(
                            resume_text, job_description
                        ),
                    }
                ],
            )

            # Extract the JSON response and validate it
            try:
                #st.write(json.loads(response.content[0].text))
                evaluation = json.loads(response.content[0].text)
                # Validate that all required fields are present
                required_fields = set(self.criteria.weights.keys())
                if not all(field in evaluation for field in required_fields):
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
        for criterion, weight in self.criteria.weights.items():
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

    def evaluate_multiple_resumes(
        self, resume_files: List, job_description_file
    ) -> pd.DataFrame:
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
        for criterion in self.criteria.weights.keys():
            if criterion in df.columns:
                # Extract scores
                df[f"{criterion}_score"] = df[criterion].apply(
                    lambda x: float(x.get("score", 0)) if isinstance(x, dict) else 0.0
                )
                # Extract justifications
                df[f"{criterion}_justification"] = df[criterion].apply(
                    lambda x: x.get("justification", "") if isinstance(x, dict) else ""
                )
    
        # Organize columns
        score_cols = ["total_score"] + [
            f"{c}_score" for c in self.criteria.weights.keys()
        ]
        justification_cols = [
            f"{c}_justification" for c in self.criteria.weights.keys()
        ]
        basic_cols = ["resume_file"]
    
        # Arrange columns and sort by total score
        final_cols = basic_cols + score_cols + justification_cols
        result_df = df[final_cols].sort_values("total_score", ascending=False)
    
        # Round numerical columns
        for col in score_cols:
            result_df[col] = result_df[col].round(3)
    
        return result_df
    
    def generate_detailed_report(self, evaluation: Dict) -> str:
        """Generate detailed evaluation report with scores and justifications."""
        if not evaluation:
            return "No evaluation data available."
    
        report = [
            f"Resume Evaluation Report: {evaluation.get('resume_file', 'Unknown')}",
            "=" * 50,
            f"\nOverall Score: {evaluation.get('total_score', 0):.2f}/1.00\n",
            "\nDetailed Criteria Evaluation:",
        ]
    
        for metric, weight in self.criteria.weights.items():
            score = evaluation.get(f"{metric}_score", 0)
            justification = evaluation.get(f"{metric}_justification", "No justification provided")
            
            report.extend(
                [
                    f"\n{metric.replace('_', ' ').title()} (Weight: {weight*100}%)",
                    f"Score: {score:.2f}/1.00",
                    f"Analysis: {justification}",
                ]
            )
    
        return "\n".join(report)

    def create_analytics_dashboard(self,results_df: pd.DataFrame):
        """Create an analytics dashboard for resume evaluation results."""
        st.subheader("Analytics Dashboard")
        
        # Layout with tabs
        tab1, tab2, tab3 = st.tabs(["Overview", "Detailed Analysis", "Comparative View"])
        
        with tab1:
            # Overview section
            col1, col2 = st.columns(2)
            
            with col1:
                # Average scores by criterion
                st.subheader("Average Scores by Criterion")
                score_cols = [col for col in results_df.columns if col.endswith('_score') and col != 'total_score']
                avg_scores = results_df[score_cols].mean().round(3)
                
                # Create a bar chart for average scores
                chart_data = pd.DataFrame({
                    'Criterion': [col.replace('_score', '').replace('_', ' ').title() for col in score_cols],
                    'Average Score': avg_scores.values
                })
                
                st.bar_chart(chart_data.set_index('Criterion'))
            
            with col2:
                # Distribution of total scores
                st.subheader("Total Score Distribution")
                fig_hist = {
                    'data': [{
                        'type': 'histogram',
                        'x': results_df['total_score'],
                        'nbinsx': 10,
                        'name': 'Total Scores'
                    }],
                    'layout': {
                        'xaxis': {'title': 'Score'},
                        'yaxis': {'title': 'Count'}
                    }
                }
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab2:
            # Detailed analysis section
            st.subheader("Individual Criteria Breakdown")
            
            # Select candidate for detailed view
            selected_candidate = st.selectbox(
                "Select Resume to Analyze",
                results_df['resume_file'].tolist()
            )
            
            if selected_candidate:
                candidate_data = results_df[results_df['resume_file'] == selected_candidate].iloc[0]
                
                # Radar chart for candidate scores
                score_cols = [col for col in results_df.columns if col.endswith('_score') and col != 'total_score']
                
                fig_radar = {
                    'data': [{
                        'type': 'scatterpolar',
                        'r': [candidate_data[col] for col in score_cols],
                        'theta': [col.replace('_score', '').replace('_', ' ').title() for col in score_cols],
                        'fill': 'toself',
                        'name': selected_candidate
                    }],
                    'layout': {
                        'polar': {'radialaxis': {'range': [0, 1]}},
                        'showlegend': True
                    }
                }
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Display justifications in an organized way
                st.subheader("Detailed Justifications")
                for criterion in score_cols:
                    base_criterion = criterion.replace('_score', '')
                    with st.expander(f"{base_criterion.replace('_', ' ').title()}"):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("Score", f"{candidate_data[criterion]:.2f}")
                        with col2:
                            st.write(candidate_data[f"{base_criterion}_justification"])
        
        with tab3:
            # Comparative analysis section
            st.subheader("Candidate Comparison")
            
            # Multi-select for candidates
            selected_candidates = st.multiselect(
                "Select Candidates to Compare",
                results_df['resume_file'].tolist(),
                default=results_df['resume_file'].tolist()[:3]  # Default to first 3
            )
            
            if selected_candidates:
                comparison_data = results_df[results_df['resume_file'].isin(selected_candidates)]
                
                # Create comparison chart
                fig_comparison = {
                    'data': [{
                        'type': 'bar',
                        'x': comparison_data['resume_file'],
                        'y': comparison_data[col],
                        'name': col.replace('_score', '').replace('_', ' ').title()
                    } for col in score_cols],
                    'layout': {
                        'barmode': 'group',
                        'xaxis': {'title': 'Candidates'},
                        'yaxis': {'title': 'Scores', 'range': [0, 1]}
                    }
                }
                st.plotly_chart(fig_comparison, use_container_width=True)


def main():
    st.set_page_config(page_title="Dynamic Resume Evaluator", layout="wide")
    st.title("Dynamic Resume Evaluator")

    api_key = st.secrets["ANTHROPIC_API_KEY"]
    
    if not api_key:
        st.error("Please set the ANTHROPIC_API_KEY in environment variables or Streamlit secrets.")
        return

    evaluator = DynamicDomainEvaluator(api_key)

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
        with st.spinner("Generating domain-specific evaluation metrics..."):
            domain_metrics = evaluator.get_domain_metrics(domain, role_description)
            
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
                evaluator.set_evaluation_metrics(metric_weights)

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
                        evaluator = ClaudeResumeEvaluator(api_key)
                        with st.spinner("Evaluating resumes... This may take a few minutes."):
                            try:
                                results_df = evaluator.evaluate_multiple_resumes(
                                    resume_files,
                                    job_description_file
                                )
                                
                                if not results_df.empty:
                                    # Create tabs for different views
                                    tab1, tab2, tab3 = st.tabs([
                                        "Summary Scores",
                                        "Detailed Analysis",
                                        "Comparative View"
                                    ])
                                    
                                    with tab1:
                                        st.dataframe(
                                            results_df[[col for col in results_df.columns 
                                                      if not col.endswith('_justification')]],
                                            use_container_width=True
                                        )
                                    
                                    with tab2:
                                        # Detailed analysis for each resume
                                        for _, row in results_df.iterrows():
                                            with st.expander(f"Detailed Analysis - {row['resume_file']}"):
                                                for metric in evaluator.selected_metrics:
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
                                        # Radar chart for comparison
                                        score_cols = [col for col in results_df.columns 
                                                    if col.endswith('_score') 
                                                    and col != 'total_score']
                                        
                                        fig = go.Figure()
                                        for _, row in results_df.iterrows():
                                            fig.add_trace(go.Scatterpolar(
                                                r=[row[col] for col in score_cols],
                                                theta=[col.replace('_score', '')
                                                      .replace('_', ' ').title() 
                                                      for col in score_cols],
                                                fill='toself',
                                                name=row['resume_file']
                                            ))
                                        
                                        fig.update_layout(
                                            polar=dict(radialaxis=dict(range=[0, 1])),
                                            showlegend=True
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                            except Exception as e:
                                st.error(f"An error occurred during evaluation: {str(e)}")


if __name__ == "__main__":
    main()
