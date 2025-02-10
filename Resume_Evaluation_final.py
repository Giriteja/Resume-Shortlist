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

logger = getLogger(__name__)


@dataclass
class EvaluationCriteria:
    """Defines evaluation criteria and weights"""

    weights: Dict[str, float] = None

    def __post_init__(self):
        self.weights = {
            "skills_match": 0.20,
            "experience_relevance": 0.20,
            "education_match": 0.10,
            "communication_skills": 0.10,
            "achievements": 0.15,
            "leadership_potential": 0.10,
            "cultural_fit": 0.05,
            "project_experience": 0.05,
            "technical_expertise": 0.05,
            "Companies"
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
            "skills_match": "Technical and soft skills alignment with job requirements",
            "experience_relevance": "Relevance and depth of work history",
            "education_match": "Educational qualifications fit",
            "communication_skills": "Quality of writing, presentation, and clarity",
            "achievements": "Quantifiable impacts and relevant accomplishments",
            "leadership_potential": "Evidence of leadership, initiative, and team management",
            "cultural_fit": "Values alignment and soft skills indication",
            "project_experience": "Relevant project work and deliverables",
            "technical_expertise": "Depth of technical knowledge in required areas",
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
            "skills_match": {{"score": 0.0, "justification": "explanation"}},
            "experience_relevance": {{"score": 0.0, "justification": "explanation"}},
            "education_match": {{"score": 0.0, "justification": "explanation"}},
            "communication_skills": {{"score": 0.0, "justification": "explanation"}},
            "achievements": {{"score": 0.0, "justification": "explanation"}},
            "leadership_potential": {{"score": 0.0, "justification": "explanation"}},
            "cultural_fit": {{"score": 0.0, "justification": "explanation"}},
            "project_experience": {{"score": 0.0, "justification": "explanation"}},
            "technical_expertise": {{"score": 0.0, "justification": "explanation"}}
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
                max_tokens=2000,
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
        """Format and organize results DataFrame."""
        if df.empty:
            return df

        # Extract scores from nested dictionaries
        for criterion in self.criteria.weights.keys():
            if criterion in df.columns:
                df[f"{criterion}_score"] = df[criterion].apply(
                    lambda x: float(x.get("score", 0)) if isinstance(x, dict) else 0.0
                )

        # Organize columns
        score_cols = ["total_score"] + [
            f"{c}_score" for c in self.criteria.weights.keys()
        ]
        basic_cols = ["resume_file"]

        # Arrange columns and sort by total score
        final_cols = basic_cols + score_cols
        result_df = df[final_cols].sort_values("total_score", ascending=False)

        # Round numerical columns
        for col in score_cols:
            result_df[col] = result_df[col].round(3)

        return result_df

    def generate_detailed_report(self, evaluation: Dict) -> str:
        """Generate detailed evaluation report."""
        if not evaluation:
            return "No evaluation data available."

        report = [
            f"Resume Evaluation Report: {evaluation.get('resume_file', 'Unknown')}",
            "=" * 50,
            f"\nOverall Score: {evaluation.get('total_score', 0):.2f}/1.00\n",
            "\nDetailed Criteria Evaluation:",
        ]

        for metric, weight in self.criteria.weights.items():
            metric_data = evaluation.get(metric, {})
            if isinstance(metric_data, dict):
                score = metric_data.get("score", 0)
                justification = metric_data.get(
                    "justification", "No justification provided"
                )
                report.extend(
                    [
                        f"\n{metric.replace('_', ' ').title()} (Weight: {weight*100}%)",
                        f"Score: {score:.2f}/1.00",
                        f"Analysis: {justification}",
                    ]
                )

        return "\n".join(report)


def main():
    st.set_page_config(page_title="Resume Evaluator", layout="wide")
    st.title("Resume Evaluator")

    # Get API key from environment or Streamlit secrets
    api_key = ""

    if not api_key:
        st.error(
            "Please set the ANTHROPIC_API_KEY in environment variables or Streamlit secrets."
        )
        return

    try:
        evaluator = ClaudeResumeEvaluator(api_key)
    except Exception as e:
        st.error(f"Error initializing evaluator: {str(e)}")
        return

    col1, col2 = st.columns(2)

    with col1:
        job_description_file = st.file_uploader(
            "Upload Job Description (PDF)", type="pdf"
        )

    with col2:
        resume_files = st.file_uploader(
            "Upload Resumes (PDF)", type="pdf", accept_multiple_files=True
        )

    if job_description_file and resume_files:
        if st.button("Evaluate Resumes"):
            with st.spinner("Evaluating resumes... This may take a few minutes."):
                try:
                    results_df = evaluator.evaluate_multiple_resumes(
                        resume_files, job_description_file
                    )

                    if not results_df.empty:
                        st.subheader("Evaluation Results")
                        st.dataframe(results_df, use_container_width=True)

                        st.subheader("Detailed Reports")
                        for _, row in results_df.iterrows():
                            with st.expander(f"Detailed Report - {row['resume_file']}"):
                                evaluation = {
                                    "resume_file": row["resume_file"],
                                    "total_score": row["total_score"],
                                }
                                for criterion in evaluator.criteria.weights.keys():
                                    evaluation[criterion] = {
                                        "score": row[f"{criterion}_score"]
                                    }
                                st.text(evaluator.generate_detailed_report(evaluation))
                    else:
                        st.warning(
                            "No evaluations were generated. Please check the uploaded files and try again."
                        )

                except Exception as e:
                    st.error(f"An error occurred during evaluation: {str(e)}")


if __name__ == "__main__":
    main()
