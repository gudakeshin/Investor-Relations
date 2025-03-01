import os
import logging
import json
import re
from typing import Dict, List, Optional, Union, Any, Tuple

# Import the OpenAI library - this is the latest version
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the available models
DEFAULT_MODEL = "gpt-4o"  # Default to OpenAI's latest model
FAST_MODEL = "gpt-3.5-turbo-0125"  # For less complex tasks

class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, role: str, model: str = DEFAULT_MODEL):
        self.name = name
        self.role = role
        self.model = model
        self.history = []
        logger.info(f"Initialized {self.name}")
        
    def add_to_history(self, message: Dict[str, str]):
        """Add a message to the conversation history."""
        self.history.append(message)
        
    def call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        """Call the OpenAI API with the given messages."""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            content = response.choices[0].message.content
            # Add the assistant's response to history
            self.add_to_history({"role": "assistant", "content": content})
            return content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
            
    def process(self, input_data: Any) -> Any:
        """Process input data and return output. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")


# 1. Input Processing Agent and Sub-agents

class InputProcessingAgent(BaseAgent):
    """Entry point for the system, handling user queries about company information."""
    
    def __init__(self):
        super().__init__(
            name="Input Processing Agent",
            role="Process user input to capture company name and initiate the workflow"
        )
        # Initialize sub-agents
        self.name_normalizer = NameNormalizationAgent()
        self.input_validator = InputValidatorAgent()
        self.disambiguator = DisambiguationAgent()
        
    def process(self, user_input: str) -> Dict[str, Any]:
        """Process user input to extract and validate company name."""
        logger.info(f"Processing input: {user_input}")
        
        # Add user input to history
        self.add_to_history({"role": "user", "content": user_input})
        
        # 1. Extract company name using LLM
        system_prompt = """
        You are an AI assistant that extracts company names from user queries.
        Extract ONLY the company name from the user's input.
        If multiple companies are mentioned, extract the primary one that seems to be the focus.
        Return ONLY the company name, nothing else.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        extracted_name = self.call_llm(messages).strip()
        logger.info(f"Extracted company name: {extracted_name}")
        
        # 2. Validate the input
        validation_result = self.input_validator.process(extracted_name)
        
        if not validation_result["is_valid"]:
            return {
                "status": "error",
                "message": validation_result["error_message"],
                "company_name": None
            }
        
        # 3. Normalize the name
        normalized_name = self.name_normalizer.process(extracted_name)
        
        # 4. Check for ambiguity
        disambiguation_needed, options = self.disambiguator.check_ambiguity(normalized_name)
        
        if disambiguation_needed:
            return {
                "status": "disambiguation_needed",
                "message": "Multiple companies match this name. Please select one:",
                "options": options,
                "original_input": normalized_name
            }
        
        return {
            "status": "success",
            "company_name": normalized_name,
            "original_input": user_input
        }
        
    def resolve_disambiguation(self, selected_option: str) -> Dict[str, Any]:
        """Handle user selection from disambiguation options."""
        return {
            "status": "success",
            "company_name": selected_option,
            "message": f"Selected company: {selected_option}"
        }


class NameNormalizationAgent(BaseAgent):
    """Standardizes company name formats (handling variations like Inc., Ltd., etc.)"""
    
    def __init__(self):
        super().__init__(
            name="Name Normalization Sub-agent",
            role="Standardize company name formats",
            model=FAST_MODEL  # Use faster model for simple tasks
        )
        
    def process(self, company_name: str) -> str:
        """Normalize company name by standardizing formats."""
        system_prompt = """
        You are a company name standardization agent. Your job is to convert company names into a standard format.
        Follow these rules:
        1. Remove unnecessary articles like 'The' at the beginning
        2. Standardize company suffix (Inc, Corp, Ltd, etc.)
        3. Remove extra spaces, punctuation or special characters
        4. Handle common abbreviations consistently
        5. Return ONLY the standardized company name, nothing else
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Standardize this company name: {company_name}"}
        ]
        
        normalized_name = self.call_llm(messages).strip()
        logger.info(f"Normalized '{company_name}' to '{normalized_name}'")
        return normalized_name


class InputValidatorAgent(BaseAgent):
    """Validates user input for proper syntax and structure."""
    
    def __init__(self):
        super().__init__(
            name="Input Validator Sub-agent",
            role="Validate user input for proper syntax and structure",
            model=FAST_MODEL
        )
        
    def process(self, company_name: str) -> Dict[str, Any]:
        """Validate company name input."""
        # Basic validation
        if not company_name or company_name.strip() == "":
            return {
                "is_valid": False,
                "error_message": "Company name cannot be empty."
            }
            
        if len(company_name) < 2:
            return {
                "is_valid": False,
                "error_message": "Company name is too short."
            }
            
        # Check for invalid characters using regex
        if re.search(r'[^\w\s.,&\'\"-]', company_name):
            return {
                "is_valid": False,
                "error_message": "Company name contains invalid characters."
            }
            
        return {
            "is_valid": True,
            "company_name": company_name
        }


class DisambiguationAgent(BaseAgent):
    """Resolves ambiguous company names or similar matches."""
    
    def __init__(self):
        super().__init__(
            name="Disambiguation Sub-agent",
            role="Resolve ambiguous company names or similar matches"
        )
        
    def check_ambiguity(self, company_name: str) -> Tuple[bool, List[str]]:
        """Check if the company name is ambiguous and might refer to multiple entities."""
        # This would typically query a database of company names
        # For demonstration, we'll simulate with a few common ambiguous names
        
        ambiguous_names = {
            "apple": ["Apple Inc. (Tech company)", "Apple Bank (Financial institution)"],
            "delta": ["Delta Air Lines", "Delta Hotels", "Delta Electronics"],
            "amazon": ["Amazon.com Inc.", "Amazon River Tours Inc."]
        }
        
        # Convert to lowercase for case-insensitive matching
        name_lower = company_name.lower()
        
        # Check if name is in our ambiguous list
        for key, options in ambiguous_names.items():
            if key in name_lower:
                logger.info(f"Ambiguity detected for '{company_name}', options: {options}")
                return True, options
                
        # No ambiguity found
        return False, []


# 2. Company Verification Agent and Sub-agents

class CompanyVerificationAgent(BaseAgent):
    """Verifies company existence and listing status."""
    
    def __init__(self):
        super().__init__(
            name="Company Verification Agent",
            role="Verify company existence and listing status"
        )
        # Initialize sub-agents
        self.bse_verifier = BSEVerificationAgent()
        self.nse_verifier = NSEVerificationAgent()
        self.status_aggregator = ListingStatusAggregatorAgent()
        
    def process(self, company_name: str) -> Dict[str, Any]:
        """Verify company existence and listing status."""
        logger.info(f"Verifying company: {company_name}")
        
        # Check BSE listing
        bse_result = self.bse_verifier.process(company_name)
        
        # Check NSE listing
        nse_result = self.nse_verifier.process(company_name)
        
        # Aggregate results
        aggregated_result = self.status_aggregator.process({
            "company_name": company_name,
            "bse_result": bse_result,
            "nse_result": nse_result
        })
        
        return aggregated_result


class BSEVerificationAgent(BaseAgent):
    """Verifies company listing on BSE."""
    
    def __init__(self):
        super().__init__(
            name="BSE Verification Sub-agent",
            role="Verify company listing on BSE",
            model=FAST_MODEL
        )
        
    def process(self, company_name: str) -> Dict[str, Any]:
        """Check if company is listed on BSE."""
        # In a real implementation, this would query the BSE API or database
        # For demonstration, we'll simulate with LLM
        
        system_prompt = """
        You are a BSE (Bombay Stock Exchange) verification agent. 
        Given a company name, determine if it's likely to be listed on the BSE.
        Return a JSON response with these fields:
        - listed: boolean indicating if the company is likely listed on BSE
        - bse_code: a simulated BSE code if listed, null if not
        - confidence: a number 0-1 indicating confidence in your answer
        
        Response should be valid JSON only, no other text.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": company_name}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            logger.info(f"BSE verification for {company_name}: {result}")
            return result
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from BSE verification: {response}")
            return {
                "listed": False,
                "bse_code": None,
                "confidence": 0,
                "error": "Failed to parse verification result"
            }


class NSEVerificationAgent(BaseAgent):
    """Verifies company listing on NSE."""
    
    def __init__(self):
        super().__init__(
            name="NSE Verification Sub-agent",
            role="Verify company listing on NSE",
            model=FAST_MODEL
        )
        
    def process(self, company_name: str) -> Dict[str, Any]:
        """Check if company is listed on NSE."""
        # In a real implementation, this would query the NSE API or database
        # For demonstration, we'll simulate with LLM
        
        system_prompt = """
        You are an NSE (National Stock Exchange) verification agent. 
        Given a company name, determine if it's likely to be listed on the NSE.
        Return a JSON response with these fields:
        - listed: boolean indicating if the company is likely listed on NSE
        - nse_code: a simulated NSE code if listed, null if not
        - confidence: a number 0-1 indicating confidence in your answer
        
        Response should be valid JSON only, no other text.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": company_name}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            logger.info(f"NSE verification for {company_name}: {result}")
            return result
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from NSE verification: {response}")
            return {
                "listed": False,
                "nse_code": None,
                "confidence": 0,
                "error": "Failed to parse verification result"
            }


class ListingStatusAggregatorAgent(BaseAgent):
    """Aggregates listing status from different exchanges."""
    
    def __init__(self):
        super().__init__(
            name="Listing Status Aggregator Sub-agent",
            role="Aggregate listing status from different exchanges",
            model=FAST_MODEL
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate listing status from different exchanges."""
        company_name = data["company_name"]
        bse_result = data["bse_result"]
        nse_result = data["nse_result"]
        
        # Determine overall listing status
        is_listed = bse_result.get("listed", False) or nse_result.get("listed", False)
        
        # Determine which exchange has higher confidence
        bse_confidence = bse_result.get("confidence", 0)
        nse_confidence = nse_result.get("confidence", 0)
        
        primary_exchange = "BSE" if bse_confidence > nse_confidence else "NSE"
        
        # Aggregate the results
        return {
            "company_name": company_name,
            "is_listed": is_listed,
            "exchanges": {
                "BSE": {
                    "listed": bse_result.get("listed", False),
                    "code": bse_result.get("bse_code")
                },
                "NSE": {
                    "listed": nse_result.get("listed", False),
                    "code": nse_result.get("nse_code")
                }
            },
            "primary_exchange": primary_exchange if is_listed else None,
            "verification_status": "verified" if is_listed else "not_found"
        }


# 3. Data Collection Agent and Sub-agents

class DataCollectionAgent(BaseAgent):
    """Collects relevant data for the verified company."""
    
    def __init__(self):
        super().__init__(
            name="Data Collection Agent",
            role="Collect relevant data for verified companies"
        )
        # Initialize sub-agents
        self.transcript_collector = QuarterlyCallTranscriptAgent()
        self.agm_collector = AGMDocumentAgent()
        self.filing_collector = RegulatoryFilingAgent()
        self.document_processor = DocumentProcessorAgent()
        
    def process(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect all relevant data for a verified company."""
        company_name = company_data["company_name"]
        logger.info(f"Collecting data for: {company_name}")
        
        # Collect data from various sources in parallel (for demonstration, we'll do sequentially)
        transcript_data = self.transcript_collector.process(company_data)
        agm_data = self.agm_collector.process(company_data)
        filing_data = self.filing_collector.process(company_data)
        
        # Combine all collected data
        all_documents = []
        all_documents.extend(transcript_data.get("documents", []))
        all_documents.extend(agm_data.get("documents", []))
        all_documents.extend(filing_data.get("documents", []))
        
        # Process all collected documents
        processed_data = self.document_processor.process({
            "company_name": company_name,
            "documents": all_documents
        })
        
        return {
            "company_name": company_name,
            "data_collection_status": "completed",
            "data": processed_data
        }


class QuarterlyCallTranscriptAgent(BaseAgent):
    """Collects quarterly earnings call transcripts."""
    
    def __init__(self):
        super().__init__(
            name="Quarterly Call Transcript Sub-agent",
            role="Collect quarterly earnings call transcripts"
        )
        
    def process(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect quarterly call transcripts for the company."""
        company_name = company_data["company_name"]
        
        # In a real implementation, this would query a database or API
        # For demonstration, we'll simulate with LLM
        
        system_prompt = """
        You are a financial data retrieval agent specializing in earnings call transcripts.
        Given a company name, simulate retrieving the most recent quarterly earnings call transcript.
        Generate a realistic but fictional transcript summary with:
        1. Call date
        2. Quarter (Q1/Q2/Q3/Q4 and year)
        3. Key participants (CEO, CFO, etc.)
        4. 3-5 key points discussed
        5. Brief summaries of opening remarks, Q&A, and closing
        
        Format as JSON with these fields.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": company_name}
        ]
        
        try:
            response = self.call_llm(messages)
            transcript_data = json.loads(response)
            
            return {
                "company_name": company_name,
                "questions": [],
                "status": "generation_error",
                "error": "Failed to generate financial questions"
            }


class RiskAssessmentQuestionAgent(BaseAgent):
    """Generates questions about risk factors and challenges."""
    
    def __init__(self):
        super().__init__(
            name="Risk Assessment Sub-agent",
            role="Generate questions about risk factors and challenges"
        )
        
    def process(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk assessment questions based on analysis results."""
        company_name = analysis_results["company_name"]
        company_data = analysis_results.get("company_data", {})
        sentiment_analysis = analysis_results.get("sentiment_analysis", {})
        
        system_prompt = """
        You are a specialist in creating risk-focused investor relations questions.
        Based on the company analysis provided, generate 3-5 insightful questions about the company's risk factors and challenges.
        
        Questions should:
        - Address key operational risks
        - Explore market and competitive threats
        - Probe regulatory or compliance challenges
        - Inquire about mitigation strategies for identified risks
        
        For each question:
        - The question text itself
        - Brief rationale explaining why this question is important
        - Data points or observations from the analysis that prompted this question
        
        Return a JSON with an array of risk assessment questions.
        """
        
        analysis_text = json.dumps({
            "company_name": company_name,
            "company_data": company_data,
            "sentiment_analysis": sentiment_analysis
        })
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analysis Results: {analysis_text}"}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            
            # Normalize the structure
            if isinstance(result, list):
                result = {"questions": result}
            elif "questions" not in result:
                result = {"questions": [result]}
                
            # Tag questions with their category
            for question in result.get("questions", []):
                question["category"] = "risk"
                
            logger.info(f"Generated {len(result.get('questions', []))} risk questions for {company_name}")
            return result
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from risk question generator: {response}")
            return {
                "company_name": company_name,
                "questions": [],
                "status": "generation_error",
                "error": "Failed to generate risk questions"
            }


class CompetitivePositionQuestionAgent(BaseAgent):
    """Generates questions about competitive positioning."""
    
    def __init__(self):
        super().__init__(
            name="Competitive Position Sub-agent",
            role="Generate questions about competitive positioning"
        )
        
    def process(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate competitive positioning questions based on analysis results."""
        company_name = analysis_results["company_name"]
        comparative_analysis = analysis_results.get("comparative_analysis", {})
        peers = analysis_results.get("peers", [])
        
        system_prompt = """
        You are a specialist in creating competitive positioning investor relations questions.
        Based on the company analysis provided, generate 3-5 insightful questions about the company's competitive position.
        
        Questions should:
        - Address market share and positioning
        - Explore competitive advantages and differentiators
        - Probe competitor threats and responses
        - Inquire about industry dynamics and disruption
        
        For each question:
        - The question text itself
        - Brief rationale explaining why this question is important
        - Data points or observations from the analysis that prompted this question
        
        Return a JSON with an array of competitive positioning questions.
        """
        
        analysis_text = json.dumps({
            "company_name": company_name,
            "comparative_analysis": comparative_analysis,
            "peers": peers
        })
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analysis Results: {analysis_text}"}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            
            # Normalize the structure
            if isinstance(result, list):
                result = {"questions": result}
            elif "questions" not in result:
                result = {"questions": [result]}
                
            # Tag questions with their category
            for question in result.get("questions", []):
                question["category"] = "competitive"
                
            logger.info(f"Generated {len(result.get('questions', []))} competitive questions for {company_name}")
            return result
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from competitive question generator: {response}")
            return {
                "company_name": company_name,
                "questions": [],
                "status": "generation_error",
                "error": "Failed to generate competitive questions"
            }


class QuestionRankingAgent(BaseAgent):
    """Ranks and selects the most insightful questions."""
    
    def __init__(self):
        super().__init__(
            name="Question Ranking Sub-agent",
            role="Rank and select the most insightful questions",
            model=FAST_MODEL
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rank questions by importance and relevance."""
        company_name = data["company_name"]
        questions = data.get("questions", [])
        
        if not questions:
            return {
                "company_name": company_name,
                "ranked_questions": [],
                "status": "no_questions_to_rank"
            }
        
        system_prompt = """
        You are a specialist in prioritizing investor relations questions.
        I'll provide a list of questions about a company. Rank them based on:
        
        1. Strategic importance
        2. Uniqueness and differentiation
        3. Potential to elicit valuable information
        4. Relevance to current market conditions
        5. Balance across categories (strategic, financial, risk, competitive)
        
        Select the top 7-10 questions total, maintaining a balance across categories.
        For each selected question, add a "rank" field (1 being highest priority).
        
        Return a JSON with an array of ranked_questions sorted by rank.
        """
        
        questions_text = json.dumps(questions)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Company: {company_name}\nQuestions: {questions_text}"}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            
            # Normalize the structure
            if isinstance(result, list):
                result = {"ranked_questions": result}
            elif "ranked_questions" not in result:
                result = {"ranked_questions": result.get("questions", [])}
                
            logger.info(f"Ranked {len(result.get('ranked_questions', []))} questions for {company_name}")
            return result
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from question ranker: {response}")
            return {
                "company_name": company_name,
                "ranked_questions": [],
                "status": "ranking_error",
                "error": "Failed to rank questions"
            }


# 7. Answer Formulation Agent and Sub-agents

class AnswerFormulationAgent(BaseAgent):
    """Formulates answers to generated questions."""
    
    def __init__(self):
        super().__init__(
            name="Answer Formulation Agent",
            role="Formulate answers to generated questions"
        )
        # Initialize sub-agents
        self.data_integrator = DataIntegrationAgent()
        self.strategic_messenger = StrategicMessagingAgent()
        self.answer_drafter = AnswerDraftingAgent()
        self.competitive_positioner = CompetitivePositioningAgent()
        self.answer_refiner = AnswerRefinementAgent()
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Formulate answers to the generated questions."""
        company_name = data["company_name"]
        questions = data.get("questions", [])
        analysis_results = data.get("analysis_results", {})
        
        logger.info(f"Formulating answers for {company_name} to {len(questions)} questions")
        
        # Process each question
        answers = []
        
        for question in questions:
            # 1. Integrate relevant data
            integrated_data = self.data_integrator.process({
                "company_name": company_name,
                "question": question,
                "analysis_results": analysis_results
            })
            
            # 2. Develop strategic messaging
            strategic_messaging = self.strategic_messenger.process({
                "company_name": company_name,
                "question": question,
                "integrated_data": integrated_data.get("integrated_data", {})
            })
            
            # 3. Draft initial answer
            draft_answer = self.answer_drafter.process({
                "company_name": company_name,
                "question": question,
                "strategic_messaging": strategic_messaging.get("messaging", {})
            })
            
            # 4. Position relative to competitors
            competitive_positioning = self.competitive_positioner.process({
                "company_name": company_name,
                "question": question,
                "draft_answer": draft_answer.get("draft_answer", ""),
                "analysis_results": analysis_results
            })
            
            # 5. Refine the final answer
            refined_answer = self.answer_refiner.process({
                "company_name": company_name,
                "question": question,
                "draft_answer": draft_answer.get("draft_answer", ""),
                "competitive_positioning": competitive_positioning.get("positioning", {})
            })
            
            # Add the final answer
            answers.append({
                "question": question.get("question", ""),
                "answer": refined_answer.get("refined_answer", ""),
                "category": question.get("category", "general"),
                "rank": question.get("rank", 999)
            })
        
        # Sort answers by rank
        answers.sort(key=lambda x: x.get("rank", 999))
        
        return {
            "company_name": company_name,
            "answers": answers,
            "answer_count": len(answers),
            "status": "completed"
        }


class DataIntegrationAgent(BaseAgent):
    """Integrates relevant data for answering a specific question."""
    
    def __init__(self):
        super().__init__(
            name="Data Integration Sub-agent",
            role="Integrate relevant data for answering a specific question",
            model=FAST_MODEL
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate relevant data for answering a specific question."""
        company_name = data["company_name"]
        question = data["question"]
        analysis_results = data.get("analysis_results", {})
        
        question_text = question.get("question", "")
        question_category = question.get("category", "general")
        
        system_prompt = """
        You are a financial data integration specialist.
        I'll provide a specific investor relations question and company analysis.
        Your task is to identify and extract all relevant data points from the analysis that would help answer this question.
        
        Pull together:
        - Directly relevant facts and figures
        - Related trends and patterns
        - Comparative data points
        - Contextual information needed to provide a complete answer
        
        Return a JSON with:
        - relevant_data: A structured collection of data points organized by source/category
        - data_gaps: Any missing information that would be helpful but isn't available
        """
        
        # Extract relevant portions of analysis based on question category
        relevant_analysis = {}
        
        if question_category == "strategic":
            relevant_analysis["topics"] = analysis_results.get("topics", [])
            relevant_analysis["trend_analysis"] = analysis_results.get("trend_analysis", {})
            
        elif question_category == "financial":
            relevant_analysis["financial_analysis"] = analysis_results.get("financial_analysis", {})
            relevant_analysis["trend_analysis"] = analysis_results.get("trend_analysis", {})
            
        elif question_category == "risk":
            relevant_analysis["sentiment_analysis"] = analysis_results.get("sentiment_analysis", {})
            relevant_analysis["company_data"] = analysis_results.get("company_data", {})
            
        elif question_category == "competitive":
            relevant_analysis["comparative_analysis"] = analysis_results.get("comparative_analysis", {})
            relevant_analysis["peers"] = analysis_results.get("peers", [])
            
        else:
            relevant_analysis = analysis_results
        
        analysis_text = json.dumps(relevant_analysis)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Company: {company_name}\nQuestion: {question_text}\nAnalysis: {analysis_text}"}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            
            logger.info(f"Integrated data for question about {company_name}")
            return {
                "question": question_text,
                "integrated_data": result,
                "status": "completed"
            }
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from data integrator: {response}")
            return {
                "question": question_text,
                "integrated_data": {},
                "status": "integration_error",
                "error": "Failed to integrate data"
            }


class StrategicMessagingAgent(BaseAgent):
    """Develops strategic messaging for answering a question."""
    
    def __init__(self):
        super().__init__(
            name="Strategic Messaging Sub-agent",
            role="Develop strategic messaging for answering a question"
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Develop strategic messaging for answering a specific question."""
        company_name = data["company_name"]
        question = data["question"]
        integrated_data = data.get("integrated_data", {})
        
        question_text = question.get("question", "")
        
        system_prompt = """
        You are an investor relations strategic messaging specialist.
        I'll provide a specific investor question and relevant integrated data.
        Your task is to develop strategic messaging to effectively answer the question.
        
        Consider:
        - Key messages to emphasize
        - Potential sensitive areas and how to address them
        - Positive framing opportunities
        - Forward-looking statements (with appropriate cautionary language)
        - Alignment with company narrative and prior communications
        
        Return a JSON with:
        - key_messages: Array of 2-4 primary messages to convey
        - sensitive_areas: Potential pitfalls to navigate carefully
        - positive_framing: How to position challenging aspects positively
        - forward_guidance: Appropriate future-oriented messaging
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Company: {company_name}\nQuestion: {question_text}\nIntegrated Data: {json.dumps(integrated_data)}"}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            
            logger.info(f"Developed strategic messaging for question about {company_name}")
            return {
                "question": question_text,
                "messaging": result,
                "status": "completed"
            }
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from strategic messenger: {response}")
            return {
                "question": question_text,
                "messaging": {},
                "status": "messaging_error",
                "error": "Failed to develop strategic messaging"
            }


class AnswerDraftingAgent(BaseAgent):
    """Drafts initial answers to questions."""
    
    def __init__(self):
        super().__init__(
            name="Answer Drafting Sub-agent",
            role="Draft initial answers to questions"
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Draft initial answer to a specific question."""
        company_name = data["company_name"]
        question = data["question"]
        strategic_messaging = data.get("strategic_messaging", {})
        
        question_text = question.get("question", "")
        
        system_prompt = """
        You are an investor relations answer drafting specialist.
        I'll provide a specific investor question and strategic messaging guidance.
        Your task is to draft a comprehensive initial answer to the question.
        
        The answer should:
        - Be clear, concise, and well-structured
        - Directly address the question
        - Incorporate the key messages from the strategic messaging
        - Balance transparency with appropriate caution
        - Use professional, confident language
        - Include appropriate data points and evidence
        - Be approximately 2-4 paragraphs in length
        
        Return a plain text answer (not JSON) that could be delivered by an IR professional.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Company: {company_name}\nQuestion: {question_text}\nStrategic Messaging: {json.dumps(strategic_messaging)}"}
        ]
        
        try:
            response = self.call_llm(messages)
            
            logger.info(f"Drafted answer for question about {company_name}")
            return {
                "question": question_text,
                "draft_answer": response,
                "status": "completed"
            }
        except Exception as e:
            logger.error(f"Error drafting answer: {e}")
            return {
                "question": question_text,
                "draft_answer": "",
                "status": "drafting_error",
                "error": "Failed to draft answer"
            }


class CompetitivePositioningAgent(BaseAgent):
    """Refines answers with competitive positioning."""
    
    def __init__(self):
        super().__init__(
            name="Competitive Positioning Sub-agent",
            role="Refine answers with competitive positioning"
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine answer with competitive positioning considerations."""
        company_name = data["company_name"]
        question = data["question"]
        draft_answer = data.get("draft_answer", "")
        analysis_results = data.get("analysis_results", {})
        
        question_text = question.get("question", "")
        
        system_prompt = """
        You are a competitive positioning specialist for investor relations.
        I'll provide a draft answer to an investor question and company analysis.
        Your task is to suggest refinements that improve how the company is positioned relative to competitors.
        
        Consider:
        - How to highlight competitive advantages
        - Appropriate competitive benchmarking
        - Market positioning statements
        - Differentiators to emphasize
        - Industry context that favors the company
        
        Return a JSON with:
        - positioning_suggestions: Specific suggestions to enhance competitive positioning
        - differentiators: Key differentiators to emphasize
        - competitive_context: Important industry context to include
        - cautions: Areas to be careful about when discussing competition
        """
        
        # Extract relevant competitive information
        competitive_analysis = analysis_results.get("comparative_analysis", {})
        peers = analysis_results.get("peers", [])
        
        competitive_context = {
            "comparative_analysis": competitive_analysis,
            "peers": peers
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Company: {company_name}\nQuestion: {question_text}\nDraft Answer: {draft_answer}\nCompetitive Context: {json.dumps(competitive_context)}"}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            
            logger.info(f"Added competitive positioning for question about {company_name}")
            return {
                "question": question_text,
                "positioning": result,
                "status": "completed"
            }
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from competitive positioner: {response}")
            return {
                "question": question_text,
                "positioning": {},
                "status": "positioning_error",
                "error": "Failed to add competitive positioning"
            }


class AnswerRefinementAgent(BaseAgent):
    """Refines and finalizes answers."""
    
    def __init__(self):
        super().__init__(
            name="Answer Refinement Sub-agent",
            role="Refine and finalize answers"
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine and finalize the answer."""
        company_name = data["company_name"]
        question = data["question"]
        draft_answer = data.get("draft_answer", "")
        competitive_positioning = data.get("competitive_positioning", {})
        
        question_text = question.get("question", "")
        
        system_prompt = """
        You are an investor relations answer refinement specialist.
        I'll provide a draft answer to an investor question and competitive positioning suggestions.
        Your task is to refine and finalize the answer for delivery.
        
        Improvements should include:
        - Incorporating competitive positioning elements
        - Improving clarity and precision
        - Ensuring the tone is confident yet measured
        - Adding any missing nuance or context
        - Polishing the language for an investor audience
        - Ensuring a logical flow and strong conclusion
        
        Return the refined answer as plain text (not JSON) ready for delivery.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Company: {company_name}\nQuestion: {question_text}\nDraft Answer: {draft_answer}\nCompetitive Positioning: {json.dumps(competitive_positioning)}"}
        ]
        
        try:
            response = self.call_llm(messages)
            
            logger.info(f"Refined answer for question about {company_name}")
            return {
                "question": question_text,
                "refined_answer": response,
                "status": "completed"
            }
        except Exception as e:
            logger.error(f"Error refining answer: {e}")
            return {
                "question": question_text,
                "refined_answer": draft_answer,  # Return the draft as fallback
                "status": "refinement_error",
                "error": "Failed to refine answer"
            }


# Main Orchestrator Class for the entire system

class InvestorRelationsAgentSystem:
    """Main orchestrator for the Investor Relations Agent System."""
    
    def __init__(self):
        """Initialize the main system and all agent components."""
        logger.info("Initializing Investor Relations Agent System")
        
        # Initialize all top-level agents
        self.input_processor = InputProcessingAgent()
        self.company_verifier = CompanyVerificationAgent()
        self.data_collector = DataCollectionAgent()
        self.peer_identifier = PeerIdentificationAgent()
        self.analyzer = AnalysisAgent()
        self.question_generator = QuestionGenerationAgent()
        self.answer_formulator = AnswerFormulationAgent()
        
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """Process a user query through the entire workflow."""
        logger.info(f"Processing query: {user_input}")
        
        # Phase 1: Process input and extract company name
        input_result = self.input_processor.process(user_input)
        
        if input_result["status"] == "error":
            return input_result
        
        if input_result["status"] == "disambiguation_needed":
            return input_result
        
        company_name = input_result["company_name"]
        
        # Phase 2: Verify company existence and listing status
        verification_result = self.company_verifier.process(company_name)
        
        if verification_result["verification_status"] == "not_found":
            return {
                "status": "error",
                "message": f"Company '{company_name}' could not be verified as a listed entity.",
                "company_name": company_name
            }
        
        # Phase 3: Collect company data
        data_collection_result = self.data_collector.process(verification_result)
        
        # Phase 4: Identify peer companies
        peer_identification_result = self.peer_identifier.process(data_collection_result)
        
        # Phase 5: Analyze collected data
        analysis_result = self.analyzer.process({
            "company_name": company_name,
            "data": data_collection_result,
            "peers": peer_identification_result.get("peers", [])
        })
        
        # Phase 6: Generate insightful questions
        questions_result = self.question_generator.process(analysis_result)
        
        # Phase 7: Formulate answers to questions
        answers_result = self.answer_formulator.process({
            "company_name": company_name,
            "questions": questions_result.get("questions", []),
            "analysis_results": analysis_result
        })
        
        # Return the final comprehensive results
        return {
            "status": "success",
            "company_name": company_name,
            "company_info": verification_result,
            "analysis": analysis_result,
            "questions_and_answers": answers_result.get("answers", [])
        }
        
    def resolve_disambiguation(self, selected_company: str) -> Dict[str, Any]:
        """Handle the disambiguation selection and continue processing."""
        disambiguation_result = self.input_processor.resolve_disambiguation(selected_company)
        
        if disambiguation_result["status"] == "success":
            # Continue with the selected company
            return self.process_query(f"Tell me about {selected_company}")
        else:
            return disambiguation_result


# Example usage for testing
def main():
    """Run a test of the IR Agent System."""
    # Load environment variables
    load_dotenv()
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please create a .env file with your OpenAI API key")
        return
    
    # Initialize the agent system
    ir_system = InvestorRelationsAgentSystem()
    
    # Example query
    test_query = "What are the investment prospects for Microsoft?"
    
    print(f"Processing query: {test_query}")
    result = ir_system.process_query(test_query)
    
    # Handle different result scenarios
    if result["status"] == "disambiguation_needed":
        print("Disambiguation needed!")
        print("Options:", result["options"])
        
        # For testing, select the first option
        selected_option = result["options"][0]
        print(f"Selecting: {selected_option}")
        
        result = ir_system.resolve_disambiguation(selected_option)
    
    if result["status"] == "success":
        print(f"Successfully processed query for {result['company_name']}")
        
        # Print a few sample Q&As
        print("\nSample Questions and Answers:")
        for i, qa in enumerate(result["questions_and_answers"][:3]):
            print(f"\nQ{i+1}: {qa['question']}")
            print(f"A{i+1}: {qa['answer'][:200]}...")
    else:
        print(f"Error: {result.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()
                "documents": [{
                    "type": "earnings_transcript",
                    "data": transcript_data
                }]
            }
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response for transcript: {response}")
            return {
                "company_name": company_name,
                "documents": [],
                "error": "Failed to retrieve transcripts"
            }


class AGMDocumentAgent(BaseAgent):
    """Collects Annual General Meeting documents."""
    
    def __init__(self):
        super().__init__(
            name="AGM Document Sub-agent",
            role="Collect Annual General Meeting documents"
        )
        
    def process(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect AGM documents for the company."""
        company_name = company_data["company_name"]
        
        # In a real implementation, this would query a database or API
        # For demonstration, we'll simulate with LLM
        
        system_prompt = """
        You are a financial document retrieval agent specializing in AGM documents.
        Given a company name, simulate retrieving the most recent Annual General Meeting details.
        Generate realistic but fictional AGM information with:
        1. Meeting date
        2. Location/venue
        3. Key agenda items (4-6 items)
        4. Key resolutions passed
        5. Brief summary of shareholder discussions
        
        Format as JSON with these fields.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": company_name}
        ]
        
        try:
            response = self.call_llm(messages)
            agm_data = json.loads(response)
            
            return {
                "company_name": company_name,
                "documents": [{
                    "type": "agm_document",
                    "data": agm_data
                }]
            }
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response for AGM: {response}")
            return {
                "company_name": company_name,
                "documents": [],
                "error": "Failed to retrieve AGM documents"
            }


class RegulatoryFilingAgent(BaseAgent):
    """Collects regulatory filings."""
    
    def __init__(self):
        super().__init__(
            name="Regulatory Filing Sub-agent",
            role="Collect regulatory filings"
        )
        
    def process(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect regulatory filings for the company."""
        company_name = company_data["company_name"]
        
        # In a real implementation, this would query a database or API
        # For demonstration, we'll simulate with LLM
        
        system_prompt = """
        You are a financial document retrieval agent specializing in regulatory filings.
        Given a company name, simulate retrieving recent regulatory filings.
        Generate 3-4 realistic but fictional filing summaries with:
        1. Filing date
        2. Filing type (annual report, quarterly results, etc.)
        3. Key financial metrics reported
        4. Any significant disclosures or changes
        
        Format as JSON with these fields, providing an array of filings.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": company_name}
        ]
        
        try:
            response = self.call_llm(messages)
            filing_data = json.loads(response)
            
            return {
                "company_name": company_name,
                "documents": [{
                    "type": "regulatory_filings",
                    "data": filing_data
                }]
            }
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response for filings: {response}")
            return {
                "company_name": company_name,
                "documents": [],
                "error": "Failed to retrieve regulatory filings"
            }


class DocumentProcessorAgent(BaseAgent):
    """Processes collected documents to extract relevant information."""
    
    def __init__(self):
        super().__init__(
            name="Document Processor Sub-agent",
            role="Process collected documents to extract relevant information"
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and organize all collected documents."""
        company_name = data["company_name"]
        documents = data["documents"]
        
        if not documents:
            return {
                "company_name": company_name,
                "processed_data": {},
                "status": "no_documents_found"
            }
        
        # Extract and organize key information from documents
        # In a real implementation, this would do more sophisticated processing
        
        system_prompt = """
        You are a financial document processing agent.
        I'll provide you with a collection of documents about a company.
        Extract and organize key information into these categories:
        1. Financial Performance: Key metrics, trends, and highlights
        2. Business Strategy: Strategic initiatives, focus areas, and plans
        3. Market Position: Competitive landscape, market share, industry trends
        4. Risk Factors: Key risks and challenges mentioned
        5. Governance: Board decisions, management changes, compensation

        Format as JSON with these categories as keys and extracted information as values.
        """
        
        # Combine all documents into a single context for the LLM
        documents_text = json.dumps(documents)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Company: {company_name}\nDocuments: {documents_text}"}
        ]
        
        try:
            response = self.call_llm(messages)
            processed_data = json.loads(response)
            
            return {
                "company_name": company_name,
                "processed_data": processed_data,
                "status": "success",
                "document_count": len(documents)
            }
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from document processor: {response}")
            return {
                "company_name": company_name,
                "processed_data": {},
                "status": "processing_error",
                "error": "Failed to process documents"
            }


# 4. Peer Identification Agent and Sub-agents

class PeerIdentificationAgent(BaseAgent):
    """Identifies peer companies for comparison."""
    
    def __init__(self):
        super().__init__(
            name="Peer Identification Agent",
            role="Identify peer companies for comparison"
        )
        # Initialize sub-agents
        self.industry_classifier = IndustryClassificationAgent()
        self.competitor_analyzer = CompetitorAnalysisAgent()
        self.similarity_scorer = SimilarityScoringAgent()
        self.peer_selector = PeerSelectionAgent()
        
    def process(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify peer companies for the given company."""
        company_name = company_data["company_name"]
        processed_data = company_data.get("data", {}).get("processed_data", {})
        
        logger.info(f"Identifying peers for: {company_name}")
        
        # 1. Classify company's industry
        industry_data = self.industry_classifier.process({
            "company_name": company_name,
            "company_data": processed_data
        })
        
        # 2. Analyze potential competitors
        competitor_data = self.competitor_analyzer.process({
            "company_name": company_name,
            "industry": industry_data.get("industry"),
            "sub_industry": industry_data.get("sub_industry")
        })
        
        # 3. Score similarity of potential peers
        similarity_data = self.similarity_scorer.process({
            "company_name": company_name,
            "potential_peers": competitor_data.get("potential_competitors", []),
            "company_data": processed_data
        })
        
        # 4. Select final peer group
        peer_selection = self.peer_selector.process({
            "company_name": company_name,
            "scored_peers": similarity_data.get("scored_peers", [])
        })
        
        return {
            "company_name": company_name,
            "industry_classification": industry_data,
            "peers": peer_selection.get("selected_peers", []),
            "peer_count": len(peer_selection.get("selected_peers", [])),
            "status": "completed"
        }


class IndustryClassificationAgent(BaseAgent):
    """Classifies company by industry."""
    
    def __init__(self):
        super().__init__(
            name="Industry Classification Sub-agent",
            role="Classify company by industry",
            model=FAST_MODEL
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify company's industry based on available data."""
        company_name = data["company_name"]
        company_data = data.get("company_data", {})
        
        system_prompt = """
        You are an industry classification specialist.
        Given a company name and data, classify its industry and sub-industry.
        Use standard industry classification systems (like GICS).
        
        Return a JSON with:
        - industry: The broad industry category
        - sub_industry: The specific sub-industry
        - confidence: A score from 0-1 indicating confidence in this classification
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Company: {company_name}\nData: {json.dumps(company_data)}"}
        ]
        
        try:
            response = self.call_llm(messages)
            industry_data = json.loads(response)
            logger.info(f"Industry classification for {company_name}: {industry_data}")
            return industry_data
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from industry classifier: {response}")
            return {
                "industry": "Unknown",
                "sub_industry": "Unknown",
                "confidence": 0,
                "error": "Failed to classify industry"
            }


class CompetitorAnalysisAgent(BaseAgent):
    """Analyzes and identifies potential competitors."""
    
    def __init__(self):
        super().__init__(
            name="Competitor Analysis Sub-agent",
            role="Analyze and identify potential competitors"
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify potential competitors based on industry classification."""
        company_name = data["company_name"]
        industry = data.get("industry", "Unknown")
        sub_industry = data.get("sub_industry", "Unknown")
        
        system_prompt = """
        You are a competitor analysis specialist.
        Given a company name, industry, and sub-industry, identify 6-8 potential competitor companies.
        Focus on companies that would be relevant peers for investor relations analysis.
        
        Return a JSON with:
        - potential_competitors: Array of company names
        - rationale: Brief explanation for each competitor's inclusion
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Company: {company_name}\nIndustry: {industry}\nSub-industry: {sub_industry}"}
        ]
        
        try:
            response = self.call_llm(messages)
            competitor_data = json.loads(response)
            logger.info(f"Identified {len(competitor_data.get('potential_competitors', []))} potential competitors for {company_name}")
            return competitor_data
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from competitor analyzer: {response}")
            return {
                "potential_competitors": [],
                "error": "Failed to identify competitors"
            }


class SimilarityScoringAgent(BaseAgent):
    """Scores similarity between the target company and potential peers."""
    
    def __init__(self):
        super().__init__(
            name="Similarity Scoring Sub-agent",
            role="Score similarity between the target company and potential peers"
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Score potential peers based on similarity to the target company."""
        company_name = data["company_name"]
        potential_peers = data.get("potential_peers", [])
        company_data = data.get("company_data", {})
        
        if not potential_peers:
            return {
                "company_name": company_name,
                "scored_peers": [],
                "status": "no_peers_to_score"
            }
        
        system_prompt = """
        You are a company similarity scoring specialist.
        For each potential peer company, estimate its similarity to the target company.
        Consider factors like:
        - Business model alignment
        - Revenue size/scale
        - Market capitalization
        - Product/service overlap
        
        For each peer, assign a similarity score from 0-100, where 100 is identical.
        Return a JSON with an array of scored_peers, each containing:
        - company_name: The peer company name
        - similarity_score: A number from 0-100
        - key_similarities: Brief notes on main similarities
        - key_differences: Brief notes on main differences
        """
        
        peers_text = json.dumps(potential_peers)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Target Company: {company_name}\nTarget Company Data: {json.dumps(company_data)}\nPotential Peers: {peers_text}"}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            
            # Ensure we have the expected structure
            if "scored_peers" not in result:
                result = {"scored_peers": result}
                
            logger.info(f"Scored {len(result.get('scored_peers', []))} peers for {company_name}")
            return result
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from similarity scorer: {response}")
            return {
                "company_name": company_name,
                "scored_peers": [],
                "status": "scoring_error",
                "error": "Failed to score peers"
            }


class PeerSelectionAgent(BaseAgent):
    """Selects the final peer group for comparison."""
    
    def __init__(self):
        super().__init__(
            name="Peer Selection Sub-agent",
            role="Select the final peer group for comparison",
            model=FAST_MODEL
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Select the final peer group based on similarity scores."""
        company_name = data["company_name"]
        scored_peers = data.get("scored_peers", [])
        
        if not scored_peers:
            return {
                "company_name": company_name,
                "selected_peers": [],
                "status": "no_peers_available"
            }
        
        # Simple selection logic: Take top 4-5 peers by similarity score
        # In a real implementation, this would be more sophisticated
        
        # Sort peers by similarity score (descending)
        sorted_peers = sorted(
            scored_peers, 
            key=lambda x: x.get("similarity_score", 0), 
            reverse=True
        )
        
        # Select top peers (4-5)
        num_peers = min(5, len(sorted_peers))
        selected_peers = sorted_peers[:num_peers]
        
        return {
            "company_name": company_name,
            "selected_peers": selected_peers,
            "peer_count": len(selected_peers),
            "status": "completed",
            "selection_method": "top_similarity_score"
        }


# 5. Analysis Agent and Sub-agents

class AnalysisAgent(BaseAgent):
    """Analyzes company data and generates insights."""
    
    def __init__(self):
        super().__init__(
            name="Analysis Agent",
            role="Analyze company data and generate insights"
        )
        # Initialize sub-agents
        self.topic_extractor = TopicExtractionAgent()
        self.financial_metric_analyzer = FinancialMetricAgent()
        self.sentiment_analyzer = SentimentAnalysisAgent()
        self.comparative_analyzer = ComparativeAnalysisAgent()
        self.trend_analyzer = TrendAnalysisAgent()
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis on company data."""
        company_name = data["company_name"]
        company_data = data.get("data", {}).get("processed_data", {})
        peers = data.get("peers", [])
        
        logger.info(f"Analyzing data for: {company_name}")
        
        # 1. Extract key topics from documents
        topics = self.topic_extractor.process({
            "company_name": company_name,
            "company_data": company_data
        })
        
        # 2. Analyze financial metrics
        financial_analysis = self.financial_metric_analyzer.process({
            "company_name": company_name,
            "company_data": company_data
        })
        
        # 3. Analyze sentiment in documents
        sentiment_analysis = self.sentiment_analyzer.process({
            "company_name": company_name,
            "company_data": company_data,
            "topics": topics.get("topics", [])
        })
        
        # 4. Perform comparative analysis with peers
        comparative_analysis = self.comparative_analyzer.process({
            "company_name": company_name,
            "company_data": company_data,
            "peers": peers
        })
        
        # 5. Analyze trends over time
        trend_analysis = self.trend_analyzer.process({
            "company_name": company_name,
            "company_data": company_data,
            "financial_metrics": financial_analysis.get("metrics", {})
        })
        
        # Combine all analyses
        analysis_results = {
            "company_name": company_name,
            "topics": topics.get("topics", []),
            "financial_analysis": financial_analysis,
            "sentiment_analysis": sentiment_analysis,
            "comparative_analysis": comparative_analysis,
            "trend_analysis": trend_analysis,
            "status": "completed"
        }
        
        return analysis_results


class TopicExtractionAgent(BaseAgent):
    """Extracts key topics from company documents."""
    
    def __init__(self):
        super().__init__(
            name="Topic Extraction Sub-agent",
            role="Extract key topics from company documents"
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key topics from company data."""
        company_name = data["company_name"]
        company_data = data.get("company_data", {})
        
        system_prompt = """
        You are a financial document topic extraction specialist.
        Analyze the provided company data and extract 5-8 key topics that are significant for investor relations.
        Topics should be specific and relevant for financial analysis.
        
        For each topic:
        - Provide a clear name/title
        - Brief description of why it's significant
        - Relevant data points or quotes supporting its importance
        
        Return a JSON with an array of topics, each containing the above information.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Company: {company_name}\nData: {json.dumps(company_data)}"}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            
            # Normalize the structure
            if isinstance(result, list):
                result = {"topics": result}
            elif "topics" not in result:
                result = {"topics": [result]}
                
            logger.info(f"Extracted {len(result.get('topics', []))} topics for {company_name}")
            return result
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from topic extractor: {response}")
            return {
                "company_name": company_name,
                "topics": [],
                "status": "extraction_error",
                "error": "Failed to extract topics"
            }


class FinancialMetricAgent(BaseAgent):
    """Analyzes financial metrics from company data."""
    
    def __init__(self):
        super().__init__(
            name="Financial Metric Sub-agent",
            role="Analyze financial metrics from company data"
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and analyze financial metrics from company data."""
        company_name = data["company_name"]
        company_data = data.get("company_data", {})
        
        system_prompt = """
        You are a financial analyst specializing in extracting and interpreting financial metrics.
        From the provided company data, extract key financial metrics and provide analysis.
        
        Identify metrics such as:
        - Revenue and growth rates
        - Profitability metrics (margins, EPS, etc.)
        - Efficiency ratios
        - Liquidity metrics
        - Leverage and debt metrics
        
        For each metric:
        - Provide the value or range
        - Brief interpretation of what it indicates
        - How it compares to prior periods (if available)
        
        Return a JSON with metrics organized by category.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Company: {company_name}\nData: {json.dumps(company_data)}"}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            
            logger.info(f"Analyzed financial metrics for {company_name}")
            return {
                "company_name": company_name,
                "metrics": result,
                "status": "completed"
            }
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from financial metric analyzer: {response}")
            return {
                "company_name": company_name,
                "metrics": {},
                "status": "analysis_error",
                "error": "Failed to analyze financial metrics"
            }


class SentimentAnalysisAgent(BaseAgent):
    """Analyzes sentiment in company documents by topic."""
    
    def __init__(self):
        super().__init__(
            name="Sentiment Analysis Sub-agent",
            role="Analyze sentiment in company documents by topic"
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment in company documents by topic."""
        company_name = data["company_name"]
        company_data = data.get("company_data", {})
        topics = data.get("topics", [])
        
        if not topics:
            return {
                "company_name": company_name,
                "sentiment_by_topic": [],
                "overall_sentiment": "neutral",
                "status": "no_topics_to_analyze"
            }
        
        system_prompt = """
        You are a sentiment analysis specialist focusing on financial documents.
        For each provided topic, analyze the sentiment expressed in the company documents.
        
        For each topic:
        - Score the sentiment from -5 (very negative) to +5 (very positive)
        - Provide a brief explanation with supporting evidence
        - Note any change in sentiment compared to previous communications (if apparent)
        
        Also provide an overall sentiment score for the company.
        
        Return a JSON with:
        - sentiment_by_topic: Array of topics with their sentiment scores and explanations
        - overall_sentiment: Overall sentiment score and summary
        """
        
        topics_text = json.dumps(topics)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Company: {company_name}\nData: {json.dumps(company_data)}\nTopics: {topics_text}"}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            
            logger.info(f"Analyzed sentiment for {company_name} across {len(topics)} topics")
            return result
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from sentiment analyzer: {response}")
            return {
                "company_name": company_name,
                "sentiment_by_topic": [],
                "overall_sentiment": "unknown",
                "status": "analysis_error",
                "error": "Failed to analyze sentiment"
            }


class ComparativeAnalysisAgent(BaseAgent):
    """Performs comparative analysis with peer companies."""
    
    def __init__(self):
        super().__init__(
            name="Comparative Analysis Sub-agent",
            role="Perform comparative analysis with peer companies"
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare company to peers and identify relative strengths/weaknesses."""
        company_name = data["company_name"]
        company_data = data.get("company_data", {})
        peers = data.get("peers", [])
        
        if not peers:
            return {
                "company_name": company_name,
                "comparative_insights": [],
                "status": "no_peers_to_compare"
            }
        
        system_prompt = """
        You are a comparative financial analysis specialist.
        Compare the target company with its peer group to identify relative strengths and weaknesses.
        
        Generate insights on:
        - Relative performance on key financial metrics
        - Competitive positioning
        - Market share and growth compared to peers
        - Strategic advantages or disadvantages
        - Areas where the company leads or lags its peers
        
        For each insight:
        - Provide a clear title/heading
        - Brief explanation with supporting data points
        - Implications for investors
        
        Return a JSON with an array of comparative insights.
        """
        
        peers_text = json.dumps(peers)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Company: {company_name}\nCompany Data: {json.dumps(company_data)}\nPeers: {peers_text}"}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            
            # Normalize the structure
            if isinstance(result, list):
                result = {"comparative_insights": result}
            elif "comparative_insights" not in result:
                result = {"comparative_insights": [result]}
                
            logger.info(f"Generated {len(result.get('comparative_insights', []))} comparative insights for {company_name}")
            return result
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from comparative analyzer: {response}")
            return {
                "company_name": company_name,
                "comparative_insights": [],
                "status": "analysis_error",
                "error": "Failed to perform comparative analysis"
            }


class TrendAnalysisAgent(BaseAgent):
    """Analyzes trends in company data over time."""
    
    def __init__(self):
        super().__init__(
            name="Trend Analysis Sub-agent",
            role="Analyze trends in company data over time"
        )
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in company data over time."""
        company_name = data["company_name"]
        company_data = data.get("company_data", {})
        financial_metrics = data.get("financial_metrics", {})
        
        system_prompt = """
        You are a financial trend analysis specialist.
        Analyze the company data to identify significant trends over time.
        
        Look for trends in:
        - Financial performance (revenue, margins, profitability)
        - Business focus and strategy
        - Market position and competitive landscape
        - Management messaging and tone
        
        For each identified trend:
        - Provide a clear name/title
        - Description of the trend with supporting evidence
        - Direction (improving, declining, stable, mixed)
        - Timeframe of the trend
        - Potential implications for the future
        
        Return a JSON with an array of identified trends.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Company: {company_name}\nCompany Data: {json.dumps(company_data)}\nFinancial Metrics: {json.dumps(financial_metrics)}"}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            
            # Normalize the structure
            if isinstance(result, list):
                result = {"trends": result}
            elif "trends" not in result:
                result = {"trends": [result]}
                
            logger.info(f"Identified {len(result.get('trends', []))} trends for {company_name}")
            return result
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from trend analyzer: {response}")
            return {
                "company_name": company_name,
                "trends": [],
                "status": "analysis_error",
                "error": "Failed to identify trends"
            }


# 6. Question Generation Agent and Sub-agents

class QuestionGenerationAgent(BaseAgent):
    """Generates insightful questions for investor relations."""
    
    def __init__(self):
        super().__init__(
            name="Question Generation Agent",
            role="Generate insightful questions for investor relations"
        )
        # Initialize sub-agents
        self.strategic_question_generator = StrategicQuestionAgent()
        self.financial_question_generator = FinancialPerformanceQuestionAgent()
        self.risk_question_generator = RiskAssessmentQuestionAgent()
        self.competitive_question_generator = CompetitivePositionQuestionAgent()
        self.question_ranker = QuestionRankingAgent()
        
    def process(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insightful questions based on analysis results."""
        company_name = analysis_results["company_name"]
        
        logger.info(f"Generating questions for: {company_name}")
        
        # 1. Generate strategic questions
        strategic_questions = self.strategic_question_generator.process(analysis_results)
        
        # 2. Generate financial performance questions
        financial_questions = self.financial_question_generator.process(analysis_results)
        
        # 3. Generate risk assessment questions
        risk_questions = self.risk_question_generator.process(analysis_results)
        
        # 4. Generate competitive position questions
        competitive_questions = self.competitive_question_generator.process(analysis_results)
        
        # Combine all questions
        all_questions = []
        all_questions.extend(strategic_questions.get("questions", []))
        all_questions.extend(financial_questions.get("questions", []))
        all_questions.extend(risk_questions.get("questions", []))
        all_questions.extend(competitive_questions.get("questions", []))
        
        # 5. Rank and select the best questions
        ranked_questions = self.question_ranker.process({
            "company_name": company_name,
            "questions": all_questions
        })
        
        return {
            "company_name": company_name,
            "questions": ranked_questions.get("ranked_questions", []),
            "question_count": len(ranked_questions.get("ranked_questions", [])),
            "status": "completed"
        }


class StrategicQuestionAgent(BaseAgent):
    """Generates questions about business strategy."""
    
    def __init__(self):
        super().__init__(
            name="Strategic Question Sub-agent",
            role="Generate questions about business strategy"
        )
        
    def process(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic questions based on analysis results."""
        company_name = analysis_results["company_name"]
        topics = analysis_results.get("topics", [])
        trend_analysis = analysis_results.get("trend_analysis", {})
        
        system_prompt = """
        You are a specialist in creating strategic investor relations questions.
        Based on the company analysis provided, generate 3-5 insightful questions about the company's business strategy.
        
        Questions should:
        - Focus on long-term strategic direction
        - Address growth initiatives and opportunities
        - Explore strategic shifts or pivots
        - Probe management's vision and priorities
        
        For each question:
        - The question text itself
        - Brief rationale explaining why this question is important
        - Data points or observations from the analysis that prompted this question
        
        Return a JSON with an array of strategic questions.
        """
        
        analysis_text = json.dumps({
            "company_name": company_name,
            "topics": topics,
            "trend_analysis": trend_analysis
        })
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analysis Results: {analysis_text}"}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            
            # Normalize the structure
            if isinstance(result, list):
                result = {"questions": result}
            elif "questions" not in result:
                result = {"questions": [result]}
                
            # Tag questions with their category
            for question in result.get("questions", []):
                question["category"] = "strategic"
                
            logger.info(f"Generated {len(result.get('questions', []))} strategic questions for {company_name}")
            return result
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from strategic question generator: {response}")
            return {
                "company_name": company_name,
                "questions": [],
                "status": "generation_error",
                "error": "Failed to generate strategic questions"
            }


class FinancialPerformanceQuestionAgent(BaseAgent):
    """Generates questions about financial performance."""
    
    def __init__(self):
        super().__init__(
            name="Financial Performance Sub-agent",
            role="Generate questions about financial performance"
        )
        
    def process(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate financial performance questions based on analysis results."""
        company_name = analysis_results["company_name"]
        financial_analysis = analysis_results.get("financial_analysis", {})
        trend_analysis = analysis_results.get("trend_analysis", {})
        
        system_prompt = """
        You are a specialist in creating financial performance investor relations questions.
        Based on the company analysis provided, generate 3-5 insightful questions about the company's financial performance.
        
        Questions should:
        - Address key financial metrics and trends
        - Explore profitability and margin developments
        - Probe revenue drivers and challenges
        - Inquire about financial outlook and guidance
        
        For each question:
        - The question text itself
        - Brief rationale explaining why this question is important
        - Data points or observations from the analysis that prompted this question
        
        Return a JSON with an array of financial performance questions.
        """
        
        analysis_text = json.dumps({
            "company_name": company_name,
            "financial_analysis": financial_analysis,
            "trend_analysis": trend_analysis
        })
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analysis Results: {analysis_text}"}
        ]
        
        try:
            response = self.call_llm(messages)
            result = json.loads(response)
            
            # Normalize the structure
            if isinstance(result, list):
                result = {"questions": result}
            elif "questions" not in result:
                result = {"questions": [result]}
                
            # Tag questions with their category
            for question in result.get("questions", []):
                question["category"] = "financial"
                
            logger.info(f"Generated {len(result.get('questions', []))} financial questions for {company_name}")
            return result
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from financial question generator: {response}")
            return {
                "company_name": company_name,