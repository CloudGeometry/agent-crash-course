from typing import List, Dict, Any, TypedDict, Annotated, Callable
from langgraph.graph import Graph, StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo")


# Define the state type
class AgentState(TypedDict):
    hazard: str
    solution: str
    is_valid: bool
    validation_feedback: str
    report: str
    attempts: int


class SummaryState(TypedDict):
    hazard_analyses: List[Dict[str, Any]]
    final_report: str


# Define the hazard generation tool
@tool
def hazard_generation_tool(tool_input: str = "") -> str:
    """Tool that generates a realistic hazard for a squirrel trying to steal an acorn.
    
    Args:
        tool_input: Not used, but required by the tool interface.
    
    Returns:
        str: A realistic hazard statement that a squirrel might face while trying to steal an acorn.
    """
    prompt = """Generate a realistic hazard that a squirrel might face while trying to steal an acorn.
    Consider:
    - Environmental factors (weather, terrain)
    - Predators and competitors
    - Physical challenges
    - Human-related obstacles
    
    Make it specific and realistic.
    Return only the hazard statement."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def analyze_hazard(state: AgentState) -> AgentState:
    """Analyze the hazard and generate a low-tech solution."""
    # If this is the first attempt, generate a new hazard using the tool
    if state.get("attempts", 0) == 0:
        state["hazard"] = hazard_generation_tool.invoke("")
    
    hazard = state["hazard"]
    attempts = state.get("attempts", 0)
    previous_feedback = state.get("validation_feedback", "")
    
    # Create a prompt for the LLM
    prompt = f"""Given the following hazard for a squirrel trying to steal an acorn:
    {hazard}
    
    {f'Previous attempt feedback: {previous_feedback}' if attempts > 0 else ''}
    
    Provide a simple, low-tech solution that a squirrel could implement to mitigate this risk.
    Focus on natural, easily accessible solutions that don't require human technology.
    Keep the response concise and practical.
    
    {'IMPORTANT: This is a revision attempt. Please address the previous feedback.' if attempts > 0 else ''}"""
    
    # Get response from LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Update state with solution
    state["solution"] = response.content
    state["attempts"] = attempts + 1
    return state


def validate_solution(state: AgentState) -> AgentState:
    """Validate the proposed solution and provide feedback."""
    hazard = state["hazard"]
    solution = state["solution"]
    
    # Create a prompt for the LLM
    prompt = f"""Review this solution for a squirrel facing the following hazard:
    Hazard: {hazard}
    Proposed Solution: {solution}
    
    Evaluate if this solution is:
    1. Truly low-tech (no human technology)
    2. Realistically implementable by a squirrel
    3. Safe and effective
    
    IMPORTANT: Respond in exactly this format:
    VALID: [true/false]
    FEEDBACK: [your feedback here]
    
    Consider a solution valid if it meets these criteria:
    - Uses only natural materials or squirrel-accessible resources
    - Is something a squirrel could reasonably do
    - Would effectively address the hazard
    - Doesn't require human intervention or technology
    
    Be generous in your validation - if the solution is mostly good but needs minor adjustments, consider it valid."""
    
    # Get response from LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()
    
    # Parse the response more robustly
    try:
        # Split by newlines and find the VALID line
        lines = response_text.split('\n')
        valid_line = next(line for line in lines if line.strip().startswith('VALID:'))
        feedback_line = next(line for line in lines if line.strip().startswith('FEEDBACK:'))
        
        # Extract the boolean value
        is_valid = 'true' in valid_line.lower()
        
        # Extract the feedback
        feedback = feedback_line.split('FEEDBACK:')[-1].strip()
    except Exception:
        # Fallback parsing if the format isn't exactly as expected
        is_valid = 'true' in response_text.lower() and 'false' not in response_text.lower()
        feedback = response_text.split('FEEDBACK:')[-1].strip() if 'FEEDBACK:' in response_text else response_text
    
    # Update state
    state["is_valid"] = is_valid
    state["validation_feedback"] = feedback
    return state


def should_retry(state: AgentState) -> str:
    """Determine if we should retry the analysis based on validation result."""
    if state["is_valid"]:
        return "generate_report"
    elif state["attempts"] < 3:  # Limit to 3 attempts
        return "analyze_hazard"
    else:
        return "generate_report"  # Give up after 3 attempts


def generate_report(state: AgentState) -> AgentState:
    """Generate a structured report of the hazard analysis and solution."""
    hazard = state["hazard"]
    solution = state["solution"]
    is_valid = state["is_valid"]
    feedback = state["validation_feedback"]
    attempts = state["attempts"]
    
    # Create a prompt for the LLM
    prompt = f"""Create a structured report for a squirrel's hazard mitigation plan:

    HAZARD: {hazard}
    PROPOSED SOLUTION: {solution}
    VALIDATION STATUS: {'✓ Valid' if is_valid else '✗ Invalid'}
    VALIDATION FEEDBACK: {feedback}
    ATTEMPTS: {attempts}

    Format the report as a clear, concise summary that:
    1. Highlights the key points
    2. Uses bullet points for clarity
    3. Includes a final recommendation
    4. Is written in a friendly, encouraging tone
    
    Keep it brief but informative."""
    
    # Get response from LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Update state with report
    state["report"] = response.content
    return state


def create_workflow() -> Graph:
    """Create the LangGraph workflow."""
    # Create a new graph
    workflow = StateGraph(AgentState)
    
    # Add the nodes
    workflow.add_node("analyze_hazard", analyze_hazard)
    workflow.add_node("validate_solution", validate_solution)
    workflow.add_node("generate_report", generate_report)
    
    # Define the edges
    workflow.add_edge("analyze_hazard", "validate_solution")
    workflow.add_conditional_edges(
        "validate_solution",
        should_retry,
        {
            "analyze_hazard": "analyze_hazard",
            "generate_report": "generate_report"
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("analyze_hazard")
    
    # Set the exit point
    workflow.set_finish_point("generate_report")
    
    return workflow.compile()


def generate_final_summary(hazard_analyses: List[Dict[str, Any]]) -> str:
    """Generate a comprehensive summary of all hazards and solutions."""
    # Create a prompt for the LLM
    prompt = f"""Create a comprehensive summary report for a squirrel's hazard mitigation strategies.
    The report should cover all the following hazard analyses:

    {chr(10).join([f'''
    HAZARD {i+1}:
    - Hazard: {analysis['hazard']}
    - Solution: {analysis['solution']}
    - Validation: {'✓ Valid' if analysis['is_valid'] else '✗ Invalid'}
    - Feedback: {analysis['validation_feedback']}
    - Attempts: {analysis['attempts']}
    ''' for i, analysis in enumerate(hazard_analyses)])}

    Format the report to include:
    1. An executive summary of all hazards and solutions
    2. A detailed analysis of each hazard and its solution
    3. Common themes or patterns across the solutions
    4. Overall recommendations for the squirrel
    5. A risk assessment matrix (high/medium/low) for each hazard
    
    Use clear headings, bullet points, and a friendly, encouraging tone.
    Make it comprehensive but easy to understand."""
    
    # Get response from LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def main():
    # Create the workflow
    workflow = create_workflow()
    
    # Collect all hazard analyses
    hazard_analyses = []
    
    # Process 5 hazards
    for _ in range(5):
        # Initialize state
        state = {
            "hazard": "",
            "solution": "",
            "is_valid": False,
            "validation_feedback": "",
            "report": "",
            "attempts": 0
        }
        
        # Run the workflow
        result = workflow.invoke(state)
        
        # Store the analysis
        hazard_analyses.append({
            "hazard": result["hazard"],
            "solution": result["solution"],
            "is_valid": result["is_valid"],
            "validation_feedback": result["validation_feedback"],
            "attempts": result["attempts"]
        })
    
    # Generate and print the final comprehensive report
    print("\n" + "="*80)
    print("COMPREHENSIVE SQUIRREL HAZARD MITIGATION REPORT")
    print("="*80)
    print(generate_final_summary(hazard_analyses))
    print("="*80)


if __name__ == "__main__":
    main()