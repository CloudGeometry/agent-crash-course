from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages

class SquirrelAgentStateNoTools(TypedDict):
    messages: Annotated[List, add_messages]
    llm_generated_solution: str

from langchain_openai import ChatOpenAI # Or your preferred LLM provider

# Ensure your API key is set (e.g., OPENAI_API_KEY)
# llm_no_tools = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7) # Temperature a bit higher for creativity
# For this example, let's mock the LLM response for clarity if you don't have an API key set up
# In a real scenario, you would use an actual LLM.

try:
    # Using a model known for instruction following.
    # Temperature might be slightly higher to encourage creative, yet relevant, solutions.
    llm_no_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, max_tokens=300)
    print("LLM for 'No Tools' scenario initialized successfully.")
except ImportError:
    print("langchain_openai not installed. Install it with 'pip install langchain-openai'")
    llm_no_tools = None
except Exception as e:
    print(f"Could not initialize LLM for 'No Tools' scenario: {e}. Using a placeholder.")
    llm_no_tools = None

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def squirrel_strategist_node_no_tools(state: SquirrelAgentStateNoTools):
    print("\n--- Squirrel Strategist Node (No Tools) Called ---")
    current_messages = state['messages']
    hazard_description = ""

    if not current_messages or not isinstance(current_messages[-1], HumanMessage):
        return {"messages": [AIMessage(content="What's the acorn hazard, little buddy? (No Tools mode)")]}

    hazard_description = current_messages[-1].content

    if llm_no_tools is None:
        print("LLM (no tools) not available. Using fallback response.")
        solution = "My brain's a bit fuzzy for direct advice now (LLM not configured for 'no tools')."
        return {
            "messages": current_messages + [AIMessage(content=solution)],
            "llm_generated_solution": solution
        }

    print(f"Asking the wise squirrel spirit (LLM - no tools) about: {hazard_description}")

    # Craft a detailed prompt to guide the LLM
    system_prompt_template = """You are a wise old squirrel, an expert in survival and outsmarting hazards when trying to secure acorns.
A younger squirrel has come to you with a problem.
Your task is to provide a creative, low-tech, and practical solution that a squirrel could realistically implement.
Focus on natural squirrel abilities (climbing, speed, agility, observation, digging, camouflage, using the environment) and simple tricks.
Do NOT suggest using any human tools, complex multi-squirrel coordinated plans unless very simple, or abilities squirrels don't possess.
The solution should be actionable and specific to the hazard.

Think step-by-step for your plan. For example:
1. Assess the situation (e.g., distance to threat, escape routes).
2. Describe the core tactic (e.g., distraction, stealth, quick grab).
3. Detail the execution.
4. Mention a quick getaway.
"""
    
    prompt_messages = [
        SystemMessage(content=system_prompt_template),
        HumanMessage(content=f"The hazard is: '{hazard_description}'. What's your low-tech advice?")
    ]

    ai_response = llm_no_tools.invoke(prompt_messages)
    print(f"LLM (No Tools) Response: {ai_response.content}")

    solution = ai_response.content
    
    return {
        "messages": current_messages + [AIMessage(content=solution)],
        "llm_generated_solution": solution
    }

from langgraph.graph import StateGraph, END, START

workflow_no_tools = StateGraph(SquirrelAgentStateNoTools)
workflow_no_tools.add_node("squirrel_strategist_no_tools", squirrel_strategist_node_no_tools)
workflow_no_tools.add_edge(START, "squirrel_strategist_no_tools")
workflow_no_tools.add_edge("squirrel_strategist_no_tools", END)

app_no_tools = workflow_no_tools.compile()
print("\n--- Squirrel Agent Graph (No Tools) Compiled ---")

if llm_no_tools: # Only run if LLM was initialized
    print("\n--- Running Squirrel Agent (No Tools Version) ---")
    hazard_no_tools = "A human is having a picnic right under the best acorn tree!"
    inputs_no_tools = {"messages": [HumanMessage(content=hazard_no_tools)]}

    print(f"\nInvoking 'No Tools' agent with hazard: '{hazard_no_tools}'")
    final_state_no_tools = app_no_tools.invoke(inputs_no_tools, {"recursion_limit": 3}) # Recursion limit

    print("\n--- 'No Tools' Agent Run Complete ---")
    print(f"\nLLM-Generated Solution for '{hazard_no_tools}':")
    print(final_state_no_tools['llm_generated_solution'])
    if final_state_no_tools.get('messages'):
        print(f"\nFinal AI Message (No Tools): {final_state_no_tools['messages'][-1].content}")

else:
    print("\nLLM for 'No Tools' scenario not initialized. Skipping 'No Tools' agent run.")

