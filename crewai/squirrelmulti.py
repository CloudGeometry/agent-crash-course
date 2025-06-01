import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

# Set up LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Custom Tools

@tool
def badger_garden_intel_briefing_tool() -> str:
    """
    Provides a static intelligence briefing about Badger's Garden including entry points, hazards, and local fauna.
    """
    return """
    INTEL REPORT: BADGER'S GARDEN - GOLDEN ACORN
    ---------------------------------------------
    TARGET: Golden Acorn, located under the large, creaky oak tree in the garden's center.
    OCCUPANT: 'Grumples' the Badger. Known Napping Cycle: 13:00-16:00 hours (local time). Temperament: Extremely territorial.
    KNOWN HAZARDS:
    1. Noisy Gravel Perimeter: Surrounds the entire garden.
    2. Automated Sprinkler System: West lawn, unpredictable schedule.
    3. 'FiFi' the Poodle: Small, loud, south fence patrol. Distractions: High-pitched noises, sudden movements.
    POTENTIAL ENTRY POINTS:
    - North Fence Gap: Behind compost bin.
    - Overhanging Maple Branch: West side, into rose bushes (thorny).
    NOTES: Last structural check: 2 days ago. Location: Fort Erie, Ontario vicinity.
    ---------------------------------------------
    """

@tool
def web_search_tool(query: str) -> str:
    """
    Searches the web for current weather or other dynamic data. This demo version is static.
    """
    if "weather" in query.lower():
        return "Weather Report: Fort Erie, Ontario - Currently 18°C, partly cloudy, wind from North at 5 km/h. Chance of light showers later."
    return f"Searched for: {query}. No specific results available."

# Agents

commander_chip = Agent(
    role="Lead Planner & Intel Integrator for the Squirrel Secret Service (SSS)",
    goal="Integrate intelligence from tools and form a solid mission plan to acquire the Golden Acorn.",
    backstory=(
        "Commander Chip 'Strategy' Swiftpaw is a seasoned SSS operative, renowned for his meticulous "
        "planning and ability to synthesize complex information into actionable strategies. "
        "He trusts his custom intel sources but always cross-references with current conditions."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[badger_garden_intel_briefing_tool, web_search_tool],
    llm=llm
)

pip_squeak = Agent(
    role="Tactical Problem Solver & Resource Optimizer for SSS",
    goal="Devise creative, nature-based tactics to bypass threats and obstacles using squirrel ingenuity.",
    backstory=(
        "Pip 'Solutions' Squeak is the SSS gadget guru, though his 'gadgets' are purely natural. "
        "He can MacGyver a solution for anything using twigs, leaves, and cleverness."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

slink_stripe = Agent(
    role="Operational Sequencer & Contingency Planner for SSS",
    goal="Create an executable operational timeline for the Golden Acorn mission, with contingencies.",
    backstory=(
        "Slink 'Executioner' Stripe is the field ops master of the SSS. "
        "He turns high-level strategy into minute-by-minute plans and always prepares for the unexpected."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Tasks

task1 = Task(
    description=(
        "1. Call the `badger_garden_intel_briefing_tool` for foundational intel.\n"
        "2. Use `web_search_tool` to find short-term weather forecast for 'Fort Erie, Ontario'.\n"
        "3. Synthesize this information and identify the MOST viable mission approach "
        "(entry point, timing considering Grumples' nap from 13:00-16:00, and current weather).\n"
        "4. Outline 2-3 critical risks for that approach."
    ),
    expected_output=(
        "Report with: (a) Summary of key intel, (b) weather summary, "
        "(c) proposed mission approach, (d) 2-3 risks."
    ),
    agent=commander_chip
)

task2 = Task(
    description=(
        "Using the mission plan from Commander Chip:\n"
        "1. Propose a simple, nature-based method to overcome one physical obstacle "
        "(e.g., crossing gravel quietly, avoiding thorns).\n"
        "2. Propose a distraction or handling technique for one minor threat "
        "(like FiFi the Poodle).\n"
        "Solutions must be low-tech and squirrel-realistic."
    ),
    expected_output="List of 1-2 tactical solutions with materials and explanation.",
    agent=pip_squeak,
    context=[task1]
)

task3 = Task(
    description=(
        "Based on Commander Chip's strategy and Pip Squeak's tactics:\n"
        "Draft a minute-by-minute plan for 'Operation: Acorn Hoard' starting at 13:30 (within nap time).\n"
        "Plan must include: approach, obstacle handling, acorn grab, exfiltration.\n"
        "Also identify one possible failure during the acorn grab and give a squirrel-level contingency plan."
    ),
    expected_output="Detailed ops plan with backup step for failure at the grab point.",
    agent=slink_stripe,
    context=[task1, task2]
)

# Crew
sss_crew = Crew(
    agents=[commander_chip, pip_squeak, slink_stripe],
    tasks=[task1, task2, task3],
    process=Process.sequential,
    verbose=True
)

# Kickoff
if __name__ == "__main__":
    print("Operation: Acorn Hoard - Initiating SSS Crew... ")
    print("----------------------------------------------------")
    try:
        result = sss_crew.kickoff()
        print("\n----------------------------------------------------")
        print("Operation: Acorn Hoard - Mission Report ")
        print("----------------------------------------------------")
        print(result)
    except Exception as e:
        print(f"\nMission Aborted! Error: {e}")
