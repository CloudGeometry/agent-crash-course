import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI

@tool
def acrobatic_distraction_display(hazard_description: str) -> str:
    """
    Perform flashy tail flicks and quick, unpredictable movements to confuse a rival.
    Best for non-aggressive threats like rival squirrels or humans. Creates an opportunity
    to secure the acorn while the threat is distracted.
    """
    return (
        "SOLUTION: Initiate an 'Acrobatic Distraction Display'. Perform flashy tail flicks "
        "and quick, unpredictable movements to confuse the rival. When they are momentarily distracted, "
        "swiftly secure the acorn and make a getaway."
    )

@tool
def camouflage_and_wait(hazard_description: str) -> str:
    """
    Hide under leaves or behind cover and stay still. Wait for the threat to pass.
    Best for slow-moving threats like dogs or persistent humans.
    """
    return (
        "SOLUTION: Employ 'Camouflage and Wait'. Silently move to the closest bush or dense leaves. "
        "Stay perfectly still and observe the dog. Once the dog loses interest or moves further away, "
        "quickly and quietly retrieve the acorn."
    )

@tool
def rapid_grab_and_scurry(hazard_description: str) -> str:
    """
    Dash in quickly, grab the acorn, and scurry up the nearest tree.
    Ideal for fast, decisive action when there’s little time or cover.
    """
    return (
        "SOLUTION: Use 'Rapid Grab and Scurry'. Calculate the quickest path. "
        "Dash in, grab the acorn with lightning speed, and immediately bolt up the nearest tree "
        "before the cat can react."
    )

@tool
def decoy_drop(hazard_description: str) -> str:
    """
    Drop a decoy like a pebble to misdirect a curious threat.
    Best used for humans or non-aggressive animals.
    """
    return (
        "SOLUTION: Implement 'Decoy Drop'. Find a small pebble. As you move, "
        "'accidentally' and somewhat noisily drop the pebble a short distance away from the prize acorn, then freeze. "
        "If the human investigates the pebble, use the opportunity to grab the real acorn."
    )

# LLM with low temperature for deterministic plans
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=300)

# Define the squirrel strategist agent
squirrel_strategist = Agent(
    role="Wise Squirrel Strategist",
    goal="Provide clever, low-tech acorn survival strategies using natural squirrel abilities.",
    backstory=(
        "You are a legendary squirrel mentor known for brilliant escape tactics and sneaky strategies. "
        "You train young squirrels to navigate dangerous environments without using human tools. "
        "Your advice must always be practical, action-oriented, and believable for a squirrel."
    ),
    verbose=True,
    tools=[
        acrobatic_distraction_display,
        camouflage_and_wait,
        rapid_grab_and_scurry,
        decoy_drop,
    ],
    llm=llm
)

# Hazard input
hazard = "A big, scary dog is barking near the acorn pile!"

# Define the decision-making task
squirrel_task = Task(
    description=(
        f"The current hazard is: '{hazard}'\n\n"
        "You must choose the single best tool to use from your toolbox and then apply it.\n"
        "**IMPORTANT**: First, name the exact tool you want to use.\n"
        "Then explain why this tool is ideal for this hazard.\n"
        "Finally, execute it by calling the tool with the hazard description.\n\n"
        "**NOTE**: The tool expects a single string input in this format:\n"
        "{ \"hazard_description\": \"<insert the full hazard as a plain string>\" }\n\n"
        "Available tools:\n"
        "- acrobatic_distraction_display\n"
        "- camouflage_and_wait\n"
        "- rapid_grab_and_scurry\n"
        "- decoy_drop\n\n"
        "Avoid magic, human technology, or unrealistic abilities. Your strategy must be plausible for a clever squirrel."
    ),
    expected_output="Tool name + justification + result from tool execution.",
    agent=squirrel_strategist
)

# Build the Crew
crew = Crew(
    agents=[squirrel_strategist],
    tasks=[squirrel_task],
    process=Process.sequential
)

# Run the plan
if __name__ == "__main__":
    result = crew.kickoff()
    print(f"\n Final Tool-Based Strategy for Hazard: '{hazard}'\n")
