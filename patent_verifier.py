"""
ULTIMATE Patent Verification System
CrewAI + Llama 3.1 8B + 5 Agents
FIXED: ollama/ prefix for LiteLLM
"""

from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama


def verify_patent_5_sections(patent_sections):
    """5-Agent Patent Verification with Llama 3.1 8B"""
   
    try:
        # FIXED: ollama/ prefix for LiteLLM routing
        local_llm = Ollama(
            model="ollama/llama3.1:8b",  # ✅ FIXED!
            base_url="http://localhost:11434",
            temperature=0.3,
            num_ctx=8192,
            num_predict=400,
            top_p=0.9
        )
       
        print("✅ Llama 3.1 8B initialized (128K context)")
        print("\n🤖 CrewAI 5-Agent Patent Verification")
        print("="*60)
       
        # Agent 1
        agent1 = Agent(
            role="Title & Abstract Validator",
            goal="Verify title and abstract USPTO compliance",
            backstory="USPTO patent examiner with 15 years experience.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )
       
        task1 = Task(
            description=f"""Analyze title and abstract:

TITLE: {patent_sections['title']}
ABSTRACT: {patent_sections['abstract']}

Provide:
1. Title word count (requirement: 10-15 words)
2. Title verdict: PASS/FAIL
3. Abstract word count (requirement: ≤150 words)
4. Abstract clarity: EXCELLENT/GOOD/FAIR/POOR
5. Abstract verdict: PASS/FAIL""",
            expected_output="Title and abstract analysis",
            agent=agent1
        )
       
        # Agent 2
        agent2 = Agent(
            role="Claims Analyzer",
            goal="Validate claims structure and dependencies",
            backstory="Senior patent examiner with 20 years experience.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )
       
        task2 = Task(
            description=f"""Analyze claims:

CLAIMS:
{patent_sections['claims']}

Provide:
1. Total claims count
2. Independent vs dependent claims
3. Numbering: PROPER/IMPROPER
4. Quality: EXCELLENT/GOOD/FAIR/POOR
5. Issues (if any)""",
            expected_output="Claims analysis",
            agent=agent2
        )
       
        # Agent 3
        agent3 = Agent(
            role="Background Reviewer",
            goal="Evaluate background and prior art",
            backstory="Patent attorney specializing in prior art.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )
       
        task3 = Task(
            description=f"""Review background:

BACKGROUND:
{patent_sections['background']}

Provide:
1. Technical field identified: YES/NO
2. Problem clarity: EXCELLENT/GOOD/FAIR/POOR
3. Prior art discussed: YES/NO
4. Quality: EXCELLENT/GOOD/FAIR/POOR""",
            expected_output="Background review",
            agent=agent3
        )
       
        # Agent 4
        agent4 = Agent(
            role="Summary Evaluator",
            goal="Assess summary completeness",
            backstory="Patent examiner evaluating summaries.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )
       
        task4 = Task(
            description=f"""Evaluate summary:

SUMMARY:
{patent_sections['summary']}

Provide:
1. Completeness: COMPLETE/PARTIAL/INCOMPLETE
2. Invention clear: YES/NO
3. Key features mentioned: YES/NO
4. Quality: EXCELLENT/GOOD/FAIR/POOR""",
            expected_output="Summary evaluation",
            agent=agent4
        )
       
        # Agent 5
        agent5 = Agent(
            role="Quality Judge",
            goal="Provide final assessment and score",
            backstory="Senior patent partner with 25 years experience.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )
       
        task5 = Task(
            description=f"""Final assessment:

TITLE: {patent_sections['title']}

Based on all findings, provide:

1. Consistency score: 0-100
2. Critical issues (top 3)
3. Strengths (top 2)
4. Overall quality score: 0-100
   - Title/Abstract: X/25
   - Claims: X/30
   - Background: X/20
   - Summary: X/15
   - Consistency: X/10
5. Filing ready: YES/NO
6. Priority actions (if not ready)""",
            expected_output="Final quality assessment",
            agent=agent5,
            context=[task1, task2, task3, task4]
        )
       
        crew = Crew(
            agents=[agent1, agent2, agent3, agent4, agent5],
            tasks=[task1, task2, task3, task4, task5],
            process=Process.sequential,
            verbose=True,
            memory=False,
            cache=False
        )
       
        print("✅ Created 5 agents")
        print("✅ Created 5 tasks")
        print("🚀 Running verification...\n")
       
        result = crew.kickoff()
       
        print("\n✅ Complete!")
        print("="*60)
       
        return f"""
╔═══════════════════════════════════════════════════════════╗
║     5-AGENT PATENT VERIFICATION REPORT                    ║
╚═══════════════════════════════════════════════════════════╝

{result}

╔═══════════════════════════════════════════════════════════╗
║ System: CrewAI + Llama 3.1 8B (5 Agents)                 ║
╚═══════════════════════════════════════════════════════════╝
"""
       
    except Exception as e:
        import traceback
        return f"❌ Error: {traceback.format_exc()}"


if __name__ == "__main__":
    test = {
        'title': 'Smart Agricultural Monitoring System Using IoT Sensors and Machine Learning',
        'abstract': '''A comprehensive smart agricultural monitoring system that integrates
        IoT sensors with machine learning algorithms to provide real-time monitoring and
        predictive analytics for crop management. The system comprises soil moisture sensors,
        temperature sensors, humidity sensors, and a processing unit that analyzes data.''',
        'claims': '''1. A smart agricultural monitoring system comprising:
   a) soil moisture sensors;
   b) temperature sensors;
   c) a central processing unit;
   d) a communication module; and
   e) a machine learning module.
2. The system of claim 1, wherein sensors detect moisture at multiple depths.
3. The system of claim 1, wherein the module predicts irrigation requirements.''',
        'background': '''Traditional farming relies on manual monitoring and scheduled irrigation,
        leading to water wastage. Farmers lack real-time data. Prior art includes basic sensors
        (US Patent 9,123,456) but these lack integrated machine learning.''',
        'summary': '''The invention provides a comprehensive smart monitoring system that addresses
        limitations of existing solutions. By integrating sensors with machine learning, the system
        provides accurate real-time monitoring and predictive analytics.'''
    }
   
    print("Testing 5-Agent Verification with Llama 3.1 8B...\n")
    result = verify_patent_5_sections(test)
    print(result)
